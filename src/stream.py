#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    Copyright (C) 2021 Triantafyllis Nikolaos

#    This file is part of Gisola.

#    Gisola is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, 
#    or any later version.

#    Gisola is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with Gisola.  If not, see <https://www.gnu.org/licenses/>.

from obspy.clients.fdsn.client import Client
from obspy.clients.filesystem.sds import Client as SDS
from obspy.clients.seedlink.basic_client import Client as SeedLink
from obspy import UTCDateTime, Stream, Trace
import multiprocessing, os.path, time, logging, operator
import warnings, math

# import local configuration to cfg variable
import config
import modules.clip as clip
import modules.snr as snr
import modules.ppsd as ppsd
import modules.ppsdoffline as ppsdoffline

# Both mouse modules are imported upfront; the active one is selected at
# runtime inside correct() based on config.cfg['Stream']['MouseAlgorithm'].
import modules.mouse  as _mouse_realmouse   # Zahradník 2026 (optional)
import modules.mouse2 as _mouse_legacy      # Vackář 2015 (default, as old Gisola)

# Stream.MouseAlgorithm: 1 = Vackar 2015 (original Gisola), 2 = Zahradnik 2026
_MOUSE_ALGORITHMS = {1: _mouse_legacy, 2: _mouse_realmouse}
# names of earlier configuration files are still accepted
_MOUSE_ALIASES = {'1': 1, '2': 2, 'legacy': 1, 'realmouse': 2}

def _mouse_requested():
    """
    Algorithm number requested by Stream.MouseAlgorithm, or None when the key is
    missing or its value is not valid.
    """
    _algo = config.cfg.get('Stream', {}).get('MouseAlgorithm')
    return _MOUSE_ALIASES.get(str(_algo).strip().lower())

def _mouse_algorithm():
    """
    Number of the mouse algorithm from Stream.MouseAlgorithm (1 or 2).
    A missing key or an unknown value gives 1 (old Gisola behaviour).
    """
    return _mouse_requested() or 1

@config.time
@config.log
def getWaveforms(save=True):
    """
    Retrieve Waveforms from SDS or FDSNWS
    """
    # get the maximum time window to retrieve data
    maxtl=max([tl for tl in config.windowRules])

    if 'ppsd' in config.cfg['Stream']['Modules']:
        starttime=config.org.time-10-3600 # half hour
    else:
        starttime=config.org.time-10

    endtime=config.org.time+math.ceil(maxtl)+1

    # wait if endtime has not come yet
    if UTCDateTime.now() <= endtime:
        time.sleep(endtime-UTCDateTime.now()+1)

    config.logger.info(('Retrieving waveforms based on inventory from {} '+ \
    'to {}').format(starttime, endtime))

    # create a list of (net,sta,loc,cha,start,end) 
    streams=[(net.code, sta.code, cha.location_code, cha.code, \
                UTCDateTime(starttime)-1, UTCDateTime(endtime)+1) \
                for net in config.inv for sta in net for cha in sta]

    # use queue priority as priority host service
    for service in config.cfg['Stream']['Service']:
        # loops will finish by two ways: 
        # i) no streams left for download 
        # ii) or no service left for requesting data

        # if no streams left for downloading, exit function
        if not streams: return 

        if service[0]=='SeedLink':
            try:
                # connect to SeedLink host
                sl=SeedLink(service[1].split(':')[0], port=int(service[1].split(':')[1]))
                config.logger.info('Connecting to SeedLink host ' +service[1])

                # make one request for all streams
                multiselect = ','.join(list(set(["%s_%s:%s%s" % (s[0], s[1], s[2], s[3][:2]+'?') for s in streams])))
                config.st+=sl._multiselect_request(multiselect, UTCDateTime(starttime)-1, UTCDateTime(endtime)+1)

            except:
                config.logger.info('Could not retrieve data from SeedLink host '+service[1])
                pass

        elif service[0]=='SDS':
            try:
                # connect to SDS host
                sds=SDS(service[1])
                config.logger.info('Connecting to SDS directory ' +service[1])

                # make one request per quad (net,sta,loc,cha)
                for s in streams:
                    try:
                        config.st+=sds.get_waveforms(*s)
                    except:
                        continue
            except:
                config.logger.info('Could not find SDS directory '+service[1])
                pass

        elif service[0]=='FDSNWS':
            try:
                # connect to host
                fdsnws = Client(service[1], eida_token=service[2])
                config.logger.info('Connecting to FDSNWS host ' +service[1])

                # make one request for all streams
                config.st+=fdsnws.get_waveforms_bulk(streams)
            except Exception as e:
                if e.__class__.__name__=='FDSNNoDataException':
                    config.logger.info('Could not retrieve data from FDSNWS host '+service[1])
                else:
                    config.logger.info(e)
                pass


        # filter streams by those that have been downloaded
        streams=list(filter(lambda s: not bool([tr for tr in config.st \
        if s[1]==tr.stats.station]), streams))

    # trim to exactly start and end time
    config.st.trim(starttime, endtime)

    # get traces with capable time length and number of samples
    config.st=Stream([tr for tr in config.st \
    if tr.stats.endtime-tr.stats.starttime >= endtime-starttime-2])

    # save waveforms
    if save:
        config.logger.info(('Saving waveforms in one mseed file in ' + \
        '{}').format(os.path.join(config.workdir,'streams.mseed')))
        config.st.write(os.path.join(config.workdir,'streams.mseed'), \
        format='MSEED')

@config.time
@config.log
def clean():
    """
    Clean waveforms by:
    * Removing waveforms with data and meta-data inconsistency
    * Removing waveforms with gaps
    * Remove waveforms with snr, clip and mouse thresholds reached
    """
    def verify(tr):
        """
        Verify consistency of data and meta-data
        """
        try:
            tr.verify()
            return True
        except:
            return False

    config.logger.info('Removing waveforms with inconsistency ' +\
    'between data and meta data')
    config.st=Stream(list(filter(verify, config.st)))

    config.logger.info('Removing waveforms with gaps')
    gaps=config.st.get_gaps()
    config.st=Stream(list(filter(lambda tr: not bool([_ for _ in gaps \
    if _[0]==tr.stats.network and _[1]==tr.stats.station and \
    _[2]==tr.stats.location and _[3]==tr.stats.channel]), config.st)))

    # create a list of Stream object, one per net,sta,loc,cha[:2]*
    streams=list(set([(tr.stats.network, tr.stats.station, \
    tr.stats.location, tr.stats.channel[:2]+'*') for tr in config.st]))

    config.logger.info('Starting waveform checks and corrections\n' + \
    'Checking for SNR, clipping and long-period disturbances (based on configuration enabling). Correcting by applying ' + \
    'linear detrend and instrument response removal')

    if 'mouse' in config.cfg['Stream']['Modules']:
        _req = config.cfg.get('Stream', {}).get('MouseAlgorithm')
        if _req is not None and _mouse_requested() is None:
            config.logger.warning('Unknown Stream.MouseAlgorithm {!r}; using 1'.format(_req))
        config.logger.info('Mouse algorithm: {}'.format(_mouse_algorithm()))

    # load the compiled clipping check once here: the forked workers inherit it
    # (otherwise every worker loads it from the Numba cache on its first station)
    if 'clip' in config.cfg['Stream']['Modules']:
        clip.warmup()
    if 'mouse' in config.cfg['Stream']['Modules'] and _mouse_algorithm() == 1:
        _mouse_legacy.warmup()

    with multiprocessing.Pool(processes=config.n_cores()) as p:
        res = p.map(correct, streams)

    # unpack (stream, timing) pairs returned by correct()
    streams_out = [r[0] for r in res]
    timings     = [r[1] for r in res]

    # reconstruct Stream (ignore None values) and get only enabled traces
    config.st = Stream([tr for st in streams_out if st
                        for tr in st if tr.enabled]).sort()

    # print sub-stage timings (average per station across workers)
    _keys  = ('snr', 'clip', 'mouse', 'remove_response')
    _n     = max(len(timings), 1)
    _mouse_label = 'mouse ({})'.format(_mouse_algorithm())
    for _k in _keys:
        _vals  = [t.get(_k, 0.0) for t in timings]
        _avg   = sum(_vals) / _n
        _label = _mouse_label if _k == 'mouse' else _k
        if _avg > 0.0:
            print('[STAGE]   {:<34s}: {:7.2f} s'.format(_label, _avg), flush=True)

    # trim to origin time
    endtime=config.org.time+math.ceil(max([tl for tl in config.windowRules]))+1
    config.st.trim(config.org.time, endtime)

# function runs for individual station/channel triplet in parallel
def warning(func):
    def inner():
        with warnings.catch_warnings(record=True) as w:
            st=func()
            if w: 
                config.logger.info(str(st) if st else ''+' '.join([x.message for x in w]))
    return inner

#@warning
def correct(stream):
    """
    Check and correct streams
    Applies per net,sta,loc,cha[:2]*
    Returns (Stream | None, timing_dict) where timing_dict holds wall-clock
    seconds spent in each sub-module for this station.
    """
    import time as _tm
    _t = {'snr': 0.0, 'clip': 0.0, 'mouse': 0.0, 'remove_response': 0.0}
    st = None
    try:
        # get specific stream and inventory of net,sta,loc,cha[:2]*
        inv=config.inv.select(network=stream[0], station=stream[1], \
        location=stream[2], channel=stream[3])

        st=config.st.select(network=stream[0], station=stream[1], \
        location=stream[2], channel=stream[3])

        # attach inventory metadata to stream
        st.attach_response(inv)

        if 'ppsd' in config.cfg['Stream']['Modules']:
            # prepare a copy of stream
            # PSD unlike other modules need the time range before the seismic event
            stPPSD=st.copy().trim(endtime=config.org.time-10)

            # trim to 10 sec before origin time
            st=st.trim(starttime=config.org.time-10)

        # rotate will always bring components to ZNE name
        st.rotate(method="->ZNE", inventory=inv, \
        components=tuple(config.cfg['Inventory']['Components']))

        if 'snr' in config.cfg['Stream']['Modules']:
            _t0 = _tm.perf_counter()
            # filter traces by SNR
            st=Stream(list(filter(snr.check, st)))
            _t['snr'] += _tm.perf_counter() - _t0

        if 'ppsdoffline' in config.cfg['Stream']['Modules']:
            # filter traces by PPSDOFFLINE
            st=Stream(list(filter(ppsdoffline.check, st)))
            # prevent Z domination in PPSD module
            # if only Z component passed, remove it
            if len(st)==1 and st[0].stats.channel[2]=='Z':
                st.remove(st.select(channel=st[0].stats.channel)[0])

        if 'ppsd' in config.cfg['Stream']['Modules']:
            # filter traces by PPSD
            stPPSD=Stream(list(filter(ppsd.check, stPPSD)))
            # filter main stream with those traces passed from PPSD
            for cha in [tr.stats.channel for tr in st]:
                if cha not in [tr.stats.channel for tr in stPPSD]:
                    st.remove(st.select(channel=cha)[0])

            # prevent Z domination in PPSD module
            # if only Z component passed, remove it
            if len(stPPSD)==1 and stPPSD[0].stats.channel[2]=='Z':
                st.remove(st.select(channel=stPPSD[0].stats.channel)[0])

        if 'clip' in config.cfg['Stream']['Modules'] or \
           'mouse' in config.cfg['Stream']['Modules']:
            _sens = st[0].stats.response.instrument_sensitivity
            if _sens is None:
                config.logger.warning(
                    'instrument_sensitivity missing for %s; station excluded', st[0].id)
                return None, _t

        if 'clip' in config.cfg['Stream']['Modules']:
            # if broadband
            if st[0].stats.response.instrument_sensitivity.input_units in \
            config.vel_types:
                _t0 = _tm.perf_counter()
                # filter traces by CLIP
                st=Stream(list(filter(lambda tr: clip.check(tr, threshold=0.98), \
                st)))
                _t['clip'] += _tm.perf_counter() - _t0

        if 'mouse' in config.cfg['Stream']['Modules']:
            # if broadband
            if st[0].stats.response.instrument_sensitivity.input_units in \
            config.vel_types:
                _t0 = _tm.perf_counter()
                # filter traces by MOUSE
                # if error occurred, False will return
                # and exception will  be raised
                _mouse = _MOUSE_ALGORITHMS[_mouse_algorithm()]
                with warnings.catch_warnings(record=True) as w:
                    for tr_id in _mouse.check(st):
                        st.remove(st.select(id=tr_id)[0])

                    if w:
                        config.logger.info('{}: {}'.format(st if st else '', w[0].message))
                _t['mouse'] += _tm.perf_counter() - _t0

        # flat signal
        st.detrend('linear')
        # remove_response
        _t0 = _tm.perf_counter()
        try:
            st.remove_response(inventory=inv,
                           output='VEL', # output units in Velocity (m/s)
                           pre_filt=(0.001, 0.002, 8, 9), # bandpass frqs (Hz)
                           zero_mean=True, # detrend(demean)
                           taper=True, # cos taper from f1 to f2 and from f3 to f4
                           taper_fraction=0.05 # percentage of tapering
                           )
        except AttributeError as _e:
            config.logger.warning('remove_response failed for %s (incomplete metadata); station excluded', stream)
            return None, _t
        _t['remove_response'] += _tm.perf_counter() - _t0

        for tr in st:
            tr.enabled=True

        return st, _t

    except:
        if st:
            config.logger.exception('Error occurred in ' + str(stream))
        return None, _t

@config.time
@config.log
def prioritize():
    """
    Choose final waveforms per station by:
    * VEL instead of ACC type
    * and those who have the most traces
    * in draw, the last found is chosen
    """

    # create a list of Stream object, one per sta
    # for each station found
    for _station in list(set([tr.stats.station for tr in config.st])):

        # get particular station's all traces
        _st=config.st.select(station=_station)
 
        # disable them in order to choose the preferred
        for tr in _st:
            tr.enabled=False
  
        # get a unique list of (net,sta,loc,cha,type,len)
        traces=list(set([(_.stats.network, _.stats.station, _.stats.location, \
        _.stats.channel[:2],'VEL' if _.stats.response.instrument_sensitivity.input_units in config.vel_types else 'ACC'\
        ,_st.select(network=_.stats.network, station=_.stats.station, \
        location=_.stats.location, channel=_.stats.channel[:2]+'*').count()) \
        for _ in _st]))

        # sort by type broadband
        # + then sort by max len
        traces = sorted(traces, key = operator.itemgetter(4, 5))

        # for the passed trace type of the particular station, enable traces
        for tr in _st.select(network=traces[-1][0], station=traces[-1][1], \
        location=traces[-1][2], channel=traces[-1][3]+'*'):
            tr.enabled=True

    # keep only the passed traces
    config.st=Stream([tr for tr in config.st if tr.enabled])

    # sort stations by priority, type, len and distance
    stations=[(sta[0],sta[1],'VEL' if config.st.select(station=sta[0])[0].stats.response.instrument_sensitivity.input_units in config.vel_types else 'ACC', config.st.select(station=sta[0]).count(),-float(sta[4][0])) for sta in config.locations if config.st.select(station=sta[0])]

    stations = sorted(stations, key = operator.itemgetter(1,2,3,4), reverse=True)

    # cut to MaxStations
    # remove the rest stations
    for sta in stations[config.cfg['Green']['MaxStations']:]:
        for tr in config.st.select(station=sta[0]):
            config.st.remove(tr)
    
    # save corrected waveforms
    config.logger.info(('Saving corrected waveforms in one mseed file in ' + \
    '{}').format(os.path.join(config.workdir,'streams_corrected.mseed')))
    config.st.write(os.path.join(config.workdir,'streams_corrected.mseed'), \
    format='MSEED')

