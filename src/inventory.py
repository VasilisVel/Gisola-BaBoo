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

import numpy as np, os.path
from obspy import read_inventory, Inventory, Stream
from obspy.clients.fdsn.client import Client
from obspy.geodetics.base import kilometers2degrees
from obspy.geodetics.base import gps2dist_azimuth # for fast run: sudo -H pip3 install geographiclib !!!!
import operator, logging, math

# import local configuration 
import config

@config.time
def getInventory(save=True):
    """
    Downloading inventory from stationXML file or FDSNWS
    """
    # find min and max distance of accepted rules
    # for defining inventory geobox selection
    mindist=min([rule[0] for rule in config.distRules])
    maxdist=max([rule[1] for rule in config.distRules])

    # get the maximum time window to retrieve data
    maxtl=max([tl for tl in config.windowRules])

    if 'ppsd' in config.cfg['Stream']['Modules']:
        starttime=config.org.time-10-3600 # half hour
    else:
        starttime=config.org.time-10

    endtime=config.org.time+math.ceil(maxtl)+1

    # channel bands of the active distance rules (e.g. ['BH', 'HH'])
    bands=sorted(set(case for rule in config.distRules for case in rule[2]))

    config.logger.info(('Retrieving inventory with working epochs from {}' + \
    ' to {} and stations between {} and {} km from event\'s ' + \
    'location').format(starttime, endtime, mindist, maxdist))

    inv=Inventory()

    # use queue priority as priority host service
    for service in config.cfg['Inventory']['Service']:
        # loops will finish when no service is left for requesting data

        if service[0]=='FDSNWS':
            try:
                # connect to host
                fdsnws = Client(service[1], eida_token=service[2])
                config.logger.info('Connecting to FDSNWS host ' +service[1])

                # download inventory from host based on config rules; only the
                # channel bands of the active distance rules are requested (e.g.
                # BH?,HH?) - the same channels are selected below, so the inventory
                # is unchanged, but the unused channels are not downloaded
                _inv=fdsnws.get_stations(starttime=starttime,
                     endtime=endtime, latitude=config.org.latitude, \
                     longitude=config.org.longitude, \
                     minradius=kilometers2degrees(mindist), \
                     maxradius=kilometers2degrees(maxdist), \
                     channel=','.join(band+'?' for band in bands), \
                     includerestricted=True, level='response')

            except Exception as e:
                if e.__class__.__name__=='FDSNNoDataException':
                    config.logger.info('Could not retrieve inventory from FDSNWS host '+service[1])
                else:
                    config.logger.info(e)
                _inv=Inventory()
                pass

        elif service[0]=='StationXML':
            try:
                config.logger.info('Reading Inventory from ' +service[1])
                _inv=read_inventory(service[1], format='STATIONXML').select(starttime=starttime, \
                     endtime=endtime, latitude=config.org.latitude, longitude=config.org.longitude, \
                     minradius=kilometers2degrees(mindist), \
                     maxradius=kilometers2degrees(maxdist))

            except:
                config.logger.info('Could not load Inventory file '+service[1])
                _inv=Inventory()
                pass

        # remove duplicates (keep only the first occurance)
        for name in list(set([sta.code for net in inv for sta in net])):
            _inv=_inv.remove(station=name)
        # append rest
        inv+=_inv

    _inv=inv
    # get only selected streams
    pass  # quiet: print("Numbers of stations: ",len([sta for net in _inv for sta in net]))
    chas=[]
    # for each accepted channel get all possible component combinations
    for cha in [case for rule in config.distRules for case in rule[2]]:
        chas+=[cha+orient for comp in config.cfg['Inventory']['Components'] \
        for orient in comp]
    chas=sorted(list(set(chas)))

    config.logger.info('Accepted channels: '+ str(chas))
    pass  # quiet: print("Unique channels: ",set([ch.code for net in _inv for sta in net for ch in sta]))
    inv=Inventory()
    for cha in chas:
        inv+=_inv.select(channel=cha)
    pass  # quiet: print("Stations chosen: ",len(inv))
    # keep unique station names
    _inv=Inventory()
    for name in list(set([sta.code for net in inv for sta in net])):
        _inv+=inv.select(station=name)
    pass  # quiet: print("Unique stations: ",len(_inv))
    # if magnitude priority is greater than default keep only the whiteList
    if config.cfg['Inventory']['WhiteList']['Filepath']:
        if config.magPriority > int(config.cfg['Inventory']['WhiteList']['Priority']):
            _inv2=Inventory()
            for sta in config.whiteList:
                _inv2+=_inv.select(network=sta[0],station=sta[1])
            _inv=_inv2

        else:
            # remove blacklisted (priority=0) stations
            for sta in config.blackList:
                _inv=_inv.remove(network=sta[0],station=sta[1])
    pass  # quiet: print("Stations after white list: ",len(_inv))
    config.inv=_inv

    # save inventory
    if save:
        config.logger.info(('Saving inventory in one StationXML file in ' + \
        '{}').format(os.path.join(config.workdir,'inventory.xml')))
        config.inv.write(os.path.join(config.workdir,'inventory.xml'), \
        format='StationXML')

@config.time
def selectDist():
    """
    Selecting subset of inventory based on Distance Rules
    """
    _inv=Inventory()
    for rule in config.distRules:
        config.logger.info(('Selecting stations with channel codes {} and that' + \
        ' they are between {} and {} km from event\'s ' + \
        'location').format(rule[2], rule[0], rule[1]))
        for cha in rule[2]:
            _inv+=config.inv.select(latitude=config.org.latitude, \
                longitude=config.org.longitude, \
                minradius=kilometers2degrees(rule[0]), \
                maxradius=kilometers2degrees(rule[1]), \
                channel=cha+'*')
 
    config.inv=_inv

    # save inventory
    config.logger.info('Saving inventory in XML file {}'.format(os.path.join(
    config.workdir,'inventory.xml')))
    config.inv.write(os.path.join(config.workdir,'inventory.xml'), format='STATIONXML')  

@config.time
@config.log
def selectAzm():
    """
    Selecting subset of streams based on Azimuthal Coverage and Rules
    """
    config.logger.info('Selecting subset of streams based on Azimuthal '+ \
    'Coverage and Rules')
    count=0

    # disable them in order to choose the preferred
    for tr in config.st:
        tr.enabled=False

    stations=[]
    for _station in list(set([tr.stats.station for tr in config.st])):
        _net=config.inv.select(station=_station)[0]
        _inv=_net[0]

        # get priority from the station file; stations not in it get the
        # default Priority (1 when no station file is used)
        priority=1 # default value in case of null stations file
        if config.cfg['Inventory']['WhiteList']['Filepath']:
            priority=config.stationPriority.get((_net.code, _station),
                int(config.cfg['Inventory']['WhiteList']['Priority']))

        stations.append((_station, priority, _inv.latitude, _inv.longitude, \
        gps2dist_azimuth(config.org.latitude, config.org.longitude, \
        _inv.latitude, _inv.longitude)))

    def _instruments(st_code):
        """
        Instruments (net, loc, channel band) of a station with valid metadata:
        {instrument: (sensor type, number of traces)}. An instrument without
        sensitivity is excluded with a warning.
        """
        inst={}
        for tr in config.st.select(station=st_code):
            key=(tr.stats.network, tr.stats.location, tr.stats.channel[:2])
            if key in inst:
                continue
            try:
                sens=tr.stats.response.instrument_sensitivity
                if sens is None:
                    raise ValueError('instrument_sensitivity is None')
                _type='VEL' if sens.input_units in config.vel_types else 'ACC'
            except Exception as e:
                config.logger.warning('Could not determine sensor type for {}.{}.{}: {}; excluding'.format(
                    st_code, key[1], key[2], e))
                inst[key]=None
                continue
            inst[key]=(_type, config.st.select(network=key[0], station=st_code,
                       location=key[1], channel=key[2]+'*').count())
        return {k: v for k, v in inst.items() if v is not None}

    # each station is ranked by the instrument it will use (as in
    # stream.prioritize: velocity before acceleration, then most traces),
    # so a station with two instruments does not count the traces of both
    instruments={_[0]: _instruments(_[0]) for _ in stations}
    best={sta: max(inst.values(), key=lambda v: (v[0]=='VEL', v[1])) if inst else None
          for sta, inst in instruments.items()}

    for deg in range(0,360,45):
        #sector=[_ for _ in stations if _[4][1]>=deg and _[4][1]<deg+45]

        sector=[(_[0],_[1],best[_[0]][0], best[_[0]][1],-float(_[4][0])) for _ in stations if _[4][1]>=deg and _[4][1]<deg+45 and best[_[0]] is not None]

        # sort them by priority value
        sector=sorted(sector, key = operator.itemgetter(1,2,3,4), reverse=True)

        # cut to the max stations per sector
        sector=sector[:config.cfg['Inventory']['Azimuth'][1]]
        config.logger.info('Sector [{},{}): {}'.format(deg, deg+45, sector))

        if sector:
            count+=1

        # mark traces to be used (only instruments with valid metadata)
        for quad in sector:
            for tr in config.st.select(station=quad[0]):
                if (tr.stats.network, tr.stats.location, tr.stats.channel[:2]) in instruments[quad[0]]:
                    tr.enabled=True

    # write station, priority, lat, lon, distance, azimuth and backazimuth
    with open(os.path.join(config.workdir,'locations'), 'w') as f:
        f.writelines('{}\n'.format(x) for x in sorted(stations, key=operator.itemgetter(4)))

    # check number of sectors
    if count < config.cfg['Inventory']['Azimuth'][0]:
        config.logger.info('Azimuthal coverage less than configuration set. ' +\
        'Terminating process')
        raise RuntimeError('Azimuthal coverage less than configuration set')

    # keep only the passed traces
    config.st=Stream([tr for tr in config.st if tr.enabled]).sort()

    config.locations=stations
