#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    Copyright (C) 2026 Velentzas Vasilis

#    This file is part of Gisola 2.0.

#    GisolaBaBoo is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, 
#    or any later version.

#    GisolaBaBoo is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with GisolaBaBoo.  If not, see <https://www.gnu.org/licenses/>.

# OpenMP worker threads (Numba's parallel loops) sleep between parallel calls instead of
# spinning: removes a fixed ~10 ms cost of every parallel call (graph-search batches, silsub)
# without changing any result; long calls (inversion, bootstrap) are unaffected. Must be set
# before the OpenMP runtime is loaded; an OMP_WAIT_POLICY set by the user is kept.
import os as _os
_os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')

import datetime, sys, argparse, yaml, logging
from obspy.clients.fdsn.client import Client
from obspy import Catalog, Stream, read_events, read
from argparse import RawTextHelpFormatter
from obspy.core import UTCDateTime
import textwrap, os, glob, shutil
import numpy as np

# local libs
import event, inventory, stream, IsolaPy, plot, web, config
import time
import warnings
warnings.filterwarnings('ignore')


# Dictionary to time all different operations
def _stage(name, t0, store):
    dt = time.perf_counter() - t0
    store[name] = store.get(name, 0.0) + dt
    print('[STAGE] {:<38s}: {:7.2f} s'.format(name, dt), flush=True)




__copyright__ = "🄯 "+str(datetime.datetime.utcnow().year)+", Patras Seismological Laboratory - University of Patras "
__credits__ = ["Vasilis Velentzas (velentzas.vasilhs@gmail.com),   Efthimios Sokos (esokos@upatras.gr)"]
__author__= 'Vasilis Velentzas (velentzas.vasilhs@gmail.com)'
__license__ = "GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Vasilis Velentzas"
__email__ = "velentzas.vasilhs@gmail.com"
__status__ = "Production"

os.environ['COLUMNS'] = "90"

def print_help(func):
    def inner():
        print(r"""  ____ _           _       ____        ____
 / ___(_)___  ___ | | __ _| __ )  __ _| __ )  ___   ___
| |  _| / __|/ _ \| |/ _` |  _ \ / _` |  _ \ / _ \ / _ \
| |_| | \__ \ (_) | | (_| | |_) | (_| | |_) | (_) | (_) |
 \____|_|___/\___/|_|\__,_|____/ \__,_|____/ \___/ \___/
                                         """)
        print('R e a l – T i m e   M o m e n t   T e n s o r   C o m p u t a t i o n')
        print('A n d   U n c e r t a i n t y   Q u a n t i f i c a t i o n\n')
        print('Version: ' + __version__)
        print('License: ' + __license__)
        print('Author: ' + __author__)
        print ('Credits: '+ textwrap.TextWrapper().fill(text=''.join(__credits__)))
        print( __copyright__+'\n')
        func()
    return inner

class LineWrapRawTextHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _split_lines(self, text, width):
        text = self._whitespace_matcher.sub(' ', text).strip()
        return textwrap.wrap(text, width)


parser = argparse.ArgumentParser(prog='GisolaBaBoo', description='Real-Time Moment Tensor Computation And Uncertainty Quantification\nFind more info at: https://github.com/VasilisVel/Gisola-BaBoo',     formatter_class=LineWrapRawTextHelpFormatter)

# decorate (overload) print help message
parser.print_help=print_help(parser.print_help)

help_event='EVENT can be in any of the following formats: \n(i) DATETIME MAGNITUDE TYPE LATITUDE LONGITUDE DEPTH\ne.g.: ' + __file__ + ' -e ' + datetime.datetime.utcnow().isoformat() + ' 3.0 ML 37.24 20.49 4.1\n(ii) DATETIME\ne.g.: ' + __file__ + ' -e ' + datetime.datetime.utcnow().isoformat()+'\n(iii) EVENTID (event identifiers are data center specific)\ne.g.: ' + __file__ + ' -e noa2020owyrp\nIn cases (ii) and (iii) the rest of the information is retrieved by the FDSNWS-event \nIn more than one results, only the first event is returned \nPassing milliseconds is optional \nPassing more events is also posible using -e or --event prefix for each event'

parser.add_argument('-c', '--config', metavar=('FILEPATH'), help='override default configuration file (./config.yaml)', nargs='?', default='./config.yaml', type=str)
parser.add_argument('-e', '--event', metavar=('EVENT'), help=help_event, action='append', type=str, nargs='+', default=None)
parser.add_argument('--event-file', metavar=('FILEPATH'), help='parse and run a file with EVENT lines', type=argparse.FileType('r'), default=None)
parser.add_argument('--event-xml', metavar=('FILEPATH'), help='parse and run a file in QuakeML', type=argparse.FileType('r'), default=None)
parser.add_argument('-d', '--datetime-range', metavar=('TIME'), help='invoke MT computation for all events found in specific datetime range (FDSN bounded)', action='append', nargs=2, default=None)
parser.add_argument("--real-time", action="store_true", help='invoke --datetime-range for real-time use (FDSN bounded)', default=False)
parser.add_argument("--home", metavar=('YEAR'), help='Render Home page for particular year (based on config.yaml results dir)', action='store', nargs=1, default=None)

parser.add_argument("-s", "--station", metavar=('STA[.NEZ]'), help='override default stations selection. Optionally, components could be also specified.\nIt can be combined with --remove for the reverse result', action="extend", nargs="+", type=str, default=[])
parser.add_argument("--remove", help='it can only be used with --station. Invokes reverse result', action="store_true", default=False)
parser.add_argument("--disable-filters", help='disable all filter checks in process', action="store_true", default=False)
parser.add_argument("-cru", "--crustal", metavar=('FILEPATH'), help='override default crustal models', action="extend", nargs="+", type=str, default=None)
parser.add_argument("-dist", "--distance", metavar=('STRING'), help='override default distance rule', action="append", nargs=4, type=str, default=None)
parser.add_argument("-tl", "--time-window", metavar=('FLOAT'), help='override default inversion time-window in seconds (any value > 0; Dt = window/1024)', action="extend", nargs=1, type=float, default=None)
parser.add_argument("-ts", "--time-shift", metavar=('INT'), help='override default time search grid', action="append", nargs=3, type=int, default=None)
parser.add_argument("-f", "--frequency", metavar=('FLOAT'), help='override default inversion frequency band (fmin fmax)', action="append", nargs=2, type=float, default=None)
parser.add_argument("-r", "--revise", action="store_true", default=False, help='Trigger revised MT procedure. Computation assumes all available input exists\n(e.g. Greens\' Functions) and runs only the inversion and plot procedures.\nOnly --frequency, --time-shift, --station (--remove), --config' )
parser.add_argument("--revise-config", metavar=('FILEPATH'), help='Alternative config file for revise mode. Drives all inversion parameters (frequency, timeshift, window, grid). Mutually exclusive with -f.', default=None, type=str)
parser.add_argument('-l', '--log', metavar=('FILEPATH'), help='override default main log file (./log)', nargs='?', default='./log', type=str)
parser.add_argument('-v', '--version', action='version', version='%(prog)s v'+__version__)

args=parser.parse_args(args=None if sys.argv[1:] else ['--help'])

if args.frequency and args.revise_config:
    parser.error('--frequency and --revise-config are mutually exclusive')
if args.time_shift and args.revise_config:
    parser.error('--time-shift and --revise-config are mutually exclusive')
if args.time_window and any(_w <= 0 for _w in args.time_window):
    parser.error('--time-window must be > 0')

# a revision re-uses the stations, data, Green's functions and window of the first run:
# these overrides do not apply to it, so they are rejected and the revision continues
if args.revise:
    for _opt, _val in (('--time-window', args.time_window), ('--distance', args.distance),
                       ('--crustal', args.crustal), ('--disable-filters', args.disable_filters)):
        if _val:
            print('[WARN] {} is not used with --revise (the first run is re-used): ignored'.format(_opt), flush=True)
    args.time_window = args.distance = args.crustal = None
    args.disable_filters = False

_cfg=config.read(args.config)
_rev_cfg = config.read(args.revise_config) if args.revise_config else None

# ---- cores: the CPUs this run may use (allowed CPUs, capped by MaxCores) -------
# Numba's parallel loops and the BLAS/LAPACK threads use them; the process/thread
# pools and AXITRA read config.n_cores() themselves
_n_cores = config.n_cores(_cfg)
try:
    import numba as _numba
    _numba.set_num_threads(min(_n_cores, _numba.config.NUMBA_NUM_THREADS))
except Exception:
    pass
try:
    from threadpoolctl import threadpool_limits as _tpl
    _tpl(limits=_n_cores)
except Exception:
    pass

# ---- GPU / CPU device selection ------------------------------------------
# Falls back to CPU if no CUDA device is available.
_compute_device = str(_cfg.get('ComputeDevice', 'CPU')).upper()
if _compute_device == 'GPU':
    try:
        from numba import cuda as _cuda
        _gpu_ok = len(list(_cuda.gpus)) > 0
    except Exception:
        _gpu_ok = False
    if _gpu_ok:
        IsolaPy._USE_GPU = True
        print('[INFO] Compute device : GPU', flush=True)
    else:
        print('[INFO] GPU requested but no CUDA device found — using CPU', flush=True)
else:
    print('[INFO] Compute device : CPU', flush=True)
# ----------------------------------------------------------------------

cat=Catalog()

logger = config.setup_logger('gisola_log', args.log)

if args.home:
    web.commandRenderHome(cfg=_cfg,year=args.home[0])

_STAGE_T = {}
_t_cat = time.perf_counter()

if args.event_xml:
    cat=read_events(args.event_xml)

# events of --event-file are added to those given with -e (blank and # lines skipped)
if args.event_file:
    with args.event_file as _f:
        args.event=(args.event or []) + [_e.split() for _e in _f.readlines()
                                         if _e.strip() and not _e.lstrip().startswith('#')]

# if --event-file or --event argument is passed
if args.event and not args.revise:
    for text in args.event:
        try:
            # if event is FDSN bound
            if len(text)==1:
                # try by event ID first; if not found, fall back to time window
                # (as old Gisola: any error of the ID request, e.g. a server that
                # rejects a date-time as an invalid ID, falls back to the time window)
                try:
                    _res = event.getFDSNWSCatalog(_cfg, logger, eventid=text[0])
                except Exception as _e:
                    logger.info('Event ID request for %s failed (%s); trying it as a date-time', text[0], _e)
                    _res = None
                if not _res:
                    _res = event.getFDSNWSCatalog(_cfg, logger, starttime=UTCDateTime(text[0])-5, endtime=UTCDateTime(text[0])+5)
                if _res:
                    cat.append(_res[0])
                else:
                    logger.warning('No event found for %s', text[0])
                    print(f'[WARN] No event found for {text[0]}', flush=True)
            # create a manual object
            elif len(text)==6:
                cat.append(event.getCatalog(text)[0])
        except Exception as _e:
            logger.warning('Could not retrieve event for %s: %s', text, _e)
            print(f'[WARN] Could not retrieve event {text}: {_e}', flush=True)

_stage('event retrieval', _t_cat, _STAGE_T)
_ev_retrieval_t = _STAGE_T['event retrieval']

# by pass configuration options
if args.disable_filters:
    _cfg['Stream']['Modules']=[]

# overrides apply to every magnitude (a rule range of [0, 0], as in old Gisola,
# matched no real event, so the flags left the rules empty)
if args.distance:
    _cfg['Inventory']['Distance']=[[-10,10, [float(_[0]), float(_[1]), _[2:]]] for _ in args.distance]

if args.crustal:
    _cfg['Green']['Crustal']=[]
    for text in args.crustal:
        _cfg['Green']['Crustal'].append({'Filepath': text, 'Geobox': None})

if args.time_window:
    _cfg['Inversion']['Window']=[[-10,10,_] for _ in args.time_window]

if args.time_shift:
    _cfg['Inversion']['TimeShift']=[[-10,10,[*_]] for _ in args.time_shift]

if args.frequency:
    _cfg['Inversion']['Frequency']=[[-10,10,[*_]] for _ in args.frequency]

# if --datetime-range is given append event found by FDSN
if args.datetime_range:
    start=args.datetime_range[0][0]
    end=args.datetime_range[0][1]

    minMag=min([_[0] for _ in _cfg['Inversion']['Window']])
    # no events (None) or a failed request must not crash the run
    try:
        _res = event.getFDSNWSCatalog(_cfg, logger, minMag=minMag, starttime=UTCDateTime(start), endtime=UTCDateTime(end))
    except Exception as _e:
        logger.warning('Event request for %s - %s failed: %s', start, end, _e)
        print('[WARN] Event request for {} - {} failed: {}'.format(start, end, _e), flush=True)
        _res = None
    if _res:
        cat+=_res
    else:
        print('[WARN] No events found between {} and {}'.format(start, end), flush=True)

if args.real_time:
    cat+=event.watch(_cfg, logger)

# traces of one -s entry STA[.CHA[.COMPS]] (normal run). CHA is the channel code
# without its component (HH; also a pattern such as HH? or *; empty = any) and
# COMPS the components to keep (e.g. NZ). Old Gisola passed channel and component
# together to Stream.select(), which ObsPy rejects, and matched a 2-letter CHA
# against the full channel code (HH never matched HHZ).
def _station_traces(spec):
    res = spec.split('.')
    cha = res[1] if len(res) > 1 and res[1] else '*'
    if len(cha) == 2:
        cha += '?'
    _sel = config.st.select(station=res[0], channel=cha)
    if len(res) > 2 and res[2]:
        _sel = Stream([tr for tr in _sel if tr.stats.channel[-1].upper() in res[2].upper()])
    return _sel

# by-pass azimuthal coverage rule
if args.station and not args.remove:
    _cfg['Inventory']['Azimuth']=[1,_cfg['Green']['MaxStations']]

if not args.revise:
    if len(cat) == 0:
        print('[WARN] No events in catalog — nothing to process. '
              'Check FDSNWS connectivity and Event Host in config.', flush=True)
    # for each event returned from catalog object
    for evt in cat:
        try:
            config.init(evt, _cfg)
        except Exception as e:
            print('Fatal Error in init: '+str(e))
            continue

        _ev_label = os.path.basename(str(evt.resource_id))
        try:
            #config.logger.info(('Loading initial configuration\n '+ \
            #'{}').format(config.dump(config.cfg)))
            org=event.getOrigin(config.cfg,evt,config.cfg['Watcher']['Historical'])
            config.logger.info('Starting MT calculation for Event {}\n{}\n{}\n'.format(evt, org,event.getMagnitude(evt,org)))
            config.logger.info('Setting working dir to: {}'.format(config.workdir))
            _ev_label = '{} (M{:.1f})'.format(_ev_label, event.getMagnitude(evt, org).mag)

            # no rule for this magnitude: the event cannot be computed, so it is skipped
            _no_rule = [_n for _r, _n in ((config.distRules, 'Inventory.Distance'),
                                          (config.gridRules, 'Green.Grid')) if not _r]
            if _no_rule:
                _msg = 'Event {} skipped: no {} rule for this magnitude'.format(_ev_label, ' or '.join(_no_rule))
                config.logger.warning(_msg); print('[SKIP] ' + _msg, flush=True)
                continue

    
            _STAGE_T = {'event retrieval': _ev_retrieval_t}
            _tpre = time.perf_counter()
            _t = time.perf_counter()
            # inventory.xml is saved by selectDist when the distance rules run; here
            # only when they are skipped (-s without --remove)
            inventory.getInventory(save=bool(args.station) and not args.remove)
            _stage('read inventory', _t, _STAGE_T)
            config.logger.info(config.inv)

            # distance rules: skipped only when -s chooses the stations (old Gisola also
            # skipped them with -s ... --remove, which should only drop the listed stations)
            if not args.station or args.remove:
                _t = time.perf_counter()
                inventory.selectDist()
                config.logger.info(config.inv)
                _stage('station selection (distance)', _t, _STAGE_T)
          
            _t = time.perf_counter()
            stream.getWaveforms()
            _stage('read waveforms', _t, _STAGE_T)

            _t = time.perf_counter()
            stream.clean()
            _stage('quality checks (snr/clip/mouse) + correction', _t, _STAGE_T)

            _t = time.perf_counter()
            inventory.selectAzm()
            _stage('azimuthal selection', _t, _STAGE_T)

            if args.station and args.remove:
                for sta in args.station:
                    for tr in _station_traces(sta):
                        config.st.remove(tr)

            elif args.station and not args.remove:
                _st=Stream()
                for sta in args.station:
                    _st+=_station_traces(sta)
                config.st=_st

            _t = time.perf_counter()
            stream.prioritize()
            _stage('prioritize + save corrected', _t, _STAGE_T)
            _STAGE_T['read + preprocessing'] = time.perf_counter() - _tpre
            print('[STAGE] {:<38s}: {:7.2f} s'.format('read + preprocessing',_STAGE_T['read + preprocessing']), flush=True)

            # ── 1. Green's functions (AXITRA) ─────────────────────────────────
            _t = time.perf_counter()
            IsolaPy.getGreens()
            _stage('green function calculation', _t, _STAGE_T)

            # ── 2–8. Inversion loop (all combinations) ────────────────────────
            # Runs getElemse → graph search → getInversions → getBootstrapping
            # → doPlots → renderSite for every (icrustal,igrid) × allstat × inpinv
            # combination, each in its own output folder.
            IsolaPy.calculateInversions(_STAGE_T)

 
        except:
            _e = sys.exc_info()[1]
            if isinstance(_e, RuntimeError) and str(_e).startswith('Azimuthal coverage'):
                # an expected quality rejection, not an error
                _msg = 'Event {} skipped: {}'.format(_ev_label, _e)
                config.logger.warning(_msg); print('[SKIP] ' + _msg, flush=True)
            else:
                config.logger.exception('Fatal Error occurred')
            config.logger.info('Moving to next event, if any')

# if revise flag is enabled
else:
    for eventid in args.event:
        try:
            eventid = eventid[0]

            # ── 1. Locate the most recent completed run for this event ─────────
            def _has_results(ts_path):
                """True if the timestamp dir contains at least one completed inversion."""
                inv_dir = os.path.join(ts_path, 'inversions')
                if not os.path.isdir(inv_dir):
                    return False
                return any(
                    os.path.isfile(os.path.join(inv_dir, combo, 'results_inversion.dat'))
                    for combo in os.listdir(inv_dir)
                    if os.path.isdir(os.path.join(inv_dir, combo))
                )

            _year    = ''.join([s for s in eventid if s.isdigit()][:4])
            _evt_dir = os.path.join(_cfg['WorkDir'], _year, eventid)
            dirs = [x for x in glob.glob(os.path.join(_evt_dir, '*'))
                    if _has_results(x)]
            if not dirs:
                # Fast two-level glob: WorkDir/*/UUID (covers year-partitioned catalogs)
                for _candidate in glob.glob(os.path.join(_cfg['WorkDir'], '*', eventid)):
                    _candidate_dirs = [x for x in glob.glob(os.path.join(_candidate, '*'))
                                       if _has_results(x)]
                    if _candidate_dirs:
                        _evt_dir = _candidate
                        dirs = _candidate_dirs
                        break
            if not dirs:
                # Slow fallback: full tree walk
                for root, dir, _ in os.walk(_cfg['WorkDir']):
                    if eventid in dir:
                        _evt_dir = os.path.join(root, eventid)
                        dirs = [x for x in glob.glob(os.path.join(_evt_dir, '*'))
                                if _has_results(x)]
                        break

            if not dirs:
                print('[REVISE] No completed results found for {} — skipping.'.format(eventid), flush=True)
                continue

            latest  = max(dirs, key=os.path.getctime)
            workdir = os.path.join(_evt_dir, os.path.basename(latest))

            # ── 2. Find best combination by VR ────────────────────────────────
            _inv_folders = [f for f in glob.glob(os.path.join(workdir, 'inversions', '*'))
                            if os.path.isfile(os.path.join(f, 'results_inversion.dat'))]

            if not _inv_folders:
                print('[REVISE] No completed inversion folders found — original run '
                      'may be incomplete. Skipping {}'.format(eventid), flush=True)
                continue

            def _parse_vr(folder):
                try:
                    with open(os.path.join(folder, 'results_inversion.dat')) as _f:
                        for line in _f:
                            if line.startswith('VR:'):
                                return float(line.split(':')[1].strip())
                except Exception:
                    pass
                return -999.0

            _best_folder = max(_inv_folders, key=_parse_vr)
            _best_vr     = _parse_vr(_best_folder)
            _best_name   = os.path.basename(_best_folder)
            config.best_combo = _best_name     # for $bestinvdir in Notification.Command
            # folder name: {ai}.{ii}.{icrustal}.{igrdat}.{igrid}
            _bparts      = _best_name.split('.')
            _ai, _ii, _icrustal, _igrdat, _igrid = \
                _bparts[0], _bparts[1], _bparts[2], _bparts[3], _bparts[4]

            print('[REVISE] Best combination: {} (VR={:.1f}%)'.format(
                  _best_name, _best_vr), flush=True)

            # ── 3. Revision goes into the run's own output/ (as old Gisola) ───
            # The event is taken as it is now (output/event.xml: the automatic
            # solution, possibly an earlier revision). The revised solution is
            # added as 'reviewed' and becomes the preferred one; an earlier
            # revision is replaced. Revised files get a '.revise' suffix, so the
            # automatic solution's files stay untouched.
            _outdir = os.path.join(workdir, 'output')
            _evt_xml = os.path.join(_outdir, 'event.xml')
            if not os.path.isfile(_evt_xml):
                _evt_xml = os.path.join(_best_folder, 'event.xml')
            evt = IsolaPy.prepare_revision_base(
                read_events(_evt_xml)[0],
                agency=_cfg['Citation']['Agency'], author=_cfg['Citation']['Author'])

            config.logger = config.setup_logger(
                str(UTCDateTime.now()),
                os.path.join(_outdir, 'log.revised'))

            # point config dirs; explicitly set workdir-derived paths
            IsolaPy._EVT_BASE   = evt               # QuakeML of the revision is built on it
            config.evt          = evt
            config.org          = event.getOrigin(_cfg, evt, _cfg['Watcher']['Historical'])
            config.revise       = True
            _mag = event.getMagnitude(evt, config.org).mag
            _pcfg = _rev_cfg if _rev_cfg else _cfg  # parameter source for inversion params
            config.cfg          = _pcfg             # drives Keyinv, STF, and all reads inside isola2
            if args.frequency:
                config.freqRules = [args.frequency[0]]
            else:
                config.freqRules = config.rules(_mag, _pcfg['Inversion']['Frequency'])
            config.shiftRules   = config.rules(_mag, _pcfg['Inversion']['TimeShift'])
            config.windowRules  = config.rules(_mag, _cfg['Inversion']['Window'])  # window fixed — tied to fetched streams
            config.gridRules    = [r for r in _pcfg['Green']['Grid']
                                   if round(r['Rule'][0],1) <= round(_mag,1) <= round(r['Rule'][1],1)]
            config.crustalRules = []
            config.workdir      = workdir
            config.outputdir    = _outdir
            config.allstatdir   = os.path.join(workdir, 'allstat')
            config.inpinvdir    = os.path.join(workdir, 'inpinv')
            config.greendir     = os.path.join(workdir, 'greens')
            config.rawdir       = os.path.join(workdir, 'raw')
            config.grdatdir     = os.path.join(workdir, 'grdat')
            config.crustaldir   = os.path.join(workdir, 'crustals')

            config.logger.info('Revise mode: best=%s VR=%.1f%% revision written to %s',
                               _best_name, _best_vr, _outdir)

            # ── 4. Load best combination's allstat, apply overrides ───────────
            _orig_allstat = os.path.join(workdir, 'allstat', 'allstat{}.dat'.format(_ai))
            with open(_orig_allstat, 'r') as _f:
                lines = _f.readlines()
            lines = list(map(str.split, lines))

            # frequency override: replace f1 f2 (columns 5-6) in every row
            # source: -f flag OR --revise-config Frequency rules for this magnitude
            _freq_override = None
            if args.frequency:
                _freq_override = args.frequency[0]
            elif _rev_cfg:
                _freq_rules = config.rules(_mag, _rev_cfg['Inversion']['Frequency'])
                if _freq_rules:
                    _freq_override = _freq_rules[0]

            if _freq_override:
                for line in lines:
                    line[5] = str(_freq_override[0])
                    line[6] = str(_freq_override[1])

            # station component override
            for sta in args.station:
                res = sta.split('.')
                if len(res) == 1:
                    res.extend([None, 'NEZ'])
                if len(res) == 3:
                    for line in lines:
                        if line[0] == res[0]:
                            if bool(int(line[2])):
                                line[2] = int('N' in list(res[2])) if not args.remove \
                                          else int('N' not in list(res[2]))
                            if bool(int(line[3])):
                                line[3] = int('E' in list(res[2])) if not args.remove \
                                          else int('E' not in list(res[2]))
                            if bool(int(line[4])):
                                line[4] = int('Z' in list(res[2])) if not args.remove \
                                          else int('Z' not in list(res[2]))
                            line[1] = int(int(line[2]) or int(line[3]) or int(line[4]))
                            break

            # if --station without --remove: zero out everything not in the list
            if args.station and not args.remove:
                _sta_list = [s.split('.')[0] for s in args.station]
                for line in lines:
                    if line[0] not in _sta_list:
                        line[1] = line[2] = line[3] = line[4] = 0

            # write modified allstat next to the original (old Gisola: allstat<ai>.dat.revise)
            lines = [list(map(str, line)) for line in lines]
            _rev_allstat_file = _orig_allstat + '.revise'
            with open(_rev_allstat_file, 'w') as _f:
                _f.writelines(' '.join(line) + '\n' for line in lines)

            # ── 5. Load corrected waveforms and apply component mask ──────────
            config.st = read(os.path.join(workdir, 'streams_corrected.mseed'))
            for line in lines:
                if int(line[2]) == 0:
                    for tr in config.st.select(station=line[0], component='N'):
                        config.st.remove(tr)
                if int(line[3]) == 0:
                    for tr in config.st.select(station=line[0], component='E'):
                        config.st.remove(tr)
                if int(line[4]) == 0:
                    for tr in config.st.select(station=line[0], component='Z'):
                        config.st.remove(tr)

            # ── 6. Re-run inversion (reuse Green's functions from best combo) ─
            # Time shifts: the best combination's own inpinv file (as old
            # Gisola). With -ts, the given time shifts are used instead; with
            # --revise-config, its TimeShift rule for this magnitude (same rule
            # index as the original when it exists, otherwise the first one).
            # Both are written as inpinv<ii>.dat.revise.
            _inpinv_file = os.path.join(config.inpinvdir, 'inpinv{}.dat'.format(_ii))
            if args.time_shift:
                _inpinv_file = _inpinv_file + '.revise'
                with open(_inpinv_file, 'w') as _f:
                    _f.write('{} {} {}'.format(*args.time_shift[0]))
            elif _rev_cfg:
                _srules = config.shiftRules
                if not _srules:
                    raise ValueError('--revise-config has no TimeShift rule for Mw {:.1f}'.format(_mag))
                _k = int(_ii) if int(_ii) < len(_srules) else 0
                if _k != int(_ii):
                    _msg = ('--revise-config has {} TimeShift rule(s) for this magnitude; '
                            'rule index {} of the original run does not exist - using rule 0'
                            .format(len(_srules), _ii))
                    config.logger.warning(_msg); print('[WARN] ' + _msg, flush=True)
                _inpinv_file = _inpinv_file + '.revise'
                with open(_inpinv_file, 'w') as _f:
                    _f.write('{} {} {}'.format(*_srules[_k]))
            elif not os.path.isfile(_inpinv_file):
                raise FileNotFoundError('time-shift file of the best combination not found: '
                                        + _inpinv_file)

            def _chunk_no_r(fname):
                return int(os.path.basename(fname).split('.')[-2])

            _hes_files = sorted(
                [os.path.join(config.greendir, f)
                 for f in os.listdir(config.greendir)
                 if (f.endswith('.hes') and
                     f.split('.')[1] == _icrustal and
                     f.split('.')[2] == _igrdat and
                     f.split('.')[3] == _igrid)],
                key=_chunk_no_r)

            _REV_STAGE_T = {}

            _t = time.perf_counter()
            IsolaPy.getElemse(hes_files=_hes_files,
                             allstat_file=_rev_allstat_file,
                             inpinv_file=_inpinv_file)
            _dt = time.perf_counter() - _t
            _REV_STAGE_T['elementary seismogram calculation'] = _dt
            print('[STAGE] {:<38s}: {:7.2f} s'.format('elementary seismogram calculation', _dt), flush=True)

            # ── 6b. Graph search (before inversion, writes *.dat to revision output) ─
            _gs_enabled = _cfg.get('GraphSearch', {}).get('Enabled', True)
            if _gs_enabled:
                IsolaPy._GRAPH_SEARCH_ALAS = None
                IsolaPy._GS_SHARED         = None

                IsolaPy.buildNeighborGraph()
                if IsolaPy._GS_SHARED is not None:
                    print('[GS] Graph build      : {:7.0f} ms  ({} nodes)'.format(
                          IsolaPy._GS_SHARED['graph_build_time'] * 1000,
                          IsolaPy._GS_SHARED['Total_gridPoints']), flush=True)

                IsolaPy.getGraphSearch()
                if IsolaPy._GRAPH_SEARCH_ALAS is not None:
                    _mn = IsolaPy._GRAPH_SEARCH_ALAS['max_node']
                    print('[GS] Graph Search     : {:7.0f} ms  '
                          '(x={:.1f}, y={:.1f}, z={:.1f})'.format(
                          IsolaPy._GRAPH_SEARCH_ALAS['gs_time'] * 1000,
                          _mn[0], _mn[1], _mn[2]), flush=True)

            IsolaPy.getInversions()
            _inv_dt_rev = (IsolaPy._BOOT_CTX or {}).get('times', {}).get('kernel', 0.0)
            _REV_STAGE_T['full inversion (kernel)'] = _inv_dt_rev
            print('[STAGE] {:<38s}: {:7.2f} s'.format('full inversion (kernel)', _inv_dt_rev), flush=True)

            _boot_enabled = _cfg.get('Bootstrap', {}).get('Enabled', True)
            if _boot_enabled:
                _t = time.perf_counter()
                IsolaPy.getBootstrapping()
                _dt = time.perf_counter() - _t
                _REV_STAGE_T['bootstrapping'] = _dt
                print('[STAGE] {:<38s}: {:7.2f} s'.format('bootstrapping', _dt), flush=True)

            # ── 7. Plots (*.revise.png), event page and catalog (reviewed) ───
            IsolaPy.visualizeResults(_REV_STAGE_T)

            print('[REVISE] Done. Reviewed solution written to: {} '
                  '(event.xml, index.html, *.revise files)'.format(_outdir), flush=True)

        except Exception:
            import traceback as _tb
            try:
                config.logger.exception('Fatal Error occurred (revise)')
                config.logger.info('Moving to next event, if any')
            except AttributeError:
                _tb.print_exc()
