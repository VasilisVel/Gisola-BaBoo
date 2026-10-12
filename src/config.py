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

import yaml, logging, sys, os, time as tm
from obspy import UTCDateTime, Stream, Trace, Inventory
#from obspy.geodetics.base import inside_geobounds
from turfpy.measurement import boolean_point_in_polygon
from geojson import Point, Polygon, Feature
import re

# local lib
import event

def n_cores(_cfg=None):
    """Cores this program may use: the CPUs it is allowed to run on
    (os.sched_getaffinity - respects taskset, cpusets and job schedulers),
    capped by the optional top-level MaxCores of the configuration (e.g. for a
    container CPU quota, which the affinity does not show). Used for every
    parallel part, so a run never starts more workers than it is given."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    c = _cfg if _cfg is not None else globals().get('cfg')
    try:
        mc = c.get('MaxCores') if isinstance(c, dict) else None
        if mc is not None and int(mc) > 0:
            n = min(n, int(mc))
    except (TypeError, ValueError):
        pass
    return max(1, n)


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    console = logging.StreamHandler() 
 
    form=logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    form.converter = tm.gmtime

    handler.setFormatter(form)
    console.setFormatter(form)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # console shows only warnings/errors; full INFO detail stays in the log file
    console.setLevel(logging.WARNING)

    logger.addHandler(handler)
    logger.addHandler(console)

    logger.propagate = False

    return logger



def read(filepath):
    with open(filepath, 'r') as _f:
        return yaml.load(_f, Loader=yaml.FullLoader)

def rules(magnitude, domain):
    return [elem[2] for elem in list(filter(lambda x: True \
    if round(x[0],1)<=round(magnitude,1) and round(magnitude,1)<=round(x[1],1) \
    else False, domain))]

def init(evnt, _cfg):
    """
    Initialize process based on event's info and configuration
    """
    # check crustal model geobox
    def check(rule):
        if rule['Geobox'] and boolean_point_in_polygon(Feature(\
           geometry=Point((org.longitude, org.latitude))), \
           Polygon([[eval('('+x.replace(')','').replace('(','')+')') for x in rule['Geobox'].replace(', ', ',').split('),(')]])):
            return True

    # global variables for the entire program
    global cfg, evt, org, inv, distRules, shiftRules, windowRules, freqRules, st, \
    gridRules, crustalRules, workdir, sourcedir, crustaldir, grdatdir, greendir,\
    vel_types, acc_types, inpinvdir, inversiondir, allstatdir, rawdir, blackList, \
    whiteList, stationPriority, magPriority, azmGap, outputdir, bestinvdir, besttl, \
    solutions, correlations, logger, locations, revise

    # make configuration global variable
    cfg=_cfg

    # make event global variable
    evt=evnt
    org=event.getOrigin(cfg,evnt,cfg['Watcher']['Historical'])
    mag=event.getMagnitude(evnt, org)

    # broadband
    vel_types=['M/S', 'M/SEC', 'NM/S', 'NM/SEC', 'CM/S', 'CM/SEC', 'MM/S', 'MM/SEC']

    # strong motion
    acc_types=['M/S**2', 'M/(S**2)', 'M/SEC**2', 'M/(SEC**2)', 'M/S/S', \
    'NM/S**2', 'NM/(S**2)', 'NM/SEC**2', 'NM/(SEC**2)', 'CM/S**2', \
    'CM/(S**2)', 'CM/SEC**2', 'CM/(SEC**2)', 'MM/S**2', 'MM/(S**2)', \
    'MM/SEC**2', 'MM/(SEC**2)']

    st=Stream() # init

    # read configuration file
    #cfg=read(filepath)

    # define workdir
    #workdir=os.path.join(cfg['WorkDir'],str(org.time)[:4],str(org.time).split('.')[0]+'_'+\
    #        os.path.basename(str(evt.resource_id)),str(UTCDateTime.now()))
    workdir=os.path.join(cfg['WorkDir'],str(org.time)[:4],os.path.basename(str(evt.resource_id)),str(UTCDateTime.now()))

    sourcedir=os.path.join(workdir, 'sources')
    crustaldir=os.path.join(workdir, 'crustals')
    grdatdir=os.path.join(workdir,'grdat')
    greendir=os.path.join(workdir,'greens')
    inpinvdir=os.path.join(workdir,'inpinv')
    allstatdir=os.path.join(workdir,'allstat')
    rawdir=os.path.join(workdir,'raw')
    inversiondir=os.path.join(workdir,'inversions')
    outputdir=os.path.join(workdir,'output')

    os.makedirs(workdir)
    os.makedirs(outputdir)

    revise=False

    logger = setup_logger(str(UTCDateTime.now()), os.path.join(outputdir, 'log'))

    #logging.basicConfig(level=getattr(logging, 'INFO'), \
    #format='%(asctime)s - %(message)s\n', handlers=[
    #    logging.FileHandler(filename=os.path.join(workdir, 'log')),
    #    logging.StreamHandler()
    #])

    distRules=rules(mag.mag, cfg['Inventory']['Distance'])
    shiftRules=rules(mag.mag, cfg['Inversion']['TimeShift'])
    windowRules=rules(mag.mag, cfg['Inversion']['Window'])
    freqRules=rules(mag.mag, cfg['Inversion']['Frequency'])

    # filter grid rules based on magnitude
    gridRules=list(filter(lambda x: True if round(x['Rule'][0],1)<=round(mag.mag,1) and \
    round(mag.mag,1)<=round(x['Rule'][1],1) else False, cfg['Green']['Grid']))

    crustalRules=list(filter(check, cfg['Green']['Crustal']))

    if not crustalRules:
        crustalRules=[rule for rule in cfg['Green']['Crustal'] if not rule['Geobox']]

    # read station file for whitelist and blacklist
    # stationPriority: (net, sta) -> priority of EVERY station in the file, used to
    # rank stations (selectAzm, prioritize); whiteList only holds the stations
    # allowed for this magnitude (priority >= magPriority)
    stationPriority={}
    if cfg['Inventory']['WhiteList']['Filepath']:
        with open(cfg['Inventory']['WhiteList']['Filepath'], 'r') as _:
           lines= _.readlines()

        magPriority=max(rules(mag.mag, cfg['Inventory']['WhiteList']['Rules']))

        whiteList=[]
        blackList=[]

        for line in lines:
            if not(line.startswith('#') or line.startswith('\n')):
                net, sta, *_=line.split()
                priority=int(_[0]) if _ else int(cfg['Inventory']['WhiteList']['Priority'])
                stationPriority[(net, sta)]=priority

                if priority==0:
                    blackList.append([net,sta])

                elif priority >= magPriority:
                    whiteList.append([net,sta,priority])

def dump(dictionary):
    return yaml.dump(dictionary, indent=2, sort_keys=True)

# decorator function for logging
def log(func):
    def inner(*args, **kwargs):
        func(*args, **kwargs)
        logger.info(st.__str__(extended=True))
    return inner

# decorator function for timing
def time(func):
    def inner(*args, **kwargs):
        start = tm.perf_counter()   # durations: monotonic clock (tm.time() can jump under WSL)
        func(*args, **kwargs)
        logger.info('Function \'{}\' executed {} seconds'.format(func.__name__,round(tm.perf_counter()-start,2)))
    return inner

def get_keyinv(cfg_data, origin_data=None):
    """
    Determines the keyinv value. Defaults to 2 (deviatoric).
    If origin_data is provided and falls within any geobox defined
    in Inversion.IsotropicGeoboxes, keyinv becomes 1 (isotropic).
    
    Note: Geobox coordinates should be entered in (lat, lon) format.
    """
    inversion_config = cfg_data.get('Inversion', {})

    # If Keyinv is set explicitly in config, honour it directly
    # (bypasses the IsotropicGeoboxes logic).
    explicit_keyinv = inversion_config.get('Keyinv')
    if explicit_keyinv is not None:
        return int(explicit_keyinv)

    isotropic_geoboxes_list = inversion_config.get('IsotropicGeoboxes')
    
    if isotropic_geoboxes_list and isinstance(isotropic_geoboxes_list, list) and origin_data:
        logger.info(f"Checking {len(isotropic_geoboxes_list)} geobox(es) for isotropic inversion.")
        for idx, geobox_entry in enumerate(isotropic_geoboxes_list):
            geobox_str = None
            if isinstance(geobox_entry, str):
                geobox_str = geobox_entry
            else:
                logger.warning(f"Item at index {idx} in 'IsotropicGeoboxes' is not a valid geobox string. Skipping: {geobox_entry}")
                continue

            if not geobox_str or not isinstance(geobox_str, str):
                logger.warning(f"Invalid or missing 'Geobox' string in 'IsotropicGeoboxes' item at index {idx}. Skipping.")
                continue
            
            try:
                # Extract tuples from the string using regex
                matches = re.findall(r'\(\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*\)', geobox_str)
                # Parse as (lat, lon) but convert to (lon, lat) for GeoJSON format
                parsed_coords = [(float(lon), float(lat)) for lat, lon in matches]

                if not parsed_coords:   
                    logger.warning(f"'Geobox' string '{geobox_str}' in 'IsotropicGeoboxes' item at index {idx} resulted in no coordinates. Skipping.")
                    continue

                # Ensure the polygon is closed
                if parsed_coords[0] != parsed_coords[-1]:
                    parsed_coords.append(parsed_coords[0])
                
                geobox_polygon = Polygon([parsed_coords])
                event_point = Feature(geometry=Point((origin_data.longitude, origin_data.latitude)))

                is_inside = boolean_point_in_polygon(event_point.geometry, geobox_polygon)
                if is_inside:
                    logger.info(f"Event at (lat: {origin_data.latitude:.4f}, lon: {origin_data.longitude:.4f}) is within IsotropicGeobox at index {idx}. Setting Keyinv to 1 (isotropic).")
                    return 1 # Isotropic
                
            except Exception as e:
                logger.error(f"Error processing 'IsotropicGeoboxes' item at index {idx} ('{geobox_str}'): {e}. Skipping.")
        
        # no geobox matched
        logger.info(f"Event not within any 'IsotropicGeoboxes'. Using default Keyinv: 2 (deviatoric).")
    
    elif isotropic_geoboxes_list and not origin_data:
        logger.info("'IsotropicGeoboxes' are defined, but no origin data provided for check. Using default Keyinv: 2 (deviatoric).")
    
    elif not isotropic_geoboxes_list:
        logger.info("'IsotropicGeoboxes' not defined or empty. Using default Keyinv: 2 (deviatoric).")

    else: # isotropic_geoboxes_list is not a list
        logger.warning(f"'IsotropicGeoboxes' is defined but is not a list. Expected a list of geobox strings. Using default Keyinv: 2 (deviatoric).")

    return 2 # Deviatoric

# add more properties to Trace
Trace.enabled=False # enabled for processing
Trace.distance=None
Trace.azimuth=None

