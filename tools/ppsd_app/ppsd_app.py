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

import os, inspect, sys, datetime, argparse, textwrap, os.path
from obspy import UTCDateTime, Stream
from obspy.core.event.origin import Origin
from obspy.geodetics.base import degrees2kilometers,  gps2dist_azimuth
from geojson import Point, Polygon, Feature
from turfpy.measurement import boolean_point_in_polygon, centroid
from turfpy.transformation import transform_scale
import numpy as np
import multiprocessing
from functools import partial

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

# local libs
import src.modules.ppsd as ppsd
import src.inventory as inventory
import src.stream as stream
import src.config as config

# app meta data
__copyright__ = "🄯 "+str(datetime.datetime.utcnow().year)+", Institute of Geodynamics - National Observatory of Athens"
__credits__ = ["Nikolaos Triantafyllis (triantafyl@noa.gr), Ioannis Venetis (venetis@unipi.gr), Ioannis Fountoulakis (ifountoul@noa.gr), Erion-Vasilis Pikoulis (pikoulis@ceid.upatras.gr), Efthimios Sokos (esokos@upatras.gr), Christos Evangelidis (cevan@noa.gr)"]
__author__= 'Nikolaos Triantafyllis (triantafyl@noa.gr)'
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Nikolaos Triantafyllis"
__email__ = "triantafyl@noa.gr"
__status__ = "Production"


# functions
def apply(trace, dirname='PPSD', threshold=70):
    """
    Check for PPSD and return True/False
    """
    try:
        os.makedirs(os.path.join(parentdir,'src','modules', dirname))
    except:
        pass
    with open(os.path.join(parentdir,'src','modules', dirname, trace.get_id()), 'w') as f:
        f.write(str(ppsd.check(trace, threshold=threshold)))

def rescale(geobox, dist):
    """
    Rescale function
    """
    #Move to Polygon
    f = Feature(geometry=Polygon(geobox))
    #Calculate the cetroid of the polygon
    c = centroid(f)
    #Calculate the maximum distance of the polygon from the centroid in earth.
    # points are (longitude, latitude); gps2dist_azimuth takes (lat, lon, lat, lon)
    dists = [gps2dist_azimuth(c['geometry']['coordinates'][1], c['geometry']['coordinates'][0], coord[1], coord[0])[0]/1000 \
            for coord in f['geometry']['coordinates'][0]]
    scale_ = 1 + dist/np.max(dists) 
   
    #Do the rescale
    # return Polygon
    return transform_scale(f, scale_, origin='centroid') 

class LineWrapRawTextHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _split_lines(self, text, width):
        text = self._whitespace_matcher.sub(' ', text).strip()
        return textwrap.wrap(text, width)


# programs parameters
parser = argparse.ArgumentParser(prog='PPSD_APP', description='Automatic PPSD inspection tool\nFind more info at: https://github.com/nikosT/Gisola',     formatter_class=LineWrapRawTextHelpFormatter)
parser.add_argument('-c', '--config', metavar=('FILEPATH'), help='override default configuration file (./config.yaml)', nargs='?', default='./config.yaml', type=str)
parser.add_argument('-l', '--log', metavar=('FILEPATH'), help='override default main log file (./log)', nargs='?', default='./log', type=str)
parser.add_argument("-th", "--threshold", metavar=('FLOAT'), help='assign threshold for setting PPSD coverage to False', action="store", nargs=1, type=float, default=70)
parser.add_argument("-r", "--range", metavar=('FLOAT'), help='set PPSD coverage check time range in minutes', action="store", nargs=1, type=float, default=24*60)
parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
parser.add_argument('-v', '--version', action='version', version='%(prog)s v'+__version__)

args=parser.parse_args(args=None if sys.argv[1:] else ['--help'])

# -th and -r give a one-element list when set on the command line
threshold=float(args.threshold[0] if isinstance(args.threshold, list) else args.threshold)
checkrange=float(args.range[0] if isinstance(args.range, list) else args.range)*60 # sec

# the data end one hour before now and must contain at least one PPSD
# segment of one hour, so the range must be at least 2 hours
if checkrange < 2*3600:
    print('PPSD time range (-r) must be at least 120 minutes')
    sys.exit(1)


# reads configuration file
cfg=config.read(args.config)

# if no stationfile in white list found, exit
if not cfg['Inventory']['WhiteList']['Filepath']:
    print('Station file cannot be null in WhiteList-Filepath')
    sys.exit(0)

# setup log
logger = config.setup_logger('ppsd_app_log', args.log)
inventory.config.cfg=cfg
inventory.config.logger=logger
stream.config.logger=logger
ppsd.config.logger=logger


logger.info('Starting PPSD calculation')

# setup inventory request
# modify at your own needs
channels=list(set([x for _ in cfg['Inventory']['Distance'] for x in _[2][2]]))
inventory.config.distRules=[[0,degrees2kilometers(180),channels]]
# data from (now - range) to (now - 1 hour)
inventory.config.windowRules=[checkrange-3600-1-10]
inventory.config.org=Origin()
inventory.config.org.latitude=0
inventory.config.org.longitude=0
inventory.config.org.time=UTCDateTime.now()-checkrange+10
inventory.config.magPriority=0
inventory.config.cfg['Stream']['Modules']=[]
inventory.config.blackList=[]
inventory.config.whiteList=[]
inventory.config.workdir='./'

if cfg['Inventory']['WhiteList']['Filepath']:
    with open(os.path.join(os.path.dirname(args.config),cfg['Inventory']['WhiteList']['Filepath']), 'r') as f:
        lines=f.readlines()
    for line in lines:
        if not(line.startswith('#') or line.startswith('\n')):
            net, sta, *_=line.split()
            priority=int(_[0]) if _ else int(cfg['Inventory']['WhiteList']['Priority'])

            if priority==0:
                inventory.config.blackList.append([net,sta])

inventory.getInventory(save=False)


# remove stations outside the Geobox (no Geobox: keep all stations)
if cfg['Watcher']['Geobox']:
    logger.info('Upscale Geobox based on longest distance and remove stations outside of it')

    geobox=[[eval('('+x.replace(')','').replace('(','')+')') for x in cfg['Watcher']['Geobox'].replace(', ', ',').split('),(')]]
    dist=max([rule[2][1] for rule in cfg['Inventory']['Distance']])

    newgeobox=rescale(geobox,dist)

    # check every station (not only the first one of each network)
    outside=[(net.code, sta.code) for net in inventory.config.inv for sta in net \
             if not boolean_point_in_polygon(Feature(geometry=Point((sta.longitude, sta.latitude))),newgeobox)]
    for net_code, sta_code in outside:
        inventory.config.inv = inventory.config.inv.remove(network=net_code, station=sta_code)
    logger.info('Stations removed outside the Geobox: {}'.format(len(outside)))
else:
    logger.info('No Watcher Geobox: all stations are checked')


# setup waveform request
# remove seedlink from service is very slow
stream.config.cfg['Stream']['Service']=[s for s in stream.config.cfg['Stream']['Service'] if not s[0]=='SeedLink']

st=Stream()

_inv=inventory.config.inv.copy()

# one request per station (network, station)
stations = sorted(list(set([(net.code, sta.code) for net in inventory.config.inv for sta in net])))
logger.info(f"Stations to process: {len(stations)}")

for idx, (net_code, sta_code) in enumerate(stations, start=1):
    logger.info(f"[{idx}/{len(stations)}] Processing station {net_code}.{sta_code}")
    stream.config.st = Stream()
    stream.config.inv = _inv.select(network=net_code, station=sta_code)
    try:
        stream.getWaveforms(save=False)

        ntr = len(stream.config.st)
        if ntr == 0:
            logger.info(f"No traces found for {net_code}.{sta_code} in requested window.")
            continue

        trace_ids = ", ".join([tr.get_id() for tr in stream.config.st])
        logger.info(f"Fetched {ntr} trace(s) for {net_code}.{sta_code}: {trace_ids}")

        # write True/False in trace file based on PPSD check
        with multiprocessing.Pool() as p:
            _ = p.map(partial(apply, threshold=threshold), stream.config.st)
        logger.info(f"Submitted PPSD checks for {net_code}.{sta_code} ({ntr} trace(s)).")

    except Exception as e:
        logger.exception(f"Error occurred for {net_code}.{sta_code}")
        pass

logger.info('End of PPSD calculation')


