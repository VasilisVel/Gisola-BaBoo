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

# OpenMP worker threads (Numba's parallel loops) sleep between parallel calls instead of
# spinning: removes a fixed ~10 ms cost of every parallel call (graph-search batches, silsub)
# without changing any result; long calls (inversion, bootstrap) are unaffected. Must be set
# before the OpenMP runtime is loaded; an OMP_WAIT_POLICY set by the user is kept.
import os as _os
_os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')

import math, os, os.path, operator
from scipy.fft import irfft
import numpy as np, multiprocessing
from collections import OrderedDict
import shutil, subprocess, logging, glob
from pathlib import Path
from obspy.geodetics.base import gps2dist_azimuth # for fast run: sudo -H pip3 install geographiclib
from obspy.core.event.source import FocalMechanism,  MomentTensor, NodalPlanes, \
                                    NodalPlane, PrincipalAxes, Axis, Tensor
from obspy.core.event.base import WaveformStreamID, DataUsed, CreationInfo
from obspy import read, UTCDateTime
from obspy.core.event.event import Event
from obspy.core.event.origin import Origin, OriginQuality
from obspy.core.event.magnitude import Magnitude
from obspy.geodetics import kilometers2degrees
from obspy.core import AttribDict
from scipy.signal import butter, sosfilt
import pandas as _pd
from scipy.spatial import KDTree as _KDTree
#import geo.sphere
# local
import config, event, modules.kagan
# Numba saves the compiled kernels (cache=True) in the folder __pycache__ next to the source
# file, and reuses them in the next runs. precompile.py fills this cache in advance. The cache
# belongs to the modification time of each source file: after any change of this file, the
# kernels are compiled again (see the manual, Pre-compiled kernels).
from numba import njit, prange, cuda, float64
import concurrent.futures
from functools import partial
import time
from threadpoolctl import threadpool_info
import numba
import sys
NUM_OF_TIME_SAMPLES = 1024

# ---- bootstrap / plotting support --------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')          # figures are saved to files, never shown
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

try:
    from pyrocko import moment_tensor as pmt
    from pyrocko.plot import hudson, beachball
    HAVE_PYROCKO = True
except ImportError:
    HAVE_PYROCKO = False

# filled by getElemse(), consumed by getBootstrapping()
_BOOT_CTX = None
_SUMMARY  = None   # filled by getBootstrapping(), printed by GisolaBaBoo.py
# filled by getBootstrapping(), consumed by renderSite()
_BOOT_STATS = None
# filled by getBootstrapping(), consumed by exportResults() for deferred figure generation
_BOOT_PLOT_CTX = None
# ---- GPU flag (set by GisolaBaBoo.py from config ComputeDevice) ----
_USE_GPU = False
# ---- graph search state (populated by getGraphSearch(); consumed by calculateInversions()) ----
_GRAPH_SEARCH_ALAS = None   # seeded adaptive lookahead result
_GS_SHARED         = None   # shared df/neighbor_graph built once before search
# ---- event every combination's QuakeML is built from (a fresh copy per
#      combination, so solutions of other combinations never accumulate) ----
_EVT_BASE = None
# ------------------------------------------------------------------------


def _rv(fname):
    """Output file name, with '.revise' inserted before the extension in
    revise mode (as old Gisola: map.png -> map.revise.png,
    solutions -> solutions.revise), so a revision never overwrites the
    automatic solution's files."""
    if not getattr(config, 'revise', False):
        return fname
    root, ext = os.path.splitext(fname)
    return root + '.revise' + ext


def prepare_revision_base(evt, agency=None, author=None):
    """Event to attach a revised solution to (old gatherResults(revise=True)):
    removes any previous 'reviewed' focal mechanism, so only ONE revision is
    kept, and any non-preferred automatic Gisola solution (duplicates written
    by other combinations), each with its derived origin and magnitude.
    Solutions from other agencies are kept."""
    import copy
    evt = copy.deepcopy(evt)

    def _ours(fm):
        ci = fm.creation_info
        return ci is not None and (agency is None or ci.agency_id == agency) \
            and (author is None or ci.author == author)

    # the automatic solution to keep: the preferred one if it is automatic,
    # otherwise (event already revised) the last automatic Gisola solution
    pref = evt.preferred_focal_mechanism()
    ours_prelim = [fm for fm in evt.focal_mechanisms
                   if _ours(fm) and fm.evaluation_status == 'preliminary']
    if pref is not None and any(fm.resource_id == pref.resource_id for fm in ours_prelim):
        keep_id = pref.resource_id
    else:
        keep_id = ours_prelim[-1].resource_id if ours_prelim else None

    ours_ids = {fm.resource_id for fm in ours_prelim}
    drop = [fm for fm in evt.focal_mechanisms
            if fm.evaluation_status == 'reviewed'
            or (fm.resource_id in ours_ids and fm.resource_id != keep_id)]
    for fm in drop:
        der = fm.moment_tensor.derived_origin_id if fm.moment_tensor else None
        evt.focal_mechanisms = [f for f in evt.focal_mechanisms
                                if f.resource_id != fm.resource_id]
        if der is not None:
            evt.origins = [o for o in evt.origins if o.resource_id != der]
            evt.magnitudes = [m for m in evt.magnitudes if m.origin_id != der]
    if keep_id is not None:
        evt.preferred_focal_mechanism_id = keep_id
    return evt

# ========================================================================
# Bootstrap: one (perturbation, source) task, shared by the CPU and the GPU
# ========================================================================
def _boot_combine_task(Rt, Ct, wc, k, rr, scale, MULTIP, R, L, A, g, y, x, vr_row, me_row):
    """One perturbation (trace weights wc) at source k, from the weight-free per-trace
    sums of bootstrap_precompute(): R = scale * sum_c w_c Rt[k, c] (the same for all
    shifts, so it is factored once: R = L L^T), then for every shift
    g = scale * sum_c w_c Ct[k, s, c], L y = g, L^T a = y, VR = a.g / MULTIP / rr.
    If R is not positive definite, each shift is solved by Gaussian elimination with
    partial pivoting instead. R, L, A (6x6) and g, y, x (6) are scratch arrays.
    Plain Python: compiled for the CPU (_boot_combine_task_cpu, njit) and the GPU
    (_boot_combine_task_gpu, CUDA device function) - the same operations in the same order."""
    n_traces = Rt.shape[1]
    nc       = Rt.shape[2]
    n_shifts = Ct.shape[1]
    for a in range(nc):
        for b in range(nc):
            R[a, b] = 0.0
    for c in range(n_traces):
        w = wc[c]
        if w == 0.0:
            continue
        for a in range(nc):
            for b in range(nc):
                R[a, b] += w * Rt[k, c, a, b]
    for a in range(nc):
        for b in range(nc):
            R[a, b] *= scale
    # Cholesky factor of R (lower triangle of L)
    chol = True
    for a in range(nc):
        for b in range(a + 1):
            s = R[a, b]
            for m in range(b):
                s -= L[a, m] * L[b, m]
            if a == b:
                if s <= 0.0:
                    chol = False
                    break
                L[a, a] = math.sqrt(s)
            else:
                L[a, b] = s / L[b, b]
        if not chol:
            break
    for si in range(n_shifts):
        for n in range(nc):
            g[n] = 0.0
        for c in range(n_traces):
            w = wc[c]
            if w == 0.0:
                continue
            for n in range(nc):
                g[n] += w * Ct[k, si, c, n]
        for n in range(nc):
            g[n] *= scale
        if chol:
            for a in range(nc):
                s = g[a]
                for m in range(a):
                    s -= L[a, m] * y[m]
                y[a] = s / L[a, a]
            for a in range(nc - 1, -1, -1):
                s = y[a]
                for m in range(a + 1, nc):
                    s -= L[m, a] * x[m]
                x[a] = s / L[a, a]
        else:
            for a in range(nc):
                for b in range(nc):
                    A[a, b] = R[a, b]
                y[a] = g[a]
            for col in range(nc):
                piv = col
                for row in range(col + 1, nc):
                    if abs(A[row, col]) > abs(A[piv, col]):
                        piv = row
                if piv != col:
                    for b in range(nc):
                        t = A[col, b]; A[col, b] = A[piv, b]; A[piv, b] = t
                    t = y[col]; y[col] = y[piv]; y[piv] = t
                for row in range(col + 1, nc):
                    f = A[row, col] / A[col, col]
                    for b in range(col, nc):
                        A[row, b] -= f * A[col, b]
                    y[row] -= f * y[col]
            for a in range(nc - 1, -1, -1):
                s = y[a]
                for m in range(a + 1, nc):
                    s -= A[a, m] * x[m]
                x[a] = s / A[a, a]
        sum1 = 0.0
        for n in range(nc):
            sum1 += x[n] * g[n]
        vr_row[si] = ((sum1 / MULTIP) / rr) * 100.0
        for n in range(nc):
            me_row[si, n] = x[n]


# ========================================================================
# GPU / CUDA support
# ========================================================================
try:
    _CUDA_AVAILABLE = cuda.is_available()
except Exception:
    _CUDA_AVAILABLE = False

if _CUDA_AVAILABLE:

    _boot_combine_task_gpu = cuda.jit(device=True)(_boot_combine_task)

    @cuda.jit(cache=True)
    def _bootstrap_combine_kernel(Rt, Ct, Wc_all, rr_arr, vr_all, mech_all, scale, MULTIP):
        """One thread per (perturbation i, source k): _boot_combine_task() - the same
        routine as the CPU path (bootstrap_combine_all)."""
        gid     = cuda.grid(1)
        n_src   = Ct.shape[0]
        n_iters = Wc_all.shape[0]
        if gid >= n_iters * n_src:
            return
        i = gid // n_src
        k = gid % n_src
        R = cuda.local.array((6, 6), dtype=float64)
        L = cuda.local.array((6, 6), dtype=float64)
        A = cuda.local.array((6, 6), dtype=float64)
        g = cuda.local.array(6, dtype=float64)
        y = cuda.local.array(6, dtype=float64)
        x = cuda.local.array(6, dtype=float64)
        _boot_combine_task_gpu(Rt, Ct, Wc_all[i], k, rr_arr[i], scale, MULTIP,
                               R, L, A, g, y, x, vr_all[i, k], mech_all[i, k])

    @cuda.jit(cache=True)
    def _pre_Rt_kernel(G, k0, Rt):
        """Rt[k, c] = G_kc^T G_kc (whole trace, unshifted) for grid points k0 + k.
        One thread per (point k, trace c, upper-triangle entry a <= b)."""
        gid   = cuda.grid(1)
        n_src = Rt.shape[0]
        n_tr  = Rt.shape[1]
        nc    = Rt.shape[2]
        ne    = nc * (nc + 1) // 2
        if gid >= n_src * n_tr * ne:
            return
        e  = gid % ne
        kc = gid // ne
        c  = kc % n_tr
        k  = kc // n_tr
        a = 0
        while e >= nc - a:
            e -= nc - a
            a += 1
        b = a + e
        kg   = k0 + k
        base = c * 1024
        acc = 0.0
        for t in range(1024):
            acc += G[kg, base + t, a] * G[kg, base + t, b]
        Rt[k, c, a, b] = acc
        Rt[k, c, b, a] = acc

    @cuda.jit(cache=True)
    def _pre_Ct_kernel(G, obs_data, shifts_array, k0, Ct):
        """Ct[k, si, c] = sum_t G_kc(t - s) d_c(t) over the window of shift s, for grid
        points k0 + k. One thread per (point k, trace c, shift si): neighbouring threads
        are neighbouring shifts, so they read neighbouring rows of G (coalesced)."""
        gid  = cuda.grid(1)
        n_src, n_sh, n_tr, nc = Ct.shape
        if gid >= n_src * n_tr * n_sh:
            return
        si = gid % n_sh
        kc = gid // n_sh
        c  = kc % n_tr
        k  = kc // n_tr
        s  = shifts_array[si]
        t_start = 0 if s < 0 else s
        t_end   = 1024 if s >= 0 else 1024 + s
        kg   = k0 + k
        base = c * 1024
        g = cuda.local.array(6, dtype=float64)
        for n in range(nc):
            g[n] = 0.0
        for t in range(t_start, t_end):
            val = obs_data[base + t]
            for n in range(nc):
                g[n] += G[kg, base + t - s, n] * val
        for n in range(nc):
            Ct[k, si, c, n] = g[n]

    def _precompute_gpu(G_d, obs_d, shifts_d, k0, n, n_tr, nc, n_sh):
        """Device arrays Rt (n, n_tr, nc, nc) and Ct (n, n_sh, n_tr, nc) of grid points
        k0 .. k0+n-1 of the device array G_d."""
        threads = 128
        Rt_d = cuda.device_array((n, n_tr, nc, nc), dtype=np.float64)
        Ct_d = cuda.device_array((n, n_sh, n_tr, nc), dtype=np.float64)
        nt = n * n_tr * (nc * (nc + 1) // 2)
        _pre_Rt_kernel[(nt + threads - 1) // threads, threads](G_d, k0, Rt_d)
        nt = n * n_tr * n_sh
        _pre_Ct_kernel[(nt + threads - 1) // threads, threads](G_d, obs_d, shifts_d, k0, Ct_d)
        return Rt_d, Ct_d

    def _bootstrap_precompute_gpu(G, obs_data, shifts_array):
        """bootstrap_precompute() on the GPU: device Rt, Ct and host dd (_trace_power,
        the same helper as the CPU path)."""
        n_src, n_total, nc = G.shape
        n_tr = n_total // 1024
        G_d      = cuda.to_device(np.ascontiguousarray(G, dtype=np.float64))
        obs_d    = cuda.to_device(np.ascontiguousarray(obs_data, dtype=np.float64))
        shifts_d = cuda.to_device(np.ascontiguousarray(shifts_array, dtype=np.int64))
        Rt_d, Ct_d = _precompute_gpu(G_d, obs_d, shifts_d, 0, n_src, n_tr, nc,
                                     shifts_array.shape[0])
        cuda.synchronize()
        del G_d
        return Rt_d, Ct_d, _trace_power(obs_data, n_tr)

    def compute_inversion_numbaGpu(G, obs_data, Dt, shifts_array, num_stations,
                                   mem_budget=1 << 30):
        """Full inversion on the GPU - the same equations and outputs as
        compute_inversion_numbaCpu: per-trace sums Rt/Ct on the GPU (grid points in
        chunks of at most mem_budget bytes), then the shared Cholesky routine
        (_boot_combine_task) with unit weights, i.e. R = sum_c Rt, g(s) = sum_c Ct(s).
        Equal to the CPU result up to rounding (sums grouped per trace)."""
        MULTIP = 1e20
        n_src, n_total, nc = G.shape
        n_tr  = n_total // 1024
        n_sh  = shifts_array.shape[0]
        scale = MULTIP * Dt
        rr    = _data_power(np.ascontiguousarray(obs_data, dtype=np.float64), Dt)
        threads = 128

        G_d      = cuda.to_device(np.ascontiguousarray(G, dtype=np.float64))
        obs_d    = cuda.to_device(np.ascontiguousarray(obs_data, dtype=np.float64))
        shifts_d = cuda.to_device(np.ascontiguousarray(shifts_array, dtype=np.int64))
        W_d      = cuda.to_device(np.ones((1, n_tr)))
        rr_d     = cuda.to_device(np.array([rr]))

        vr_mat   = np.empty((n_src, n_sh))
        mech_mat = np.empty((n_src, n_sh, nc))
        R        = np.empty((n_src, nc, nc))
        per_src  = 8 * (n_tr * nc * nc + n_sh * n_tr * nc + n_sh * (1 + nc))
        chunk    = max(1, min(n_src, mem_budget // per_src))
        for k0 in range(0, n_src, chunk):
            n = min(chunk, n_src - k0)
            Rt_d, Ct_d = _precompute_gpu(G_d, obs_d, shifts_d, k0, n, n_tr, nc, n_sh)
            vr_d   = cuda.device_array((1, n, n_sh), dtype=np.float64)
            mech_d = cuda.device_array((1, n, n_sh, nc), dtype=np.float64)
            _bootstrap_combine_kernel[(n + threads - 1) // threads, threads](
                Rt_d, Ct_d, W_d, rr_d, vr_d, mech_d, scale, MULTIP)
            cuda.synchronize()
            vr_mat[k0:k0 + n]   = vr_d.copy_to_host()[0]
            mech_mat[k0:k0 + n] = mech_d.copy_to_host()[0]
            R[k0:k0 + n]        = Rt_d.copy_to_host().sum(axis=1) * scale
            del Rt_d, Ct_d, vr_d, mech_d
        del G_d
        return _inversion_summary(vr_mat, mech_mat, R, shifts_array)

    def _bootstrap_combine_gpu(Rt, Ct, Wc, rr, scale, MULTIP, batch):
        """GPU combine of all perturbations, in batches of `batch` perturbations:
        yields (i0, vr_all, mech_all) per batch. Rt/Ct are device arrays (GPU
        precompute) or host arrays (uploaded once)."""
        n_src, n_shifts, _, n_comp = Ct.shape
        threads = 128
        Rt_d = Rt if cuda.is_cuda_array(Rt) else cuda.to_device(Rt)
        Ct_d = Ct if cuda.is_cuda_array(Ct) else cuda.to_device(Ct)
        for i0 in range(0, Wc.shape[0], batch):
            Wb     = np.ascontiguousarray(Wc[i0:i0 + batch])
            nb     = Wb.shape[0]
            W_d    = cuda.to_device(Wb)
            rr_d   = cuda.to_device(np.ascontiguousarray(rr[i0:i0 + batch]))
            vr_d   = cuda.device_array((nb, n_src, n_shifts), dtype=np.float64)
            mech_d = cuda.device_array((nb, n_src, n_shifts, n_comp), dtype=np.float64)
            blocks = (nb * n_src + threads - 1) // threads
            _bootstrap_combine_kernel[blocks, threads](Rt_d, Ct_d, W_d, rr_d, vr_d, mech_d,
                                                       scale, MULTIP)
            cuda.synchronize()
            yield i0, vr_d.copy_to_host(), mech_d.copy_to_host()
            del W_d, rr_d, vr_d, mech_d

# ---- dispatcher (always present) ---------------------------------------
def _compute_inversion(G, obs_data, Dt, shifts_array, num_stations):
    """Full inversion on the device chosen by ComputeDevice. GPU: worthwhile on GPUs with
    fast double precision (data-centre cards); on consumer GPUs the CPU is usually faster
    (see benchmark_device.py). Falls back to the CPU if G does not fit or the GPU fails."""
    if _USE_GPU and _CUDA_AVAILABLE:
        need = G.nbytes + obs_data.nbytes + (1 << 30)     # G + chunk budget
        try:
            free = cuda.current_context().get_memory_info()[0]
        except Exception:
            free = None
        if free is not None and need > 0.9 * free:
            _msg = ('GPU memory too small for the inversion ({:.2f} GB needed, {:.2f} GB '
                    'free) - running on CPU'.format(need / 1e9, free / 1e9))
            config.logger.warning(_msg); print('[WARN] ' + _msg, flush=True)
        else:
            try:
                return compute_inversion_numbaGpu(G, obs_data, Dt, shifts_array, num_stations)
            except (MemoryError, cuda.cudadrv.driver.CudaAPIError) as _e:
                _msg = 'GPU inversion failed ({}) - running on CPU'.format(_e)
                config.logger.warning(_msg); print('[WARN] ' + _msg, flush=True)
    return compute_inversion_numbaCpu(G, obs_data, Dt, shifts_array, num_stations)
# ========================================================================

def kaganCalc(x):
    return modules.kagan.get_kagan_angle(*x[0:3], *x[3:6])


def _kagan_angles(sdr_ref, sdr):
    """Kagan angles (deg) between one double couple sdr_ref = (strike, dip,
    rake) and many sdr (n, 3). Vectorised copy of modules/kagan.py
    (get_kagan_angle): same tensor from strike/dip/rake, same eigenvector
    frame (ascending eigenvalues, third axis = cross product), same minimum
    over the double-couple symmetries - identical values, one call."""
    def tensors(s, d, r):
        s, d, r = np.radians(s), np.radians(d), np.radians(r)
        mrr = np.sin(2 * d) * np.sin(r)
        mtt = -(np.sin(d) * np.cos(r) * np.sin(2 * s) + np.sin(2 * d) * np.sin(r) * np.sin(s) ** 2)
        mpp = (np.sin(d) * np.cos(r) * np.sin(2 * s) - np.sin(2 * d) * np.sin(r) * np.cos(s) ** 2)
        mrt = -(np.cos(d) * np.cos(r) * np.cos(s) + np.cos(2 * d) * np.sin(r) * np.sin(s))
        mrp = (np.cos(d) * np.cos(r) * np.sin(s) - np.cos(2 * d) * np.sin(r) * np.cos(s))
        mtp = -(np.sin(d) * np.cos(r) * np.cos(2 * s) + 0.5 * np.sin(2 * d) * np.sin(r) * np.sin(2 * s))
        return np.stack([np.stack([mrr, mrt, mrp], -1),
                         np.stack([mrt, mtt, mtp], -1),
                         np.stack([mrp, mtp, mpp], -1)], -2)

    def frames(T):
        _, V = np.linalg.eigh(T)                 # ascending eigenvalues, as argsort in kagan.py
        V = V.copy()
        V[..., :, 2] = np.cross(V[..., :, 0], V[..., :, 1])
        return V

    sdr = np.atleast_2d(np.asarray(sdr, dtype=float))
    V1 = frames(tensors(*np.asarray(sdr_ref, dtype=float)))          # (3, 3)
    V2 = frames(tensors(sdr[:, 0], sdr[:, 1], sdr[:, 2]))            # (n, 3, 3)

    def ang(Vb):                                  # arccos((trace(V1 Vb^T) - 1) / 2)
        return np.arccos(np.clip((np.einsum('ij,nij->n', V1, Vb) - 1.0) / 2.0, -1.0, 1.0))

    th = ang(V2)
    for j in range(3):
        k = (j + 1) % 3
        V3 = V2.copy()
        V3[:, :, j] *= -1.0
        V3[:, :, k] *= -1.0
        th = np.minimum(th, ang(V3))
    return np.degrees(th)

def cart2earth(lat,lon,x,y):
    """
    BSD 3-Clause License
    Copyright (c) 2017, Juno Inc.
    All rights reserved.
    https://github.com/gojuno/geo-py/blob/master/geo/sphere.py
    Calculate lat and lon from origin lat lon and cart position
    """
    def destination(point, distance, bearing):
        '''
            Given a start point, initial bearing, and distance, this will
            calculate the destina?tion point and final bearing travelling
            along a (shortest distance) great circle arc.

            (see http://www.movable-type.co.uk/scripts/latlong.htm)
        '''
        EARTH_MEAN_RADIUS = 6371008.8

        lon1, lat1 = (math.radians(coord) for coord in point)
        radians_bearing = math.radians(bearing)

        delta = distance / EARTH_MEAN_RADIUS

        lat2 = math.asin(
            math.sin(lat1)*math.cos(delta) +
            math.cos(lat1)*math.sin(delta)*math.cos(radians_bearing)
        )
        numerator = math.sin(radians_bearing) * math.sin(delta) * math.cos(lat1)
        denominator = math.cos(delta) - math.sin(lat1) * math.sin(lat2)

        lon2 = lon1 + math.atan2(numerator, denominator)

        lon2_deg = (math.degrees(lon2) + 540) % 360 - 180
        lat2_deg = math.degrees(lat2)

        return (lon2_deg, lat2_deg)

    dist=math.sqrt(x**2+y**2)*1000
    try:
        if x==0 and y==0:
            azm=0
        elif y>=0 and x>=0:
            azm=math.atan(x/y)
        elif y<0 and x>=0:
            azm=180-math.atan(x/abs(y))
        elif y<=0 and x<=0:
            azm=180+math.atan(abs(x)/abs(y))
        elif y>0 and x<0:
            azm=270+math.atan(abs(y)/abs(x))
    except ZeroDivisionError:
        azm=90 if x>0 else -90

    lon2,lat2=destination((lon,lat), dist, azm)
    return round(lat2, 4), round(lon2, 4)


def calculateVariance(observed, synthetic, tl):
    """
    Calculates variance reduction of stations' streams
    """
    with np.errstate(divide='raise'):
        try:
            dt = round((tl/float(NUM_OF_TIME_SAMPLES)),4)
            obs = np.array(observed)
            syn = np.array(synthetic)

            ds = obs-syn
            dsn = (np.linalg.norm(ds)**2)*dt
            d = (np.linalg.norm(obs)**2)*dt

            return round(1-(dsn/float(d)),2)
        except FloatingPointError:
            return None

def _green_chunk_size():
    """Sources per AXITRA run: Green.MaxSources, capped so that the spectra of one
    run (nfreq x 18 x stations x sources complex values, 16 bytes each) stay within
    Green.RunMemoryMB (default 3000 MB; not in the configuration file). The chunk size does not change the Green's
    functions - every source is computed on its own (checked bit-identical for
    100, 200 and 400 sources per run) - only how often AXITRA repeats its set-up."""
    green = config.cfg['Green']
    n_max = int(green['MaxSources'])
    nyq, mult = _green_freq_settings()
    fmax = max([rule[1] for rule in config.freqRules])
    nfreq = max([int(nyq) if nyq is not None else int(np.ceil(tl * mult * fmax))
                 for tl in config.windowRules])
    nfreq = min(nfreq, NUM_OF_TIME_SAMPLES // 2 + 1)
    n_sta = int(green.get('MaxStations') or 16)
    budget = float(green.get('RunMemoryMB') or 3000) * 1e6
    cap = max(1, int(budget // (nfreq * 18 * n_sta * 16)))
    return max(1, min(n_max, cap))


def createSources():
    """
    Creating the Source files (source.dat)
    """
    # for each accepted grid rule, perform the following actions
    for i,rule in enumerate(config.gridRules):

        # create respective source dir
        os.makedirs(os.path.join(config.sourcedir,'grid'+str(i)))

        # calculate source points
        # for x,y (y=x) distance
        x=[]
        for drange in rule['Distance']:
            x=np.append(x,np.unique(np.append(-np.arange(*drange), \
            np.arange(*drange))))

        # for z depth
        z=[]
        for zrange in rule['Depth']:    
            z=np.append(z,np.unique(np.append(-np.arange(*zrange), \
            np.arange(*zrange))))
       
        # use event's depth as offset
        z+=round(config.org.depth/1000,1) # km
        # keep only >= 1 km 
        z=z[z>=1.0]

        # all possible source points
        mesh=np.array(np.meshgrid(x,x,z)).T.reshape(-1, 3)

        # write one file per chunk (one AXITRA run each)
        chunk = _green_chunk_size()
        for k in range(math.ceil(len(mesh)/chunk)):

            # write chunck size in text buffer and then in file
            # with the step of chuck size
            text=""
            for j,elem in enumerate(mesh[k*chunk:(k+1)*chunk]):
                text+='{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(\
                k*chunk+j+1, *elem)
            with open(os.path.join(config.sourcedir,'grid'+str(i),\
            'source'+str(k)+'.dat'), 'w') as _:
                _.write(text)

        config.logger.info(('Grid index: {}\nCalculated source points: {}\n' + \
        'Dispatched in {} source file(s) of at most {} sources').format(i, \
        len(mesh), k+1, chunk))

def createCrustals():
    """
    Copying the Crustal Model files (crustal.dat)
    """
    # create respective source dir
    os.makedirs(config.crustaldir)

    for i,rule in enumerate(config.crustalRules):
        try:
            config.logger.info(('Crustal model Index: {}\nCopying ' + \
            'crustal file: {} to {} directory').format(i, \
            rule['Filepath'], config.crustaldir))
            shutil.copy(rule['Filepath'], os.path.join(config.crustaldir, \
            'crustal'+str(i)+'.dat'))
        except:
            config.logger.info('Crustal model file: {} not found'.format( \
            rule['Filepath']))
            config.logger.info('Continue with next crustal model, if any')
            continue

def _green_freq_settings(warn=False):
    """
    Green.NyquistSamples and Green.FreqMultiplier from the configuration, checked.

    Returns (nyquist_samples or None, freq_multiplier). An invalid value does not
    stop the run: NyquistSamples (a number >= 2) falls back to the FreqMultiplier
    rule, FreqMultiplier (a number > 0) falls back to 2.5 (old Gisola). With
    warn=True (once per event, in createGrdat) the fallbacks are reported.
    """
    def _number(v):
        if isinstance(v, bool):
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    green = config.cfg.get('Green', {}) or {}
    msgs = []

    mult = green.get('FreqMultiplier', 2.5)
    if mult is None:
        mult = 2.5
    elif _number(mult) is None or _number(mult) <= 0:
        msgs.append('Green.FreqMultiplier={!r} is invalid (must be a number > 0); '
                    'using 2.5 (old Gisola)'.format(mult))
        mult = 2.5
    mult = float(mult)

    nyq = green.get('NyquistSamples', None)
    if nyq is not None:
        if _number(nyq) is None or _number(nyq) < 2:
            msgs.append('Green.NyquistSamples={!r} is invalid (must be a number >= 2); '
                        'using FreqMultiplier {:g} instead'.format(nyq, mult))
            nyq = None
        else:
            nyq = int(_number(nyq))

    if warn:
        for m in msgs:
            config.logger.warning(m)
            print('[WARN] ' + m, flush=True)
    return nyq, mult


def _check_band(f1, f2, tl, warn=False):
    """
    Check the filter band for this window (1024 samples) and return the lower
    corner to use. fmin = 0 becomes 1/tl, the lowest frequency of the window,
    as in old ISOLA (XAPIIR: a band-pass cannot start at 0). Otherwise the event
    stops with a clear message unless 0 < fmin < fmax < Nyquist = 512 / tl
    (without this check the Butterworth design fails with a scipy message).
    """
    f_nyq = NUM_OF_TIME_SAMPLES / (2.0 * tl)
    if f1 == 0:
        f1 = 1.0 / tl
        if warn:
            _msg = ('Filter band 0-{} Hz: fmin = 0 replaced by 1/tl = {:.5f} Hz (window {:g} s), '
                    'as in old ISOLA'.format(f2, f1, tl))
            config.logger.warning(_msg); print('[WARN] ' + _msg, flush=True)
    if not (f1 > 0 and f2 > f1):
        raise ValueError('Filter band {:g}-{:g} Hz is invalid: it needs 0 <= fmin < fmax'.format(f1, f2))
    if f2 >= f_nyq:
        raise ValueError('Filter band {}-{} Hz: the upper corner is at or above the Nyquist '
                         'frequency {:.4f} Hz of the {:g} s window ({} samples)'.format(
                             f1, f2, f_nyq, tl, NUM_OF_TIME_SAMPLES))
    return f1


def createGrdat():
    """
    Creating the Greens Function configuration (grdat.hed)
    """
    maxdist=max([rule[1] for rule in config.distRules])
    xl=2000000 if maxdist<=100 else int(np.ceil(20*1000*maxdist))

    # Axitra synthesis frequency — two modes (NyquistSamples takes priority):
    #   FreqMultiplier (default 2.5): nfreq = ceil(tl * multiplier * Fmax)
    #   NyquistSamples (e.g. 512):   nfreq = fixed value regardless of window
    # (invalid values fall back with a warning; more than 513 is limited below)
    _nyq_samples, _freq_mult = _green_freq_settings(warn=True)
    _fmax_filter = max([rule[1] for rule in config.freqRules])

    # an impossible band stops the event here, before AXITRA runs
    for tl in config.windowRules:
        for rule in config.freqRules:
            _check_band(rule[0], rule[1], tl)

    # create respective source dir
    os.makedirs(config.grdatdir)

    for i,tl in enumerate(config.windowRules):
        if _nyq_samples is not None:
            nfreq = int(_nyq_samples)
        else:
            nfreq = int(np.ceil(tl * _freq_mult * _fmax_filter))
        # a 1024-sample series has NUM_OF_TIME_SAMPLES//2 + 1 = 513 frequency
        # bins (0 Hz .. Nyquist); more AXITRA frequencies would not be used
        # (irfft takes the first 513), so a larger request is limited to 513
        _nf_max = NUM_OF_TIME_SAMPLES // 2 + 1
        if nfreq > _nf_max:
            config.logger.warning('nfreq=%d exceeds the %d usable frequency bins '
                                  '(window %.1f s) - limited to %d', nfreq, _nf_max, tl, _nf_max)
            print('[WARN] nfreq={} exceeds the {} usable frequency bins (window {:.1f} s) '
                  '- limited to {}'.format(nfreq, _nf_max, tl, _nf_max), flush=True)
            nfreq = _nf_max
        text=('&input\nnfreq={:n}\ntl={}\naw=1.0\nxl={}\nikmax=100000\n' + \
        'uconv=0.1E-03\nfref=1.\n/end\n').format(nfreq,tl,xl)
 
        with open(os.path.join(config.grdatdir,\
        'grdat'+str(i)+'.hed'), 'w') as _:
            _.write(text)

        config.logger.info('Greens\' Functions configuration index: {}'.format(i))

def createStations():
    """
    Creating the necessary Greens Function configuration (station.dat)
    """
    text=''

    lstations=[]
    for i,sta in enumerate(list(set([tr.stats.station for tr in config.st]))):
        distance, azimuth, _ = gps2dist_azimuth(config.org.latitude, \
        config.org.longitude, config.inv.select(station=sta)[0][0].latitude, \
        config.inv.select(station=sta)[0][0].longitude)

        lstations.append([distance,distance*math.cos(math.radians(azimuth))/1000.0, \
        distance*math.sin(math.radians(azimuth))/1000.0,0,sta])

    # sort by distance and write the necessary info
    for i,sta in enumerate(sorted(lstations, key=operator.itemgetter(0))):
        text+='{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\n'.format(i+1,*sta[1:])
    
    with open(os.path.join(config.workdir,'station.dat'), 'w') as _:
        _.write(text)

@config.time
def calculateGreens():
    """
    Calculating Greens Function (gr.hes)
    """
    def getIdx(text):
        return ''.join(list(filter(str.isdigit, text)))

    # create respective source dir
    os.makedirs(config.greendir)

    # one AXITRA run per (crustal model, grdat, grid, source file)
    jobs = []
    for crustal in sorted(os.listdir(config.crustaldir)):

        for grdat in sorted(os.listdir(config.grdatdir)):

            for grid in sorted(os.listdir(config.sourcedir)):
                for source in sorted(os.listdir(os.path.join(\
                config.sourcedir,grid))):
                    grhes='gr.{}.{}.{}.{}.hes'.format(getIdx(crustal),
                                                  getIdx(grdat),
                                                  getIdx(grid),
                                                  getIdx(source))

                    command='{} {} {} {} {} {}\n'.format(\
                                    os.path.join(os.path.dirname(os.path.abspath(__file__)),config.cfg['Green']['ExePath']),
                                    os.path.join('station.dat'),
                                    os.path.join('crustals', crustal),
                                    os.path.join('grdat', grdat),
                                    os.path.join('sources', grid, source),
                                    os.path.join('greens', grhes))
                    jobs.append((grhes, command))

    # the runs are executed one after the other; AXITRA itself is parallel over
    # frequencies (OpenMP) and uses the cores this program may use
    env = dict(os.environ)
    if 'OMP_NUM_THREADS' not in os.environ:
        env['OMP_NUM_THREADS'] = str(max(1, config.n_cores()))

    def _run(job):
        grhes, command = job
        try:
            proc=subprocess.Popen(command, cwd=config.workdir, \
            shell=True, universal_newlines=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env)
            out,err=proc.communicate()
            return grhes, out, err, None
        except Exception as e:
            return grhes, None, None, e

    for job in jobs:
        grhes, out, err, e = _run(job)
        if e is not None:
            config.logger.info('It could not create {}'.format(grhes))
            continue
        config.logger.info(err)
        config.logger.info(out)
 
def createInpinv():
    """
    Creating Inversions configuration files (inpinv)
    """
    os.makedirs(config.inpinvdir)

    for i,srule in enumerate(config.shiftRules):
        text='{} {} {}'.format(*srule)
        with open(os.path.join(config.inpinvdir, \
        'inpinv'+str(i)+'.dat'), 'w') as _:
            _.write(text)

def createAllstat():
    """
    Creating Inversions configuration files (inpinv)
    """
    # create respective source dir
    os.makedirs(config.allstatdir)

    # read all stations from station.dat file
    with open(os.path.join(config.workdir,'station.dat'), 'r') as _:
        stations=_.readlines()

    # create allstat dir
    for i,rule in enumerate(config.freqRules):
        text=''
        for stat in stations:
            sta=stat.split()
            _st=config.st.select(station=sta[4])

            if _st:
                text+='{} {} {} {} {} {}\n'.format(sta[4],
                      1 if _st.count() else 0,
                      1 if _st.select(component='N') else 0,
                      1 if _st.select(component='E') else 0,
                      1 if _st.select(component='Z') else 0,
                      ' '.join(map(str, rule)))

            else:
                text+='{} {} {} {} {} {}\n'.format(sta[4], 0, \
                      0, 0, 0, ' '.join(map(str, rule)))


            with open(os.path.join(config.allstatdir, \
            'allstat'+str(i)+'.dat'), 'w') as _:
                _.write(text[:-1])

def createRaw():
    """
    Creates raw file of data; contains 4 columns: time, N, E, Z data
    if there's less than 3 components data, it fills with ajacent data
    (dummy data) in order ISOLA to work
    """
    # create respective source dir
    os.makedirs(config.rawdir)

    # read all stations from station.dat file
    with open(os.path.join(config.workdir,'station.dat'), 'r') as _:
        stations=_.readlines()

    # break sts in unique triplets
    sts=list(set([(tr.stats.station, tr.stats.location, \
    tr.stats.channel[:-1]) for tr in config.st])) 

    # create rawfiles for each accepted tl
    for i,tl in enumerate(config.windowRules):

        # one dir for each tl found
        os.makedirs(os.path.join(config.rawdir,str(i)))

        _st=config.st.copy()
        # downsampling at known frequency at NUM_OF_TIME_SAMPLES elements
        _st.resample(NUM_OF_TIME_SAMPLES/tl)
        # be sure than no point exceeds number
        for tr in _st:
            tr.data=tr.data[:NUM_OF_TIME_SAMPLES]

        for stat in stations:
            _st2=_st.select(station=stat.split()[4])

            text=""
            if _st2.count():
                for k,time in enumerate(_st2[0].times()):
                    text+='{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(time, 
                          _st2.select(component='N')[0].data[k] if _st2.select(component='N') else 0,
                          _st2.select(component='E')[0].data[k] if _st2.select(component='E') else 0,
                          _st2.select(component='Z')[0].data[k] if _st2.select(component='Z') else 0
                          )

            else: # dummy data
                for _ in range(NUM_OF_TIME_SAMPLES):
                    text+='{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\n'.format(0, 0, 0, 0)

            with open(os.path.join(config.rawdir,str(i),stat.split()[4].upper()+'raw.dat'), 'w') as _:
                _.write(text)

@njit(fastmath=True, cache=True)
def cart2trendplunge(v):
    x, y, z = v
    if z < 0:
        x, y, z = -x, -y, -z
    trend = np.degrees(np.arctan2(y, x)) % 360.0
    plunge = np.degrees(np.arcsin(z))
    return trend, plunge


@njit(fastmath=True, cache=True)
def angles(n, s):
    """Strike, dip, rake (deg) from nodal-plane normal n and slip s.
    Line-by-line port of ANGLES (angles.inc) in ISOLA15, including its
    special cases for (near-)vertical planes, except: ISOLA's clamp of dips
    below 8.1 deg to 2.56 deg (the true dip is kept), and its horizontal-plane
    case, which here applies only to an exactly horizontal plane."""
    EPS = 1e-3          # (near-)vertical planes: the rake formula divides by n2
    HORIZ_EPS = 1e-9    # exactly horizontal plane: the strike is undefined
    HALF_PI = 0.5 * np.pi

    n0, n1, n2 = n[0], n[1], n[2]
    s0, s1, s2 = s[0], s[1], s[2]

    if n2 > 0:
        n0, n1, n2 = -n0, -n1, -n2
        s0, s1, s2 = -s0, -s1, -s2

    # ISOLA15 set every dip below 8.1 deg (n2 < -0.99) to arccos(0.999) = 2.56 deg,
    # which disagrees with the P/T/B axes; here only rounding (|n2| > 1) is guarded
    dip = np.arccos(min(1.0, -n2))

    # ISOLA used EPS (1e-3) here, i.e. every dip below 2.56 deg, and replaced
    # strike / rake by 'strike = slip direction, rake = 0' (up to ~2 x dip off);
    # the strike formula below is valid for any dip above ~0.003 deg
    if abs(abs(n2) - 1.0) < HORIZ_EPS:
        rake = 0.0
        strike = np.arctan2(s1, s0)
        if strike < 0.0:
            strike += 2.0 * np.pi
    else:
        sdi = 1.0 / np.sqrt(1.0 - n2 * n2)
        strike = np.arctan2(-n0 * sdi, n1 * sdi)
        if strike < 0.0:
            strike += 2.0 * np.pi
        if abs(n2) >= EPS:
            CF = np.cos(strike)
            SF = np.sin(strike)
            rake = np.arctan2((s0 * SF - s1 * CF) / (-n2), s0 * CF + s1 * SF)
        elif abs(strike) < EPS:
            rake = np.arctan2(-s2, s0)
        elif abs(abs(strike) - HALF_PI) < EPS:
            rake = np.arctan2(-s2, s1)
        else:
            CF = np.cos(strike)
            SF = np.sin(strike)
            if abs(CF) > abs(SF):
                cr = s0 / CF
            else:
                cr = s1 / SF
            rake = np.arctan2(-s2, cr)

    return np.degrees(strike), np.degrees(dip), np.degrees(rake)


@njit(fastmath=True, cache=True)
def build_moment_tensor(A):
    M = np.empty((3, 3))

    A1, A2, A3, A4, A5, A6 = A

    M[0, 0] = -A4 + A6
    M[1, 1] = -A5 + A6
    M[2, 2] = A4 + A5 + A6

    M[0, 1] = A1
    M[0, 2] = A2
    M[1, 2] = -A3

    M[1, 0] = M[0, 1]
    M[2, 0] = M[0, 2]
    M[2, 1] = M[1, 2]

    return M


def moment_tensor_to_a(M):
    """Inverse of build_moment_tensor: 3x3 NED moment tensor -> ISOLA
    coefficients a1..a6 (exact, build_moment_tensor(moment_tensor_to_a(M)) == M
    for any symmetric M)."""
    a6 = (M[0, 0] + M[1, 1] + M[2, 2]) / 3.0
    return np.array([M[0, 1], M[0, 2], -M[1, 2],
                     a6 - M[0, 0], a6 - M[1, 1], a6], dtype=np.float64)


@njit(fastmath=True, cache=True)
def normalize_moment_tensor(M):
    M0 = np.sqrt(
        0.5 * (M[0,0]**2 + M[1,1]**2 + M[2,2]**2)
        + M[0,1]**2 + M[0,2]**2 + M[1,2]**2
    )

    if M0 < 1e-20:
        return M, M0, False

    factor = 1.0 / (M0 * np.sqrt(2.0))

    for i in range(3):
        for j in range(3):
            M[i, j] *= factor

    return M, M0, True


@njit(fastmath=True, cache=True)
def eigendecomp_sorted(M):
    vals, vecs = np.linalg.eigh(M)

    # sort descending
    for i in range(2):
        for j in range(i+1, 3):
            if vals[i] < vals[j]:
                vals[i], vals[j] = vals[j], vals[i]
                for k in range(3):
                    vecs[k, i], vecs[k, j] = vecs[k, j], vecs[k, i]

    # right-handed system
    cross = np.array([
        vecs[1,0]*vecs[2,1] - vecs[2,0]*vecs[1,1],
        vecs[2,0]*vecs[0,1] - vecs[0,0]*vecs[2,1],
        vecs[0,0]*vecs[1,1] - vecs[1,0]*vecs[0,1]
    ])

    dot = cross[0]*vecs[0,2] + cross[1]*vecs[1,2] + cross[2]*vecs[2,2]

    if dot < 0:
        for i in range(3):
            vecs[i,2] *= -1

    return vals, vecs


@njit(fastmath=True, cache=True)
def decompose(vals):
    """ISO, CLVD and DC percentages of a moment tensor from its eigenvalues
    vals (descending, M1 >= M2 >= M3), after Vavrycuk (2015, J. Seismol. 19,
    231-252, Eqs. 6-11): the three components are scaled by the same norm
    M = |M_ISO| + |M_CLVD| + M_DC, so |ISO| + |CLVD| + DC = 100.
    ISO and CLVD are signed: both > 0 for tensile (opening) and both < 0 for
    compressive (closing) sources. ISOLA's SILSUB (Vavrycuk 2001) reported
    |CLVD| and scaled ISO by max|Mi|; the two agree whenever ISO and CLVD have
    the same sign."""
    M1 = vals[0]
    M2 = vals[1]
    M3 = vals[2]
    m_iso = (M1 + M2 + M3) / 3.0
    m_clvd = 2.0 / 3.0 * (M1 + M3 - 2.0 * M2)
    m_dc = 0.5 * (M1 - M3 - abs(M1 + M3 - 2.0 * M2))
    if m_dc < 0.0:                      # rounding only
        m_dc = 0.0
    norm = abs(m_iso) + abs(m_clvd) + m_dc
    if norm < 1e-30:
        return 0.0, 0.0, 0.0
    ISO = 100.0 * m_iso / norm
    CLVD = 100.0 * m_clvd / norm
    DC = 100.0 * m_dc / norm
    return ISO, CLVD, DC


# -----------------------------
# Main function
# -----------------------------

@njit(parallel=True, fastmath=True, cache=True) 
def silsub(moment_tensors):

    n_sources = moment_tensors.shape[0]

    plane1_out = np.zeros((n_sources, 3))
    plane2_out = np.zeros((n_sources, 3))
    M0_out = np.zeros(n_sources)
    Mw_out = np.zeros(n_sources)
    DC_out = np.zeros(n_sources)
    CLVD_out = np.zeros(n_sources)
    ISO_out = np.zeros(n_sources)
    P_axes_out = np.zeros((n_sources, 2))
    T_axes_out = np.zeros((n_sources, 2))
    B_axes_out = np.zeros((n_sources, 2))
    MT_out = np.zeros((n_sources, 3, 3))

    for k in prange(n_sources):

        A = np.zeros(6)
        for i in range(moment_tensors.shape[1]):
            A[i] = moment_tensors[k, i]

        M = build_moment_tensor(A)
        MT_out[k, :, :] = M
        M, M0, ok = normalize_moment_tensor(M)
        if not ok:
            continue

        Mw = (2.0 / 3.0) * np.log10(M0) - 6.0333

        vals, vecs = eigendecomp_sorted(M)

        ISO, CLVD, DC = decompose(vals)

        T = vecs[:, 0]
        P = vecs[:, 2]

        n1 = (T + P) / np.sqrt(2.0)
        n2 = (T - P) / np.sqrt(2.0)
        s1 = (T - P) / np.sqrt(2.0)
        s2 = (T + P) / np.sqrt(2.0)

        plane1_out[k, :] = angles(n1, s1)
        plane2_out[k, :] = angles(n2, s2)

        M0_out[k] = M0
        Mw_out[k] = Mw
        DC_out[k] = DC
        CLVD_out[k] = CLVD
        ISO_out[k] = ISO

        P_axes_out[k, :] = cart2trendplunge(P)
        T_axes_out[k, :] = cart2trendplunge(T)

        B = np.array([
            P[1]*T[2] - P[2]*T[1],
            P[2]*T[0] - P[0]*T[2],
            P[0]*T[1] - P[1]*T[0]
        ])

        B_axes_out[k, :] = cart2trendplunge(B)

    return (
        plane1_out, plane2_out,
        M0_out, Mw_out,
        DC_out, CLVD_out, ISO_out,
        P_axes_out, T_axes_out, B_axes_out,MT_out
    )


def load_hes_optimized(file_path, nfreq, nr):

    # Initialize Byte info to read .hes files
    BYTES_PER_COMPLEX = 16
    RECORDS_PER_BLOCK = 3
    CHANNELS_PER_RECORD = 6
    MARKER_SIZE = 4

    # 1. Read the entire file as bytes
    with open(file_path, "rb") as f:
        raw_data = f.read()
    
    # 2. Convert to a 1D array of uint8 or similar to handle byte-offsets
    # Alternatively, if markers are consistent, we can calculate the total block size
    # Total size of one "set" of 18 channels:
    # (Marker + 6*Complex + Marker) * 3
    single_record_full_size = MARKER_SIZE + (CHANNELS_PER_RECORD * BYTES_PER_COMPLEX) + MARKER_SIZE
    total_set_size = single_record_full_size * RECORDS_PER_BLOCK
    
   
    # Calculate how many bytes to skip to get to the next record's data
    dt = np.dtype([
        ('m1', 'i4'), 
        ('data', 'c16', (6,)), 
        ('m2', 'i4')
    ])
    
    # View the whole file as a series of Fortran records
    # This happens at C-speed, no Python loops!
    clean_view = np.frombuffer(raw_data, dtype=dt)['data'] # Shape: (N_total_records, 6)
    
    # =====================================================
    # AUTO-DETECT NUMBER OF SOURCES
    # =====================================================

    components = 6

    expected_per_source = nfreq * nr * RECORDS_PER_BLOCK

    total_records = clean_view.shape[0]

    if total_records % expected_per_source != 0:
        raise ValueError(
            f"Invalid HES file structure:\n"
            f"{total_records=} not divisible by "
            f"{expected_per_source=}"
        )

    ns_actual = total_records // expected_per_source

    # 4. Reshape and Reorganize
    # Current shape of clean_view: (ns * nfreq * nr * 3, 6)
    # We want to group the '3' records into the 18 channels
    reorganized = clean_view.reshape(ns_actual, nfreq, nr, 3, 6)
    
    # Transpose to your desired shape: (nfreq, channels, nr, ns)
    # The '3' and '6' combine to form the 18 channels
    final_data = reorganized.transpose(1, 3, 4, 2, 0).reshape(nfreq, 18, nr, ns_actual)
    
    return final_data


def _hes_n_sources(file_path, nfreq, nr):
    """Number of sources in a .hes file, from its size (104-byte records:
    marker + 6 complex values + marker; 3 records per frequency and station)."""
    size = os.path.getsize(file_path)
    per_source = 104 * 3 * nfreq * nr
    if size % per_source:
        raise ValueError('{}: {} bytes is not a whole number of sources ({} bytes each)'.format(
            file_path, size, per_source))
    return size // per_source


def load_hes_into(file_path, nfreq, nr, out5):
    """Read a .hes file straight into out5, a (nfreq, 3, 6, nr, ns) view of the
    final U_all array: the same values as load_hes_optimized, without the
    intermediate (nfreq, 18, nr, ns) copy."""
    dt = np.dtype([('m1', 'i4'), ('data', 'c16', (6,)), ('m2', 'i4')])
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    clean_view = np.frombuffer(raw_data, dtype=dt)['data']
    ns = out5.shape[4]
    if clean_view.shape[0] != ns * nfreq * nr * 3:
        raise ValueError('{}: {} records, expected {}'.format(
            file_path, clean_view.shape[0], ns * nfreq * nr * 3))
    out5[...] = clean_view.reshape(ns, nfreq, nr, 3, 6).transpose(1, 3, 4, 2, 0)


def apply_stf(U_all, nfreq, tl, LOCAL_AW):
    """
    Apply source time function to frequency-domain Green's functions.
    STF type and duration are read from config.yaml (Inversion.STF).
    delta  : no-op (default)
    triangle: symmetric triangle of half-duration t0, total duration 2*t0
    """
    stf_cfg  = config.cfg.get('Inversion', {}).get('STF', {'type': 'delta'})
    stf_type = stf_cfg.get('type', 'delta')

    if stf_type == 'delta':
        return U_all

    t0    = float(stf_cfg.get('duration', 5.0))
    omega = 2 * np.pi * np.arange(nfreq) / tl + 1j * LOCAL_AW
    s     = 1j * omega * t0
    uu    = (1 - np.exp(-s)) / s                        # causal boxcar
    s2    = s / 2
    uxx   = (np.exp(s2) - np.exp(-s2)) / (2 * s2)      # centred boxcar (sinc)
    stf   = uu * uxx                                    # triangle moment rate

    U_all *= stf[:, None, None, None]
    return U_all


def build_G_transposed(U_all, CC, sos, Dt, keep_pairs, components, NFFT, n_workers=None):
    """
    Builds the Greens matrix directly in the (n_sources, n_total, n_comp)
    layout that the inversion kernel needs.

    Numerically IDENTICAL to the previous pipeline
        irfft -> *CC -> sosfilt -> cumsum -> *Dt -> reshape/transpose -> mask
    because every operation acts independently per (channel, source) column,
    so processing in source-chunks and dropping unused channels/components
    BEFORE the FFT cannot change any kept value.

    Speed comes from:
      - slicing the moment-tensor components 6 -> `components` and removed
        channels BEFORE the FFT/filter (less data through every stage)
      - chunking over sources + ThreadPoolExecutor (irfft/sosfilt/cumsum
        release the GIL, so chunks run on multiple cores)
      - writing straight into the final kernel layout (no giant transpose +
        ascontiguousarray copy, no reorder copy inside the kernel)
      - peak memory drops from ~3x the array size to ~1x + small chunks
    """
    nfreq, _, nr, ns_total = U_all.shape
    n_kept = len(keep_pairs)

    # view: (nfreq, 3 records(N/E/Z), components, nr, ns)
    # drops the unused MT components before any heavy work
    U5 = U_all.reshape(nfreq, 3, 6, nr, ns_total)[:, :, :components]
    # only the used (station, record) channels go through irfft/filter/integration:
    # every channel is computed on its own, so the kept ones are unchanged
    rec_idx = np.array([rec for sta, rec in keep_pairs], dtype=np.int64)
    sta_idx = np.array([sta for sta, rec in keep_pairs], dtype=np.int64)

    G = np.empty((ns_total, n_kept * NFFT, components), dtype=np.float64)

    if n_workers is None:
        n_workers = config.n_cores()
    n_chunks = max(1, min(ns_total, 4 * n_workers))
    edges = np.linspace(0, ns_total, n_chunks + 1).astype(np.int64)

    def process(s0, s1):
        # exact same operation sequence as before, on a source-slice and on the
        # used channels only: (n_kept, nfreq, comp, nc) -> time axis first
        Us = np.moveaxis(U5[:, rec_idx, :, sta_idx, s0:s1], 0, 1)   # (nfreq,kept,comp,nc)
        ts = irfft(Us, n=NFFT, axis=0) * NFFT               # (NFFT,kept,comp,nc)
        ts *= CC[:, None, None, None]
        # second-order sections, like XAPIIR in ISOLA15 (ba form is
        # ill-conditioned for 8-pole band-pass at low normalized corners)
        ts = sosfilt(sos, ts, axis=0)
        np.cumsum(ts, axis=0, out=ts)
        ts *= Dt
        for row in range(n_kept):
            # (NFFT, comp, nc) -> (nc, NFFT, comp)
            G[s0:s1, row * NFFT:(row + 1) * NFFT, :] = \
                ts[:, row, :, :].transpose(2, 0, 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(process, int(edges[i]), int(edges[i + 1]))
                for i in range(n_chunks) if edges[i] < edges[i + 1]]
        for f in futs:
            f.result()   # re-raise any worker exception

    return G


@njit(fastmath=True, cache=True)
def _data_power(obs_data, Dt):
    """rr = Dt * d^T d over the whole data vector (denominator of VR), shared by the
    full inversion and the graph search."""
    rr = 0.0
    for i in range(obs_data.shape[0]):
        rr += obs_data[i] * obs_data[i]
    return rr * Dt


@njit(fastmath=True, cache=True)
def _point_inversion(Gk, obs_data, shifts_array, scale, MULTIP, rr, R, vr_row, me_row):
    """Inversion of ONE grid point for every time shift (full inversion and graph search):
        R = scale * G_k^T G_k               (unshifted, all samples - ISOLA's ELEMAT),
            factored once: R = L L^T (Cholesky; Gaussian elimination with partial
            pivoting for every shift if R is not positive definite)
        g(s) = scale * G_k(s)^T d,  L y = g,  L^T a = y,  VR = a.g / MULTIP / rr * 100
    G_k (n_total, n_comp) stays in cache for all shifts. Writes R (n_comp, n_comp),
    vr_row (n_shifts,) and me_row (n_shifts, n_comp)."""
    n_total    = Gk.shape[0]
    nc         = Gk.shape[1]
    n_stations = n_total // 1024
    n_shifts   = shifts_array.shape[0]
    for a in range(nc):
        for b in range(nc):
            R[a, b] = 0.0
    for i in range(n_total):
        for a in range(nc):
            for b in range(nc):
                R[a, b] += Gk[i, a] * Gk[i, b]
    for a in range(nc):
        for b in range(nc):
            R[a, b] *= scale
    L = np.empty((nc, nc))
    A = np.empty((nc, nc))
    g = np.empty(nc)
    y = np.empty(nc)
    x = np.empty(nc)
    chol = True
    for a in range(nc):
        for b in range(a + 1):
            s = R[a, b]
            for m in range(b):
                s -= L[a, m] * L[b, m]
            if a == b:
                if s <= 0.0:
                    chol = False
                    break
                L[a, a] = math.sqrt(s)
            else:
                L[a, b] = s / L[b, b]
        if not chol:
            break
    for si in range(n_shifts):
        sh = shifts_array[si]
        for n in range(nc):
            g[n] = 0.0
        for stat in range(n_stations):
            base = stat * 1024
            for t in range(1024):
                gf_idx = t - sh
                if 0 <= gf_idx < 1024:
                    d = obs_data[base + t]
                    for n in range(nc):
                        g[n] += Gk[base + gf_idx, n] * d
        for n in range(nc):
            g[n] *= scale
        if chol:
            for a in range(nc):
                s = g[a]
                for m in range(a):
                    s -= L[a, m] * y[m]
                y[a] = s / L[a, a]
            for a in range(nc - 1, -1, -1):
                s = y[a]
                for m in range(a + 1, nc):
                    s -= L[m, a] * x[m]
                x[a] = s / L[a, a]
        else:
            for a in range(nc):
                for b in range(nc):
                    A[a, b] = R[a, b]
                y[a] = g[a]
            for col in range(nc):
                piv = col
                for row in range(col + 1, nc):
                    if abs(A[row, col]) > abs(A[piv, col]):
                        piv = row
                if piv != col:
                    for b in range(nc):
                        t2 = A[col, b]; A[col, b] = A[piv, b]; A[piv, b] = t2
                    t2 = y[col]; y[col] = y[piv]; y[piv] = t2
                for row in range(col + 1, nc):
                    f = A[row, col] / A[col, col]
                    for b in range(col, nc):
                        A[row, b] -= f * A[col, b]
                    y[row] -= f * y[col]
            for a in range(nc - 1, -1, -1):
                s = y[a]
                for m in range(a + 1, nc):
                    s -= A[a, m] * x[m]
                x[a] = s / A[a, a]
        sum1 = 0.0
        for n in range(nc):
            sum1 += x[n] * g[n]
        vr_row[si] = ((sum1 / MULTIP) / rr) * 100.0
        for n in range(nc):
            me_row[si, n] = x[n]


@njit(parallel=True, fastmath=True, cache=True)
def compute_inversion_numbaCpu(G, obs_data, Dt, shifts_array, num_stations):
    """
    Full inversion: every grid point (in parallel) x every time shift.
    G: (n_sources, n_total, n_comp), C-contiguous (built by build_G_transposed).
    Each grid point is inverted for all shifts at once by _point_inversion()
    (the same routine as the graph search): its G_k is read from memory once
    and reused for all shifts from the cache, and R_k is factored once.
    Returns (best_shift, best_mech, best_idx, best_vr, best_corr, svals,
    cond_nums, vr_mat, mech_mat); vr_mat / mech_mat (n_sources, n_shifts[,
    n_comp]) hold the VR and moment tensor of every (source, shift) pair, as
    ISOLA writes them to corr00.dat (correlations file, FMVAR).
    """
    MULTIP = 1e20
    VARDAT = 2.0e-12  # fixed prior variance (matches Fortran ISOLA15)
    n_sources, n_total, n_comp = G.shape
    n_shifts = shifts_array.shape[0]
    scale = MULTIP * Dt
    rr = _data_power(obs_data, Dt)

    R = np.empty((n_sources, n_comp, n_comp))
    vr_mat = np.empty((n_sources, n_shifts))          # VR for every (source, shift)
    mech_mat = np.empty((n_sources, n_shifts, n_comp))   # MT for every (source, shift)
    svals = np.zeros((n_sources, n_comp))
    cond_nums = np.zeros(n_sources)
    best_vr_per_source = np.full(n_sources, -1.0)
    best_corr_per_source = np.full(n_sources, -1.0)
    best_shift_per_source = np.zeros(n_sources, dtype=np.int64)
    best_mech_per_source = np.zeros((n_sources, n_comp))

    for k in prange(n_sources):
        _point_inversion(G[k], obs_data, shifts_array, scale, MULTIP, rr,
                         R[k], vr_mat[k], mech_mat[k])
        # singular values and condition number of R_k
        eigvals = np.linalg.eigvalsh(R[k])
        svals_k = np.sqrt(eigvals / VARDAT) / 1e10
        svals[k, :] = svals_k
        cond_nums[k] = svals_k.max() / svals_k.min()
        # best shift of this grid point (first maximum)
        for si in range(n_shifts):
            vr = vr_mat[k, si]
            if vr > best_vr_per_source[k]:
                best_vr_per_source[k] = vr
                best_shift_per_source[k] = shifts_array[si]
                best_corr_per_source[k] = np.sqrt(max(vr, 0.0) / 100.0)
                for n in range(n_comp):
                    best_mech_per_source[k, n] = mech_mat[k, si, n]

    # global best (first maximum)
    best_idx = 0
    bv = best_vr_per_source[0]
    for k in range(1, n_sources):
        if best_vr_per_source[k] > bv:
            bv = best_vr_per_source[k]
            best_idx = k

    return (
        best_shift_per_source,
        best_mech_per_source,
        best_idx,
        best_vr_per_source,
        best_corr_per_source,
        svals,
        cond_nums,
        vr_mat,
        mech_mat
    )



def _select_top_pairs(vr_all, mech_all, shifts_array, Coord, n_keep):
    """
    ISOLA-BaBoo selection for one perturbation: rank ALL (source, shift)
    pairs by VR and keep the top n_keep (NBEST). A source may enter several
    times with different shifts. Shared by the CPU and GPU paths.
    The order is that of a stable sort of -VR (ties by pair index): the
    n_keep-th value is found with a partition, then only the pairs up to it
    (all ties included, in index order) are sorted stably - the same pairs
    in the same order as the full sort, without sorting all pairs.
    Returns (shifts, source_idx, mechs, vrs, coords), one row per member.
    """
    n_shifts = vr_all.shape[1]
    v = -vr_all.ravel()
    order = None
    if n_keep < v.size:
        kth = np.partition(v, n_keep - 1)[n_keep - 1]
        if not np.isnan(kth):
            cand = np.flatnonzero(v <= kth)
            order = cand[np.argsort(v[cand], kind='stable')][:n_keep]
    if order is None:
        order = np.argsort(v, kind='stable')[:n_keep]
    k_idx, s_idx = np.divmod(order, n_shifts)
    return (
        np.asarray(shifts_array)[s_idx],
        k_idx,
        mech_all[k_idx, s_idx],
        vr_all[k_idx, s_idx],
        Coord[k_idx],
    )


@njit(fastmath=True, cache=True)
def _trace_power(obs_data, n_traces):
    """dd[c] = d_c^T d_c of every 1024-sample trace (bootstrap data norms); shared by
    the CPU and GPU precompute, so both use bit-identical values."""
    dd = np.zeros(n_traces)
    for c in range(n_traces):
        base = c * 1024
        acc = 0.0
        for t in range(1024):
            acc += obs_data[base + t] * obs_data[base + t]
        dd[c] = acc
    return dd


@njit(parallel=True, fastmath=True, cache=True)
def _inversion_summary(vr_mat, mech_mat, R, shifts_array):
    """Outputs of the GPU full inversion from its per-pair VR / MT and R (scaled):
    singular values, condition numbers, best shift per grid point and the global best -
    the same rules as compute_inversion_numbaCpu (first maximum)."""
    VARDAT = 2.0e-12
    n_sources, n_shifts, n_comp = mech_mat.shape
    svals = np.zeros((n_sources, n_comp))
    cond_nums = np.zeros(n_sources)
    best_vr_per_source = np.full(n_sources, -1.0)
    best_corr_per_source = np.full(n_sources, -1.0)
    best_shift_per_source = np.zeros(n_sources, dtype=np.int64)
    best_mech_per_source = np.zeros((n_sources, n_comp))
    for k in prange(n_sources):
        eigvals = np.linalg.eigvalsh(R[k])
        svals_k = np.sqrt(eigvals / VARDAT) / 1e10
        svals[k, :] = svals_k
        cond_nums[k] = svals_k.max() / svals_k.min()
        for si in range(n_shifts):
            vr = vr_mat[k, si]
            if vr > best_vr_per_source[k]:
                best_vr_per_source[k] = vr
                best_shift_per_source[k] = shifts_array[si]
                best_corr_per_source[k] = np.sqrt(max(vr, 0.0) / 100.0)
                for n in range(n_comp):
                    best_mech_per_source[k, n] = mech_mat[k, si, n]
    best_idx = 0
    bv = best_vr_per_source[0]
    for k in range(1, n_sources):
        if best_vr_per_source[k] > bv:
            bv = best_vr_per_source[k]
            best_idx = k
    return (best_shift_per_source, best_mech_per_source, best_idx, best_vr_per_source,
            best_corr_per_source, svals, cond_nums, vr_mat, mech_mat)


@njit(parallel=True, fastmath=True, cache=True)
def bootstrap_precompute(G, obs_data, shifts_array):
    """Weight-free per-trace sums of the bootstrap normal equations, computed once
    for all perturbations:
        Rt[k, c]     = G_kc^T G_kc over the whole trace c (unshifted, as ISOLA's ELEMAT)
        Ct[k, si, c] = sum_t G_kc(t - s) d_c(t) over the window of shift s
        dd[c]        = sum_t d_c(t)^2
    A perturbation with trace weights w_c then has R_k = sum_c w_c Rt[k, c],
    g_k(s) = sum_c w_c Ct[k, s, c] and d^T W d = sum_c w_c dd[c] - the weighted
    sums over all samples, grouped per trace (exact up to rounding)."""
    n_sources, n_total, n_comp = G.shape
    n_traces = n_total // 1024
    n_shifts = shifts_array.shape[0]
    Rt = np.zeros((n_sources, n_traces, n_comp, n_comp))
    Ct = np.zeros((n_sources, n_shifts, n_traces, n_comp))
    dd = _trace_power(obs_data, n_traces)
    for k in prange(n_sources):
        Gk = G[k]
        for c in range(n_traces):
            base = c * 1024
            for t in range(1024):
                for a in range(n_comp):
                    ga = Gk[base + t, a]
                    for b in range(n_comp):
                        Rt[k, c, a, b] += ga * Gk[base + t, b]
            for si in range(n_shifts):
                s = shifts_array[si]
                t_start = 0 if s < 0 else s
                t_end = 1024 if s >= 0 else 1024 + s
                for t in range(t_start, t_end):
                    val = obs_data[base + t]
                    for n in range(n_comp):
                        Ct[k, si, c, n] += Gk[base + t - s, n] * val
    return Rt, Ct, dd


_boot_combine_task_cpu = njit(fastmath=True, cache=True)(_boot_combine_task)


@njit(parallel=True, fastmath=True, cache=True)
def bootstrap_combine_all(Rt, Ct, Wc, rr, scale, MULTIP):
    """All perturbations (trace weights Wc (n_iter, n_traces), rr = Dt d^T W d per
    perturbation) in one parallel call, one task per (perturbation, source):
    _boot_combine_task() - the same routine as the GPU kernel. Returns vr_all
    (n_iter, n_sources, n_shifts) and mech_all (n_iter, n_sources, n_shifts, n_comp)."""
    n_iter = Wc.shape[0]
    n_sources, n_shifts, n_traces, n_comp = Ct.shape
    vr_all = np.empty((n_iter, n_sources, n_shifts))
    mech_all = np.empty((n_iter, n_sources, n_shifts, n_comp))
    for job in prange(n_iter * n_sources):
        i = job // n_sources
        k = job % n_sources
        R = np.empty((6, 6))
        L = np.empty((6, 6))
        A = np.empty((6, 6))
        g = np.empty(6)
        y = np.empty(6)
        x = np.empty(6)
        _boot_combine_task_cpu(Rt, Ct, Wc[i], k, rr[i], scale, MULTIP,
                               R, L, A, g, y, x, vr_all[i, k], mech_all[i, k])
    return vr_all, mech_all


def _bootstrap_members(Gboot, Dobs, Wc, Dt, shifts_array, Coords, n_keep, use_gpu,
                       mem_budget=1 << 30):
    """Bootstrap iterations with trace weights Wc (n_iter, n_traces), CPU or GPU:
    the weight-free per-trace sums are computed once (bootstrap_precompute on the CPU,
    _bootstrap_precompute_gpu on the GPU),
    the perturbations are combined with them in one parallel call per batch
    (bootstrap_combine_all on the CPU, _bootstrap_combine_kernel on the GPU - the same
    _boot_combine_task), and the NBEST pairs of each perturbation are selected on the
    host. Batches keep the per-pair VR/MT arrays within mem_budget bytes.
    Returns one _select_top_pairs() tuple per iteration."""
    MULTIP = 1e20
    n_sources, n_total, n_comp = Gboot.shape
    n_shifts = shifts_array.shape[0]
    if use_gpu:
        Rt, Ct, dd = _bootstrap_precompute_gpu(Gboot, Dobs, shifts_array)   # device Rt, Ct
    else:
        Rt, Ct, dd = bootstrap_precompute(Gboot, Dobs, shifts_array)
    Wc = np.ascontiguousarray(Wc, dtype=np.float64)
    rr = (Wc @ dd) * Dt                                 # Dt d^T W d per perturbation
    scale = Dt * MULTIP
    per_iter = 8 * n_sources * n_shifts * (1 + n_comp)
    batch = max(1, min(Wc.shape[0], mem_budget // per_iter))
    if use_gpu:
        batches = _bootstrap_combine_gpu(Rt, Ct, Wc, rr, scale, MULTIP, batch)
    else:
        batches = ((i0,) + bootstrap_combine_all(Rt, Ct, Wc[i0:i0 + batch], rr[i0:i0 + batch],
                                                 scale, MULTIP)
                   for i0 in range(0, Wc.shape[0], batch))
    results = []
    for _, vr_all, mech_all in batches:
        for i in range(vr_all.shape[0]):
            results.append(_select_top_pairs(vr_all[i], mech_all[i], shifts_array, Coords, n_keep))
    return results


def initialize_inversion(job, config):
    """
    Minimal initialization step:
    - reads station file
    - builds Dobs
    - computes masks
    - builds filter + time parameters
    """

    # =========================================================
    # 1. Extract paths from job
    # =========================================================
    

    allstatPath = Path(job["allstat"])

    # =========================================================
    # 2. Read station configuration (allstat)
    # =========================================================
    Num_Stations = []
    Num_Channels = []
    Removed_Channels = []
    Station_Flags = {}

    allstat_lines = []
    with open(allstatPath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            allstat_lines.append(line)
            parts = line.split()
            # a station is used only if its "use" flag (col 2) is set,
            # as STAT(IR) in ISOLA15; component flags alone are not enough
            flags = np.array(list(map(int, parts[2:5]))) * int(parts[1])

            Removed_Channels.append(np.where(flags == 0)[0])
            Station_Flags[parts[0]] = flags
            Num_Stations.append(parts[0])
            Num_Channels.append(np.sum(flags))
            f1 = float(parts[5])
            f2 = float(parts[6])

    Num_Stations = np.array(Num_Stations)
    Num_Channels = np.array(Num_Channels)

    StationsUsed = Num_Stations[Num_Channels > 0]
    ChannelsUsed = Num_Channels[Num_Channels > 0]

    # =========================================================
    # 3. Read grdat configuration
    # =========================================================
    grdatPath = Path(job["grdat"])
    with open(grdatPath, "r") as f:
        for line in f:
            if "nfreq" in line:
                nfreq = int(line.split("=")[1])
            elif "tl" in line:
                tl = float(line.split("=")[1])
            elif "aw" in line:
                aw = float(line.split("=")[1])

    # =========================================================
    # 3. Load waveform data (Dobs)
    # =========================================================
    # Read from allstat file F1 F4. Compute the normalized frequencies
    Dt = tl/1024 # Dt to use ro resampling
    fs = 1/Dt # Fs to use for filter design

    f1 = _check_band(f1, f2, tl, warn=True)   # fmin 0 -> 1/tl; e.g. revise -f above Nyquist

    # the Green's functions contain the frequencies k/tl, k = 0 .. nfreq-1, so they
    # reach (nfreq-1)/tl; a band reaching close to (or beyond) that limit gives
    # truncated synthetics (e.g. revise with -f above the band the Green's
    # functions were made for, or a small NyquistSamples)
    _f_gf = (nfreq - 1) / tl
    _gf_mult = _green_freq_settings()[1]
    # frequencies the standard rule gives for this band (createGrdat, as old Gisola)
    _nf_std = min(int(np.ceil(tl * _gf_mult * f2)), NUM_OF_TIME_SAMPLES // 2 + 1)
    if f2 >= _f_gf:
        _msg = ('Filter band {:.4f}-{:.4f} Hz exceeds the Green\'s functions\' maximum '
                'frequency {:.4f} Hz (nfreq={}, tl={}): synthetics are truncated'.format(
                    f1, f2, _f_gf, nfreq, tl))
        config.logger.warning(_msg); print('[WARN] ' + _msg, flush=True)
    elif nfreq < _nf_std:
        _msg = ('Filter corner {:.4f} Hz is close to the Green\'s functions\' maximum '
                'frequency {:.4f} Hz (< FreqMultiplier {:g} x corner): the filter roll-off '
                'above the band is cut off'.format(f2, _f_gf, _gf_mult))
        config.logger.warning(_msg); print('[WARN] ' + _msg, flush=True)
    low = f1 / (0.5 * fs)
    high = f2 / (0.5 * fs)
    sos = butter(4, [low, high], btype='band', output='sos')
    # resample a copy: config.st must stay untouched, since this runs once
    # per combination (the old createRaw also resampled a copy)
    D = config.st.copy()
    D.resample(1024 / tl)

    CHANNEL_ORDER = ["N", "E", "Z"]
    ordered_signals = []

    for sta in StationsUsed:
        sta_traces = D.select(station=sta)

        chan_map = {}
        for tr in sta_traces:
            velocityObs = sosfilt(sos, tr.data)
            dt = tr.stats.delta
            displacement = np.cumsum(velocityObs) * dt
            chan_map[tr.stats.channel[-1]] = displacement

        # Dobs must hold exactly the channels kept in G (allstat flag == 1)
        for ic, ch in enumerate(CHANNEL_ORDER):
            if not Station_Flags[sta][ic]:
                continue
            if ch not in chan_map:
                raise ValueError(
                    'Station {} component {} is enabled in allstat but has '
                    'no trace'.format(sta, ch))
            if len(chan_map[ch]) < 1024:
                raise ValueError(
                    'Station {} component {}: only {} samples after '
                    'resampling (1024 required)'.format(
                        sta, ch, len(chan_map[ch])))
            ordered_signals.append(chan_map[ch][:1024])

    Dobs = np.concatenate(ordered_signals)

    # =========================================================
    # 4. Build removeIdx mask
    # =========================================================
    removeIdx = []

    for i, rmv in enumerate(Removed_Channels):
        if len(rmv) > 0:
            for shift in rmv:
                step = i * 3 * 1024
                removeIdx.append(np.arange(step + shift * 1024,
                                           step + (shift + 1) * 1024))

    if len(removeIdx) > 0:
        removeIdx = np.concatenate(removeIdx)
    else:
        removeIdx = np.array([], dtype=int)

    # =========================================================
    # 5. Read inpinv - Create Timeshifts
    # =========================================================
    inpinvPath = Path(job["inpinv"])

    with open(inpinvPath, "r") as f:
        tmin, tstep, tmax = map(int, f.readline().split())
   
    Timeshifts = np.arange(tmin,tmax + 1,tstep).astype(int)

    # =========================================================
    # 6. Attenuation correction
    # =========================================================
    NFFT = 1024

    LOCAL_AW = -np.pi * aw / tl

    CK = np.arange(NFFT) / NFFT
    CC = np.exp(-LOCAL_AW * tl * CK) / tl

    CCm = CC[:, None, None, None]



    # =========================================================
    # 8. Basic metadata
    # =========================================================
    nr = len(Num_Channels)

    # =========================================================
    # 9. Package everything into setup object
    # =========================================================
    setup = {

        "Dobs": Dobs,
        "removeIdx": removeIdx,
        "StationsUsed": StationsUsed,
        "ChannelsUsed": ChannelsUsed,
        "Timeshifts": Timeshifts,
        "CCm": CCm,
        "nr": nr,
        "NFFT": NFFT,
        "sos": sos,
        "Dt": Dt,
        "Fs": fs,
        "nfreq": nfreq,
        "tstep": tstep,
        "allstat_lines": allstat_lines,
        "tl": tl,
        "LOCAL_AW": LOCAL_AW,

    }

    return setup

def _hudson_uv(M_all):
    """Hudson (u, v) for a stack of moment tensors (n, 3, 3) at once.
    Same formula as pyrocko.plot.hudson.project, vectorised."""
    eig = np.linalg.eigvalsh(M_all)                      # ascending: m3, m2, m1
    eig = eig / np.abs(eig).max(axis=1, keepdims=True)
    m3, m2, m1 = eig[:, 0], eig[:, 1], eig[:, 2]
    u = -2. / 3. * (m1 + m3 - 2.0 * m2)
    v = 1. / 3. * (m1 + m2 + m3)
    return u, v


def _lune_coords(M_all):
    """Lune longitude gamma and latitude delta (degrees) of Tape & Tape (2012)
    for a stack of moment tensors (n, 3, 3), vectorised. With the
    eigenvalues sorted l1 >= l2 >= l3:
        tan(gamma) = (-l1 + 2 l2 - l3) / (sqrt(3) (l1 - l3)),  gamma in [-30, 30]
        cos(beta)  = (l1 + l2 + l3) / (sqrt(3) |l|),  delta = 90 - beta
    DC is at (0, 0), +/-ISO at the poles, CLVDs at gamma = -/+30, delta = 0."""
    lam = np.linalg.eigvalsh(M_all)[:, ::-1]               # l1 >= l2 >= l3
    l1, l2, l3 = lam[:, 0], lam[:, 1], lam[:, 2]
    den = np.sqrt(3.0) * (l1 - l3)
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = np.where(den > 0, np.degrees(np.arctan((-l1 + 2 * l2 - l3) / den)), 0.0)
        cosb = (l1 + l2 + l3) / (np.sqrt(3.0) * np.linalg.norm(lam, axis=1))
    delta = 90.0 - np.degrees(np.arccos(np.clip(cosb, -1.0, 1.0)))
    return gamma, delta


def _hammer_xy(gamma_deg, delta_deg):
    """Hammer (equal-area) projection of lune coordinates, as used for lune
    plots (GMT -JH): x from longitude gamma, y from latitude delta."""
    g = np.radians(np.asarray(gamma_deg, dtype=float))
    d = np.radians(np.asarray(delta_deg, dtype=float))
    den = np.sqrt(1.0 + np.cos(d) * np.cos(g / 2.0))
    return (2.0 * np.sqrt(2.0) * np.cos(d) * np.sin(g / 2.0) / den,
            np.sqrt(2.0) * np.sin(d) / den)


def _lune_plot(M_all, VR, median_M, best_M, outpath):
    """Lune source-type plot (Tape & Tape 2012) of the whole bootstrap
    ensemble, Hammer equal-area projection, members coloured by VR."""
    fig, ax = plt.subplots(figsize=(5.5, 8))

    # graticule: meridians every 10 deg (boundary at +/-30), parallels every 15 deg
    dd = np.linspace(-90, 90, 361)
    for g in (-30, -20, -10, 0, 10, 20, 30):
        x, y = _hammer_xy(np.full_like(dd, g), dd)
        ax.plot(x, y, color='black' if abs(g) == 30 else '0.8',
                lw=1.5 if abs(g) == 30 else 0.6, zorder=1)
    gg = np.linspace(-30, 30, 121)
    for d in range(-75, 90, 15):
        x, y = _hammer_xy(gg, np.full_like(gg, d))
        ax.plot(x, y, color='0.6' if d == 0 else '0.85', lw=0.9 if d == 0 else 0.6, zorder=1)

    # ensemble, best-fitting members on top
    gam, dlt = _lune_coords(M_all)
    x, y = _hammer_xy(gam, dlt)
    VR = np.asarray(VR, dtype=float)
    order = np.argsort(VR)
    sc = ax.scatter(x[order], y[order], c=VR[order], cmap='viridis',
                    s=10 if len(VR) <= 5000 else 3, alpha=0.6,
                    edgecolors='none', rasterized=True, zorder=2)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.08)
    cbar.set_label('Variance Reduction (%)')

    # reference source types, placed from their eigenvalues (l1, l2, l3)
    refs = [('DC', (1, 0, -1)), ('+ISO', (1, 1, 1)), ('-ISO', (-1, -1, -1)),
            ('+CLVD', (2, -1, -1)), ('-CLVD', (1, 1, -2)),
            ('+LVD', (1, 0, 0)), ('-LVD', (0, 0, -1)),
            ('+crack', (3, 1, 1)), ('-crack', (-1, -1, -3))]   # tensile crack, Poisson 0.25
    for label, lam in refs:
        rg, rd = _lune_coords(np.diag(np.asarray(lam, dtype=float))[None])
        rx, ry = _hammer_xy(rg, rd)
        ax.plot(rx, ry, 'o', color='black', ms=5, zorder=4)
        ha = 'right' if rg[0] < -1 else ('left' if rg[0] > 1 else 'center')
        dx = -0.04 if ha == 'right' else (0.04 if ha == 'left' else 0.0)
        if abs(rd[0]) > 85:                                   # poles: above / below
            dy = 0.06 if rd[0] > 0 else -0.1
        else:                                                 # DC: just above its dot
            dy = 0.07 if ha == 'center' else 0.0
        ax.text(rx[0] + dx, ry[0] + dy, label, ha=ha, va='center', fontsize=9, zorder=5)

    # best solution, then the median moment tensor on top of it
    for M, mk, col, lab, z in ((best_M, 's', 'black', 'Best solution', 6),
                               (median_M, '*', 'red', 'Median MT', 7)):
        if M is None:
            continue
        mg, md = _lune_coords(np.asarray(M, dtype=float)[None])
        mx, my = _hammer_xy(mg, md)
        ax.plot(mx, my, mk, color=col, ms=16 if mk == '*' else 8,
                markeredgecolor='black' if mk == '*' else 'white', zorder=z, label=lab)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.85)

    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title('Lune source-type plot (Tape & Tape 2012)\n'
                 'Bootstrap ensemble, N = %d' % len(VR), fontsize=17, fontweight='bold')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


def _fuzzy_amplitudes(M_all, vecs3, chunk=2000):
    """Fraction of ensemble members with positive radiation v^T M_dev v at
    each focal-sphere direction vecs3 (P, 3; NED). Equivalent to pyrocko's
    mts2amps(..., mask=True) on the deviatoric members, vectorised: the
    quadratic form is one matrix product B @ m6 per chunk of members.
    The product runs in single precision (only its sign is used; a pixel can
    differ from double precision only where a member's nodal line passes
    within ~1e-7 of it)."""
    x, y, z = vecs3[:, 0], vecs3[:, 1], vecs3[:, 2]
    B = np.column_stack([x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z]
                        ).astype(np.float32)
    count = np.zeros(vecs3.shape[0], dtype=np.int64)
    for k0 in range(0, M_all.shape[0], chunk):
        M = M_all[k0:k0 + chunk]
        iso = np.trace(M, axis1=1, axis2=2) / 3.0            # deviatoric part
        m6 = np.column_stack([M[:, 0, 0] - iso, M[:, 1, 1] - iso, M[:, 2, 2] - iso,
                              M[:, 0, 1], M[:, 0, 2], M[:, 1, 2]])
        # normalise each member (sign-preserving) so float32 never sees 1e17-scale values
        m6 /= np.maximum(np.abs(m6).max(axis=1, keepdims=True), 1e-300)
        count += ((B @ m6.T.astype(np.float32)) > 0.0).sum(axis=1)
    return count / float(M_all.shape[0])


def _plot_fuzzy_beachball_all(M_all, ax, best_mt, size=250, position=(0, 0),
                              color_t='black', color_p='white', edgecolor='black',
                              best_color='red', linewidth=2, grid_resolution=200,
                              extra_mts=()):
    """Fuzzy beachball of ALL ensemble members. Same grid, projection
    (Lambert, top view), colour map and best-solution contour as
    pyrocko's plot_fuzzy_beachball_mpl_pixmap(method='imshow'), but the
    amplitudes are computed vectorised by _fuzzy_amplitudes()."""
    from matplotlib.colors import LinearSegmentedColormap
    transform, position, size = beachball.choose_transform(ax, 'data', position, size)
    nx = ny = grid_resolution
    xg = np.linspace(-1., 1., nx)
    yg = np.linspace(-1., 1., ny)
    vecs2 = np.zeros((nx * ny, 2))
    vecs2[:, 0] = np.tile(xg, ny)
    vecs2[:, 1] = np.repeat(yg, nx)
    ii_ok = vecs2[:, 0] ** 2 + vecs2[:, 1] ** 2 <= 1.0
    amps = np.full(nx * ny, np.nan)
    amps[ii_ok] = _fuzzy_amplitudes(M_all, beachball.inverse_project(vecs2[ii_ok], 'lambert'))
    amps = amps.reshape(ny, nx)

    cmap = LinearSegmentedColormap.from_list('fuzzy', [color_p, color_t], N=256)
    ax.imshow(amps.T,
              extent=(position[0] + yg[0] * size, position[0] + yg[-1] * size,
                      position[1] - xg[0] * size, position[1] - xg[-1] * size),
              cmap=cmap, transform=transform, zorder=-0.1)
    # nodal lines of the best solution and of any extra tensors (mt, colour)
    for _mt, _col in ([(best_mt, best_color)] if best_mt is not None else []) + list(extra_mts):
        _amps, bx, by = beachball.mts2amps([_mt], grid_resolution=grid_resolution,
                                           projection='lambert', beachball_type='full',
                                           mask=False)
        ax.contour(position[0] + by * size, position[1] + bx * size, _amps.T,
                   levels=[0.], colors=[_col], linewidths=linewidth,
                   transform=transform, zorder=0)
    phi = np.linspace(0., 2 * np.pi, 361)
    ax.plot(position[0] + np.cos(phi) * size, position[1] + np.sin(phi) * size,
            linewidth=linewidth, color=edgecolor, transform=transform, zorder=0)


def _hudson_cloud_plot(u_vals, v_vals, VR_plot, center_mt, label, outpath):
    """Hudson projection of the full bootstrap ensemble (precomputed u, v)."""
    norm = mcolors.Normalize(vmin=VR_plot.min(), vmax=VR_plot.max())
    cmap = plt.get_cmap('viridis')

    fig, ax = plt.subplots(figsize=(8, 8))
    hudson.draw_axes(ax)

    sort_idx = np.argsort(VR_plot)  # higher VR plotted on top
    n_pts = len(VR_plot)
    ax.scatter(u_vals[sort_idx], v_vals[sort_idx],
               c=cmap(norm(VR_plot[sort_idx])),
               s=10 if n_pts <= 5000 else 3, alpha=0.6,
               edgecolors='none', rasterized=True)

    u_b, v_b = hudson.project(center_mt)
    beachball.plot_beachball_mpl(
        center_mt, ax, position=(u_b, v_b), size=15,
        color_t='red', color_p='white', edgecolor='black',
        linewidth=1.0, alpha=1.0, zorder=10)
    ax.text(u_b, v_b + 0.05, label, color='red', fontweight='bold',
            ha='center', va='bottom', fontsize=9)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Variance Reduction (%)', rotation=270, labelpad=15)

    ax.set_aspect('equal')
    ax.set_title("Hudson Plot: Bootstrap Cloud with " + label,
                 fontsize=17, fontweight='bold')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


def _fmt_nz(fmt, v):
    """fmt.format(v) without a negative zero (e.g. -0.4 with {:.0f} gives '0', not '-0')."""
    s = fmt.format(v)
    try:
        if s.startswith('-') and float(s) == 0.0:
            s = s[1:]
    except ValueError:
        pass
    return s


def _ci_stats(values):
    """Median, 68% CI (16th-84th pct) and 95% CI (2.5th-97.5th pct) of an
    ensemble quantity, as reported in ISOLA-BaBoo (Zahradnik et al. 2026).
    Returns (median, p16, p84, p2.5, p97.5)."""
    v = np.asarray(values, dtype=float)
    return (float(np.median(v)),) + tuple(
        float(p) for p in np.percentile(v, [15.87, 84.13, 2.5, 97.5]))


def _centroid_box_lines(info):
    """Human-readable description of the bootstrap centroid box (rule + limits)."""
    r = info.get('rule')
    if r == 'CentroidBox':
        out = ['Rule: CentroidBox, magnitude rule {:g}-{:g}: Dx=+/-{:g} Dy=+/-{:g} Dz=+/-{:g} km '
               'around the best centroid'.format(info['mw_rule'][0], info['mw_rule'][1],
                                                 info['Dx'], info['Dy'], info['Dz'])]
    elif r == 'CentroidBoxVR':
        out = ['Rule: CentroidBoxVR, grid points with VR >= {:g} x VRmax {:.2f}% = {:.2f}% '
               '({} points)'.format(info['fraction'], info['vrmax'], info['threshold'],
                                    info['n_qualifying'])]
    else:
        out = ['Rule: full grid']
    if 'box' in info:
        b = info['box']
        out.append('Box: X [{:.2f}, {:.2f}] km  Y [{:.2f}, {:.2f}] km  Z [{:.2f}, {:.2f}] km  '
                   'T [{:.2f}, {:.2f}] s'.format(*b, *info['t']))
        out.append('Size: {:.2f} x {:.2f} x {:.2f} km, {} of {} grid points ({:.1f}%)'.format(
            b[1] - b[0], b[3] - b[2], b[5] - b[4], info['n_inbox'], info['n_total'],
            100.0 * info['n_inbox'] / info['n_total'] if info['n_total'] else 0.0))
    return out


def _bootstrap_iterations(default=100):
    """Number of bootstrap perturbations (= sets of station weights) from
    Bootstrap.Iterations. A missing key gives the default (100); an invalid value
    (not a whole number >= 1) falls back to the default with a warning."""
    _v = (config.cfg.get('Bootstrap') or {}).get('Iterations')
    if _v is None:
        return default
    try:
        _n = int(_v)
        if isinstance(_v, bool) or _n < 1 or _n != float(_v):
            raise ValueError(_v)
    except (TypeError, ValueError):
        _m = 'Bootstrap.Iterations {!r} is not a whole number >= 1; using {}'.format(_v, default)
        config.logger.warning(_m); print('[WARN] ' + _m, flush=True)
        return default
    return _n


@config.time
def getBootstrapping(num_iterations=None, num_realizations=None):
    """
    Bootstrap uncertainty analysis.
    Reuses the Greens matrix and inversion results kept in memory by
    run_python_inversion(); all figures are saved to <workdir>/output/plots.
    """
    if _BOOT_CTX is None:
        config.logger.info('No inversion in memory - run getInversions() first')
        return

    startBoot = time.perf_counter()

    # number of perturbations = number of different sets of station weights:
    # Bootstrap.Iterations (default 100)
    if num_iterations is None:
        num_iterations = _bootstrap_iterations()
    if num_realizations is None:
        num_realizations = num_iterations

    G = _BOOT_CTX["G"]
    vr = _BOOT_CTX["vr"]
    shift = _BOOT_CTX["shift"]
    best_idx = int(_BOOT_CTX["best_idx"])
    Dobs = _BOOT_CTX["Dobs"]
    Dt = _BOOT_CTX["Dt"]
    setup = _BOOT_CTX["setup"]
    result = _BOOT_CTX["result"]
    invtimes = _BOOT_CTX["times"]

    ChannelsUsed = np.asarray(setup["ChannelsUsed"], dtype=np.int64)

    plotdir = os.path.join(config.outputdir, 'plots')
    os.makedirs(plotdir, exist_ok=True)

    # =========================================================
    # 1. Station-weight realizations (Dirichlet), as in TimeTest2
    # =========================================================
    # at least one weight set per iteration (CPU and GPU both use the first
    # num_iterations rows)
    # (with RandomSeed in the configuration the weights are the same on every run of this event)
    _rng = config.rng('bootstrap', config.org.time)
    _dirichlet = _rng.dirichlet if _rng is not None else np.random.dirichlet
    dirichlet_samples = _dirichlet(alpha=[1] * len(ChannelsUsed),
                                   size=max(num_realizations, num_iterations))
    # station-level weights: the same weight on every used component (trace)
    # of a station (ISOLA-BaBoo, Zahradnik et al. 2026, sec. 2.4); removed
    # components are already absent from Dobs/G. One weight per trace, in the
    # trace order of G (station by station, its used components consecutive).
    Wc = np.ascontiguousarray(np.repeat(dirichlet_samples,
                                        np.asarray(ChannelsUsed, dtype=np.int64), axis=1))
    config.logger.debug("Bootstrap: trace weights %s, time elapsed: %.2f s",
                        Wc.shape, time.perf_counter() - startBoot)

    # =========================================================
    # 2. Grid and CMT coordinates from the inversion context
    #    (source files were already read in calculateInversions2())
    # =========================================================
    grid = _BOOT_CTX["grid"][:len(vr)]
    Xcmt = float(result["Xcmt"])
    Ycmt = float(result["Ycmt"])
    Zcmt = float(result["Zcmt"])
    config.logger.info(
        'Bootstrap CMT (from inversion): X={} Y={} Z={}'.format(Xcmt, Ycmt, Zcmt))
    gid, gx, gy, gz = grid[:, 0], grid[:, 1], grid[:, 2], grid[:, 3]

    _boot_cfg = config.cfg.get('Bootstrap', {})
    _keep_frac       = float(_boot_cfg.get('KeepFraction',        0.1))

    # ── centroid-box subset ──────────────────────────────────────────────────
    xmin, xmax = gx.min(), gx.max()
    ymin, ymax = gy.min(), gy.max()
    zmin, zmax = gz.min(), gz.max()

    # the CentroidBox rules are selected by the catalog magnitude of the event, like
    # every other magnitude rule (Window, Frequency, TimeShift, ...)
    try:
        _rule_mag = float(event.getMagnitude(config.evt, config.org).mag)
    except Exception:
        _rule_mag = float(result.get('BestMw', 4.0))
        config.logger.warning('Bootstrap: the catalog magnitude is not available; the CentroidBox '
                              'rule is selected by the Mw of the best solution')
    _mag_rounded = round(_rule_mag, 1)
    _valid = _BOOT_CTX.get("valid")
    _ok = np.ones(len(gx), dtype=bool) if _valid is None else _valid[:len(gx)].copy()

    # ── choice of the centroid box ───────────────────────────────────────────
    # 1. CentroidBox rules [Mmin, Mmax, Dx, Dy, Dz] (if set and one matches the
    #    catalog magnitude): box of +/-Dx, +/-Dy, +/-Dz km around the best centroid
    # 2. otherwise CentroidBoxVR (fraction of VRmax, e.g. 0.8): bounding box of
    #    all grid points whose VR (best over time shifts) >= fraction x VRmax
    # 3. otherwise: the full grid
    _box_rules = _boot_cfg.get('CentroidBox') or []
    _vr_frac   = _boot_cfg.get('CentroidBoxVR')
    Dx = Dy = Dz = None
    _box_info = {'rule': 'full grid'}
    for _rule in _box_rules:
        if round(float(_rule[0]), 1) <= _mag_rounded <= round(float(_rule[1]), 1):
            Dx, Dy, Dz = float(_rule[2]), float(_rule[3]), float(_rule[4])
            _box_info = {'rule': 'CentroidBox', 'mw_rule': [float(_rule[0]), float(_rule[1])],
                         'Dx': Dx, 'Dy': Dy, 'Dz': Dz}
            break
    if Dx is None and _box_rules:
        config.logger.warning('Bootstrap: no CentroidBox rule matches magnitude %.1f', _mag_rounded)

    if Dx is not None:
        xminb = Xcmt - Dx;  xmaxb = Xcmt + Dx
        if xminb < xmin:
            xminb = xmin;   xmaxb = min(xmin + 2*Dx, xmax)
        elif xmaxb > xmax:
            xmaxb = xmax;   xminb = max(xmax - 2*Dx, xmin)

        yminb = Ycmt - Dy;  ymaxb = Ycmt + Dy
        if yminb < ymin:
            yminb = ymin;   ymaxb = min(ymin + 2*Dy, ymax)
        elif ymaxb > ymax:
            ymaxb = ymax;   yminb = max(ymax - 2*Dy, ymin)

        zminb = Zcmt - Dz;  zmaxb = Zcmt + Dz
        if zminb < zmin:
            zminb = zmin;   zmaxb = min(zmin + 2*Dz, zmax)
        elif zmaxb > zmax:
            zmaxb = zmax;   zminb = max(zmax - 2*Dz, zmin)
    else:
        xminb, xmaxb, yminb, ymaxb, zminb, zmaxb = xmin, xmax, ymin, ymax, zmin, zmax
        if _vr_frac is not None:
            try:
                _vr_frac = float(_vr_frac)
            except (TypeError, ValueError):
                _vr_frac = float('nan')
            _vr_ok = np.asarray(vr[:len(gx)], dtype=float)
            _vrmax = float(np.max(_vr_ok[_ok])) if _ok.any() else float('nan')
            if not (0.0 < _vr_frac <= 1.0):
                config.logger.warning('Bootstrap: CentroidBoxVR=%r is invalid (must be 0 < value <= 1) '
                                      '- using the full grid', _boot_cfg.get('CentroidBoxVR'))
            elif not _vrmax > 0.0:
                config.logger.warning('Bootstrap: VRmax=%.2f%% <= 0, CentroidBoxVR cannot be used '
                                      '- using the full grid', _vrmax)
            else:
                _thr = _vr_frac * _vrmax
                _qual = _ok & (_vr_ok >= _thr)
                xminb, xmaxb = float(gx[_qual].min()), float(gx[_qual].max())
                yminb, ymaxb = float(gy[_qual].min()), float(gy[_qual].max())
                zminb, zmaxb = float(gz[_qual].min()), float(gz[_qual].max())
                _box_info = {'rule': 'CentroidBoxVR', 'fraction': _vr_frac, 'vrmax': _vrmax,
                             'threshold': _thr, 'n_qualifying': int(_qual.sum())}

    inbox = ((gx >= xminb) & (gx <= xmaxb) &
             (gy >= yminb) & (gy <= ymaxb) &
             (gz >= zminb) & (gz <= zmaxb))
    # rejected sources (placeholder shift/VR) must not enter the bootstrap
    inbox &= _ok

    if not inbox.any():
        config.logger.warning('Bootstrap: centroid box contains no grid points — using full grid')
        inbox = _ok.copy()
        xminb, xmaxb, yminb, ymaxb, zminb, zmaxb = xmin, xmax, ymin, ymax, zmin, zmax
        _box_info = {'rule': 'full grid'}
    if int(inbox.sum()) < 5:
        config.logger.warning('Bootstrap: only %d grid point(s) in the centroid box - the ensemble '
                              'varies almost only in time', int(inbox.sum()))

    # the box restricts space only; every perturbation searches the full
    # time-shift range of the inversion (not the range of the unperturbed
    # best shifts, which would truncate the centroid-time uncertainty)
    _tshifts_full = np.asarray(setup["Timeshifts"], dtype=np.int64)
    tminb, tmaxb = int(_tshifts_full.min()), int(_tshifts_full.max())

    _box_info.update({'box': [float(xminb), float(xmaxb), float(yminb), float(ymaxb),
                                   float(zminb), float(zmaxb)],
                          't': [tminb * Dt, tmaxb * Dt],
                          'n_inbox': int(inbox.sum()), 'n_total': int(len(gx))})
    _BOOT_CTX['centroid_box'] = _box_info
    for _line in _centroid_box_lines(_box_info):
        print('[BOOT] ' + _line, flush=True)
        config.logger.info('Bootstrap centroid box: ' + _line)

    config.logger.debug("Grid boundaries: X [%s, %s] Y [%s, %s] Z [%s, %s]",
                        xmin, xmax, ymin, ymax, zmin, zmax)
    config.logger.debug("CMT coordinates: X: %s, Y: %s, Z: %s", Xcmt, Ycmt, Zcmt)
    config.logger.debug("Bootstrap box: X [%s, %s] Y [%s, %s] Z [%s, %s] T [%s, %s]",
                        xminb, xmaxb, yminb, ymaxb, zminb, zmaxb, tminb, tmaxb)

    idx = gid[inbox].astype(np.int64)
    Coords = np.ascontiguousarray(grid[inbox][:, 1:4])

    idx0 = idx - 1
    valid = (idx0 >= 0) & (idx0 < G.shape[0])
    valid_idx = idx0[valid]
    Coords = Coords[valid]

    Gboot = np.ascontiguousarray(G[valid_idx])
    TimeshiftsBoot = np.ascontiguousarray(_tshifts_full)
    config.logger.debug("Bootstrap GF selection: %.2f s -> %s grid points, %s shifts",
                        time.perf_counter() - startBoot, Gboot.shape[0], len(TimeshiftsBoot))

    # =========================================================
    # 2b. Whole grid coloured by VR + bootstrap subset figure
    # =========================================================
    try:
        bsel = np.zeros(len(gx), dtype=bool)
        bsel[np.where(inbox)[0][valid]] = True

        n_total  = len(gx)
        n_boot   = int(bsel.sum())
        pct_boot = 100.0 * n_boot / n_total if n_total else 0.0

        order = np.argsort(vr)
        norm  = mcolors.Normalize(vmin=vr.min(), vmax=vr.max())
        cmap  = plt.get_cmap('viridis')

        fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
        panels = [
            (gx, gy, 'X (km)', 'Y (km)',       (Xcmt, Ycmt), False),
            (gx, gz, 'X (km)', 'Depth Z (km)', (Xcmt, Zcmt), True),
            (gy, gz, 'Y (km)', 'Depth Z (km)', (Ycmt, Zcmt), True),
        ]
        sc = None
        for ax, (u, v, xl, yl, cmt_xy, flip) in zip(axes, panels):
            sc = ax.scatter(u[order], v[order], c=vr[order], cmap=cmap,
                            norm=norm, s=18, alpha=0.9, edgecolors='none',
                            label='Grid points (VR colour)')
            ax.scatter(u[bsel], v[bsel], facecolors='none', edgecolors='red',
                       s=46, linewidths=0.9,
                       label='Bootstrap subset ({} pts, {:.1f}%)'.format(n_boot, pct_boot))
            ax.scatter([cmt_xy[0]], [cmt_xy[1]], marker='*', s=280, color='lime',
                       edgecolors='black', linewidths=1.0, zorder=5,
                       label='Best solution (CMT)')
            ax.set_xlabel(xl, fontsize=12)
            ax.set_ylabel(yl, fontsize=12)
            if flip:
                ax.invert_yaxis()
        cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label('Variance Reduction (%)')

        fig.suptitle('Source grid coloured by Variance Reduction',
                     fontsize=18, fontweight='bold', y=1.12)
        fig.text(0.5, 1.015,
                 'Total: {n} pts  |  Bootstrap subset: {m} pts ({p:.1f}%)  |  {r}'.format(
                     n=n_total, m=n_boot, p=pct_boot,
                     r=_centroid_box_lines(_box_info)[0].replace('Rule: ', '')),
                 ha='center', va='bottom', fontsize=13)
        # legend below the three panels, so it never covers grid points
        _h, _l = axes[0].get_legend_handles_labels()
        fig.legend(_h, _l, loc='lower center', ncol=3, fontsize=10, framealpha=0.9,
                   markerscale=0.9, bbox_to_anchor=(0.5, -0.04))

        fig.savefig(os.path.join(plotdir, _rv('grid_vr_bootstrap.png')), dpi=150,
                    bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)
        config.logger.debug("Grid/VR overview figure saved: %s",
                            os.path.join(plotdir, _rv('grid_vr_bootstrap.png')))
    except Exception:
        import traceback
        traceback.print_exc()
        config.logger.warning("Grid/VR figure failed (bootstrap continues)")

    # =========================================================
    # 3. Bootstrap iterations
    # =========================================================
    BootstrapResults = []
    BootstrapAngles = []
    startIters = time.perf_counter()
    # NBEST = KeepFraction of all space-time (source, shift) pairs
    _n_keep = max(1, int(_keep_frac * Gboot.shape[0] * len(TimeshiftsBoot)))
    # the station weights are constant within each trace: the weight-free per-trace
    # sums are computed once (CPU) and combined with the weights of all perturbations
    # (CPU or GPU, the same routine)
    if Wc.shape[1] != Gboot.shape[1] // 1024:
        raise ValueError('bootstrap: {} trace weights for {} traces'.format(
            Wc.shape[1], Gboot.shape[1] // 1024))
    BootstrapResults = _bootstrap_members(Gboot, Dobs, Wc[:num_iterations], Dt, TimeshiftsBoot,
                                          Coords, _n_keep, use_gpu=_USE_GPU and _CUDA_AVAILABLE)
    # decomposition of all members in one call: silsub is a parallel Numba
    # function with ~20 ms start-up per call, so 100 calls (one per perturbation)
    # cost ~2 s, one call ~0.05 s. Every later use concatenates the members of
    # all perturbations, so a one-element list gives the same arrays.
    BootstrapAngles = [silsub(np.ascontiguousarray(
        np.vstack([res[2] for res in BootstrapResults]), dtype=np.float64))]
    itersTime = time.perf_counter() - startIters
    config.logger.debug("Bootstrap iterations done: %s x %s grid points = %.2f s (%s ms/iter)",
                        num_iterations, Gboot.shape[0], itersTime,
                        round(1000.0 * itersTime / num_iterations))

    # =========================================================
    # 5. Aggregate the ensemble: all NBEST members of all perturbations.
    #    No VR cut afterwards - NBEST (KeepFraction) is the only acceptance
    #    rule, as in ISOLA-BaBoo (Zahradnik et al. 2026).
    # =========================================================
    mt_array = np.vstack([it[2] for it in BootstrapResults])
    all_shifts = np.concatenate([r[0] for r in BootstrapResults])
    all_coords = np.vstack([r[4] for r in BootstrapResults])

    M0_all = np.hstack([r[2] for r in BootstrapAngles])
    Mw_all = np.hstack([r[3] for r in BootstrapAngles])
    DC_all = np.hstack([r[4] for r in BootstrapAngles])
    CLVD_all = np.hstack([r[5] for r in BootstrapAngles])
    ISO_all = np.hstack([r[6] for r in BootstrapAngles])
    VR_all = np.concatenate([r[3] for r in BootstrapResults])

    config.logger.debug("Bootstrap fit boundaries, Max: %s, Min: %s",
                        VR_all.max(), VR_all.min())

    SolutionParts = np.column_stack((DC_all, CLVD_all, ISO_all))
    SolutionPartsMean = np.mean(SolutionParts, axis=0)
    Magnitudes = np.column_stack((M0_all, Mw_all))
    MagnitudesMean = np.mean(Magnitudes, axis=0)

    mt_mean = np.mean(mt_array, axis=0)
    mt_std = np.std(mt_array, axis=0)

    # ---- median moment tensor (ISOLA-BaBoo, Zahradnik et al. 2026) ----
    # component-wise median of the members' six MT components (Mxx..Myz, NED),
    # taken on the tensors themselves, not on the ISOLA a-coefficients
    M_all = np.concatenate([r[10] for r in BootstrapAngles])     # (n, 3, 3)
    M_median = np.median(M_all, axis=0)
    if _BOOT_CTX.get("components", mt_array.shape[1]) == 5:
        # deviatoric inversion: every member has zero trace, but the
        # component-wise median need not - remove the spurious isotropic part
        M_median = M_median - (np.trace(M_median) / 3.0) * np.eye(3)
    # a-coefficients of the median MT (input format of silsub / plots)
    mt_median = moment_tensor_to_a(M_median)
    # per-component statistics of the members' tensors
    _comp_idx = (('Mxx', 0, 0), ('Myy', 1, 1), ('Mzz', 2, 2),
                 ('Mxy', 0, 1), ('Mxz', 0, 2), ('Myz', 1, 2))
    M_comp_stats = OrderedDict((name, _ci_stats(M_all[:, i, j]))
                               for name, i, j in _comp_idx)

    # nodal planes, M0 and Mw of the median moment tensor, not statistics of
    # per-member strike/dip/rake
    _med_angl = silsub(np.ascontiguousarray(mt_median[None, :], dtype=np.float64))
    plane1_median = _med_angl[0][0]
    plane2_median = _med_angl[1][0]
    # M0 / Mw are reported for the median MT (paper, Table 1)
    M0_medianMT = float(_med_angl[2][0])
    Mw_medianMT = float(_med_angl[3][0])

    # ensemble statistics as in the paper: median, 68% and 95% intervals
    ens_stats = OrderedDict(
        (name, _ci_stats(arr)) for name, arr in (
            ('M0', M0_all), ('Mw', Mw_all),
            ('DC', DC_all), ('CLVD', CLVD_all), ('ISO', ISO_all),
            ('X', all_coords[:, 0]), ('Y', all_coords[:, 1]),
            ('Z', all_coords[:, 2]), ('T', all_shifts * Dt)))

    BestParts = np.array([result["BestDC"], result["BestCLVD"], result["BestISO"]])
    BestMagnitudes = np.array([result["BestMo"], result["BestMw"]])
    # centroid time in seconds relative to origin time (shift in samples * Dt)
    CMT_BEST = [result["BestShift"] * Dt, Xcmt, Ycmt, Zcmt]

    # =========================================================
    # 6. Store figure args for deferred generation in exportResults()
    # =========================================================
    global _BOOT_PLOT_CTX
    _BOOT_PLOT_CTX = dict(
        plotdir=plotdir, Dobs=Dobs, result=result, VR_all=VR_all,
        mt_mean=mt_mean, mt_median=mt_median,
        SolutionParts=SolutionParts, BestParts=BestParts, Magnitudes=Magnitudes,
        MagnitudesMedianMT=np.array([M0_medianMT, Mw_medianMT]),
        BestMagnitudes=BestMagnitudes, all_times=all_shifts * Dt, all_coords=all_coords,
        CMT_BEST=CMT_BEST, BootstrapAngles=BootstrapAngles,
        tlim=(tminb * Dt, tmaxb * Dt), xlim=(xminb, xmaxb), ylim=(yminb, ymaxb), zlim=(zminb, zmaxb)
    )
    plotsTime = 0.0

    # ---- Store bootstrap stats for renderSite() to embed in final HTML ----
    try:
        boot_stats = {
            'n_solutions': int(len(VR_all)),
            'n_iterations': int(num_iterations),
            'vr_max': float(VR_all.max()),
            'vr_min': float(VR_all.min()),
            'p1_med': plane1_median,
            'p2_med': plane2_median,
            'Mo_mt': M0_medianMT,
            'Mw_mt': Mw_medianMT,
            'full_mt': bool(np.std(ISO_all) > 1e-6 or abs(result.get('BestISO', 0.0)) > 1e-8),
        }
        # (median, p16, p84, p2.5, p97.5) per quantity
        for _key, _name in (('x', 'X'), ('y', 'Y'), ('z', 'Z'), ('time', 'T'),
                            ('Mo', 'M0'), ('Mw', 'Mw'), ('DC', 'DC'),
                            ('CLVD', 'CLVD'), ('ISO', 'ISO')):
            boot_stats[_key] = ens_stats[_name]
        global _BOOT_STATS
        _BOOT_STATS = (plotdir, boot_stats, result)
        config.logger.info('Bootstrap stats stored; renderSite() will embed them in index.html')
    except Exception:
        import traceback
        traceback.print_exc()
        print("WARNING: bootstrap stats storage failed (analysis unaffected)")

    # =========================================================
    # 7. Store final summary for deferred printing in gisolaBootstrap.py
    # =========================================================
    endBoot = time.perf_counter()
    computeTime = (endBoot - startBoot) - plotsTime
    global _SUMMARY
    _SUMMARY = {
        'invtimes':           invtimes,
        'num_iterations':     num_iterations,
        'n_grid':             Gboot.shape[0],
        'itersTime':          itersTime,
        'computeTime':        computeTime,
        'result':             result,
        'best_idx':           best_idx,
        'plane1_median':      plane1_median,   # planes of the median MT
        'plane2_median':      plane2_median,
        'M0_medianMT':        M0_medianMT,
        'Mw_medianMT':        Mw_medianMT,
        'M_median':           M_median,        # 3x3 NED median moment tensor
        'M_comp_stats':       M_comp_stats,    # per-component median/68%/95%
        'ens_stats':          ens_stats,       # median, p16, p84, p2.5, p97.5
        'vr_min':             float(VR_all.min()),
        'vr_max':             float(VR_all.max()),
        'n_members':          int(len(VR_all)),
        'MagnitudesMedian':   np.median(Magnitudes, axis=0),
        'MagnitudesStd':      np.std(Magnitudes, axis=0),
        'MagnitudesP16':      np.percentile(Magnitudes, 15.87, axis=0),
        'MagnitudesP84':      np.percentile(Magnitudes, 84.13, axis=0),
        'MagnitudesMean':     MagnitudesMean,
        'SolutionPartsMedian': np.median(SolutionParts, axis=0),
        'SolutionPartsStd':   np.std(SolutionParts, axis=0),
        'SolutionPartsP16':   np.percentile(SolutionParts, 15.87, axis=0),
        'SolutionPartsP84':   np.percentile(SolutionParts, 84.13, axis=0),
        'SolutionPartsMean':  SolutionPartsMean,
        'mt_mean':            mt_mean,
        'mt_median':          mt_median,
        'mt_std':             mt_std,
        'plotsTime':          plotsTime,
        'plotdir':            plotdir,
        'xyt_stats': {
            'X (km)': (float(np.median(all_coords[:, 0])),
                       float(np.percentile(all_coords[:, 0], 15.87)),
                       float(np.percentile(all_coords[:, 0], 84.13))),
            'Y (km)': (float(np.median(all_coords[:, 1])),
                       float(np.percentile(all_coords[:, 1], 15.87)),
                       float(np.percentile(all_coords[:, 1], 84.13))),
            'Z (km)': (float(np.median(all_coords[:, 2])),
                       float(np.percentile(all_coords[:, 2], 15.87)),
                       float(np.percentile(all_coords[:, 2], 84.13))),
            'T (s)':  (float(np.median(all_shifts * Dt)),
                       float(np.percentile(all_shifts * Dt, 15.87)),
                       float(np.percentile(all_shifts * Dt, 84.13))),
        },
        'box_Dx':      Dx,
        'box_Dy':      Dy,
        'box_Dz':      Dz,
        'box_xminb':   float(xminb),
        'box_xmaxb':   float(xmaxb),
        'box_yminb':   float(yminb),
        'box_ymaxb':   float(ymaxb),
        'box_zminb':   float(zminb),
        'box_zmaxb':   float(zmaxb),
        'box_n_inbox': int(inbox.sum()),
        'box_n_total': int(len(gx)),
        'box_pct':     float(inbox.sum()) / len(gx) * 100.0,
        'boot_tminb':     int(tminb),
        'boot_tmaxb':     int(tmaxb),
        'boot_n_shifts':  int(len(TimeshiftsBoot)),
        'boot_Dt':        float(Dt),
    }


    # write bootstrap_results.dat
    try:
        _s    = _SUMMARY
        _pm1, _pm2 = _s['plane1_median'], _s['plane2_median']
        _es = _s['ens_stats']
        _bl = []
        def _bw(line=''):
            _bl.append(line)
        def _bstat(label, key, fmt, unit=''):
            med, p16, p84, p025, p975 = (_fmt_nz(fmt, v) for v in _es[key])
            _bw('{:<16s}: {}   68%: [{} – {}]   95%: [{} – {}]{}'.format(
                label, med, p16, p84, p025, p975, unit))
        _bw()
        _bw('================ BOOTSTRAP RESULTS ================')
        _bw()
        _bw('Iterations: {}'.format(_s['num_iterations']))
        _bw('Ensemble members: {}'.format(_s['n_members']))
        _bw('Computation time: {:.1f} s ({} perturbations: {:.1f} s, {:.2f} s each)'.format(
            _s['computeTime'], _s['num_iterations'], _s['itersTime'],
            _s['itersTime'] / max(1, _s['num_iterations'])))
        _bw('VR range of the ensemble: {:.1f} – {:.1f} %'.format(_s['vr_min'], _s['vr_max']))
        _bw()
        _Mm = _s['M_median']
        _bw('Median moment tensor (component-wise median of the ensemble, NED, N·m):')
        _bw('Mxx = {:.3e}  Myy = {:.3e}  Mzz = {:.3e}'.format(_Mm[0, 0], _Mm[1, 1], _Mm[2, 2]))
        _bw('Mxy = {:.3e}  Mxz = {:.3e}  Myz = {:.3e}'.format(_Mm[0, 1], _Mm[0, 2], _Mm[1, 2]))
        _bw('M0 = {:.3e} N·m,  Mw = {:.2f}'.format(_s['M0_medianMT'], _s['Mw_medianMT']))
        _bw('Plane 1: Strike = {:.0f}, Dip = {:.0f}, Rake = {:.0f}'.format(*_pm1))
        _bw('Plane 2: Strike = {:.0f}, Dip = {:.0f}, Rake = {:.0f}'.format(*_pm2))
        _bw()
        _bw('Ensemble statistics — median, 68% CI [16th – 84th percentile],')
        _bw('                      95% CI [2.5th – 97.5th percentile]:')
        _bstat('M0 (N·m)', 'M0', '{:.3e}')
        _bstat('Mw', 'Mw', '{:.2f}')
        _bstat('DC (%)', 'DC', '{:.0f}')
        _bstat('CLVD (%)', 'CLVD', '{:.0f}')
        _bstat('ISO (%)', 'ISO', '{:.0f}')
        _bw()
        _bw('Moment tensor components (NED, N·m):')
        for _cname, (_med, _p16, _p84, _p025, _p975) in _s['M_comp_stats'].items():
            _bw('{:<16s}: {:.3e}   68%: [{:.3e} – {:.3e}]   95%: [{:.3e} – {:.3e}]'.format(
                _cname, _med, _p16, _p84, _p025, _p975))
        _bw()
        _bw('Centroid position and time:')
        _bstat('X (km)', 'X', '{:.2f}')
        _bstat('Y (km)', 'Y', '{:.2f}')
        _bstat('Z (km)', 'Z', '{:.2f}')
        _bstat('T (s)', 'T', '{:.2f}')
        _bw()
        _bw('==================================================')
        _bp = os.path.join(config.outputdir, _rv('bootstrap_results.dat'))
        with open(_bp, 'w') as _f:
            _f.write('\n'.join(_bl) + '\n')
        config.logger.info('Bootstrap results saved to %s', _bp)
    except Exception as _e:
        config.logger.warning('Could not save bootstrap_results.dat: %s', _e)


def _ci_lines(ax, data, fmt='%.2f'):
    """Median (red), 68% CI (blue) and 95% CI (green) as vertical lines,
    the convention of the ISOLA-BaBoo histograms (Zahradnik et al. 2026)."""
    med, p16, p84, p025, p975 = _ci_stats(data)
    ax.axvline(med, color='red', linestyle='-', linewidth=2.0,
               label=('Median ' + fmt) % med)
    ax.axvline(p16, color='blue', linestyle='--', linewidth=1.5,
               label=('68%% CI [' + fmt + ', ' + fmt + ']') % (p16, p84))
    ax.axvline(p84, color='blue', linestyle='--', linewidth=1.5)
    ax.axvline(p025, color='green', linestyle=':', linewidth=1.8,
               label=('95%% CI [' + fmt + ', ' + fmt + ']') % (p025, p975))
    ax.axvline(p975, color='green', linestyle=':', linewidth=1.8)


# Plot data of the bootstrap figures. Set in the parent process right before
# the figure tasks run (exportResults / _make_bootstrap_figures); the forked
# worker processes inherit it, so nothing large is pickled.
_FIG_CTX = None


def _hist_row(plotdir, data2d, bests, titles, suptitle, fname,
              xranges=None, col_indices=None, fmts=None, mt_values=None,
              panel_size=(6, 4)):
    """Histograms with median + 68% + 95% CI lines and the best solution.
    col_indices: which columns of data2d/bests to use (default: all).
    xranges: per-panel (min, max) or None. mt_values: optional value of
    the median moment tensor per column, drawn as an extra line."""
    if col_indices is None:
        col_indices = list(range(len(titles)))
    n = len(col_indices)
    fig, axes = plt.subplots(1, n, figsize=(panel_size[0] * n, panel_size[1]))
    if n == 1:
        axes = [axes]
    for plot_i, data_i in enumerate(col_indices):
        ax = axes[plot_i]
        xr = xranges[plot_i] if xranges else None
        kw = {'range': xr} if xr else {}
        ax.hist(data2d[:, data_i], bins=25, color='#5b9bd5',
                edgecolor='white', linewidth=0.5, alpha=0.85, **kw)
        _ci_lines(ax, data2d[:, data_i], fmts[plot_i] if fmts else '%.2f')
        if mt_values is not None:
            ax.axvline(mt_values[data_i], color='purple', linestyle='-.',
                       linewidth=1.8, label='Median MT')
        ax.axvline(bests[data_i], color='black', linestyle='--',
                   linewidth=1.8, label='Best')
        ax.set_title(titles[plot_i], fontsize=13, fontweight='bold')
        ax.set_xlabel('Value', fontsize=10)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=9, framealpha=0.85)
    axes[0].set_ylabel('Frequency', fontsize=10)
    fig.suptitle(suptitle, fontsize=17, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(plotdir, _rv(fname)), dpi=150)
    plt.close(fig)


def _fig_is_full_mt(c):
    """True for full-MT ensembles (ISO present), False for deviatoric."""
    return (abs(c['result'].get('BestISO', 0.0)) > 1e-8
            or np.std(c['SolutionParts'][:, 2]) > 1e-6)


def _fig_fit_waveforms():
    c = _FIG_CTX
    fig = plt.figure(figsize=(12, 4))
    plt.plot(c['Dobs'])
    plt.plot(c['result']["Dsynth"])
    plt.legend(['Dobs', 'Dsynth'])
    plt.title('Waveform fit (best solution)')
    fig.savefig(os.path.join(c['plotdir'], _rv('fit_waveforms.png')), dpi=150)
    plt.close(fig)


def _fig_decomposition():
    c = _FIG_CTX
    sp = c['SolutionParts']
    # DC% is 0..100; CLVD% and ISO% are signed (-100..100); ISO only for full MT
    if _fig_is_full_mt(c):
        titles, cols = ['DC%', 'CLVD%', 'ISO%'], [0, 1, 2]
        ranges = [(0, 100), (-100, 100), (-100, 100)]
    else:
        titles, cols = ['DC%', 'CLVD%'], [0, 1]
        ranges = [(0, 100), (-100, 100)]
    _hist_row(c['plotdir'], sp, c['BestParts'], titles,
              'Moment Tensor Decomposition (N = %d)' % len(sp),
              'hist_decomposition.png', xranges=ranges,
              col_indices=cols, fmts=['%.0f'] * len(cols))


def _fig_magnitudes():
    c = _FIG_CTX
    mags = c['Magnitudes']
    _hist_row(c['plotdir'], mags, c['BestMagnitudes'], ['Mo', 'Mw'],
              'Magnitude Distributions (N = %d)' % len(mags), 'hist_magnitudes.png',
              fmts=['%.2e', '%.2f'], mt_values=c['MagnitudesMedianMT'],
              panel_size=(7.5, 5.5))


def _discrete_bins(data, max_bins=60):
    """Histogram edges for values on a regular grid (centroid positions,
    time shifts): one bin per grid value, centred on it. Falls back to
    max_bins equal bins for irregular or very dense values."""
    u = np.unique(np.round(np.asarray(data, dtype=float), 6))
    if len(u) == 1:
        return np.array([u[0] - 0.5, u[0] + 0.5])
    step = np.min(np.diff(u))
    n = int(round((u[-1] - u[0]) / step)) + 1
    if n <= max_bins:
        return u[0] - step / 2.0 + step * np.arange(n + 1)
    return np.linspace(u[0], u[-1], max_bins + 1)


def _fig_centroid_time():
    c = _FIG_CTX
    all_coords, CMT_BEST = c['all_coords'], c['CMT_BEST']
    fig, _axs = plt.subplots(2, 2, figsize=(15, 10))
    axes = _axs.ravel()
    titles = ['Centroid Time (s, rel. origin)', 'X Coordinate (km)', 'Y Coordinate (km)', 'Depth Z (km)']
    # all_times, CMT_BEST[0] and tlim are in seconds (shift * Dt)
    data_list = [c['all_times'], all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]]
    colors_ = ['#e05c5c', '#4a9e6b', '#d4813a', '#7e6abf']
    custom_limits = [c['tlim'], c['xlim'], c['ylim'], c['zlim']]
    for i in range(4):
        data = np.asarray(data_list[i], dtype=float)
        low_lim, high_lim = custom_limits[i]
        ax = axes[i]
        # histogram of the whole ensemble; positions and times take discrete
        # grid values, so bins are centred on the grid spacing / time step
        edges = _discrete_bins(data)
        counts, _, _ = ax.hist(data, bins=edges, color=colors_[i], alpha=0.55,
                               edgecolor='white', linewidth=0.5,
                               label='Ensemble (N = %d)' % len(data))
        low_lim, high_lim = min(low_lim, edges[0]), max(high_lim, edges[-1])
        _ci_lines(ax, data)
        ax.axvline(CMT_BEST[i], color='black', linestyle='--', linewidth=1.8,
                   label='Best  %.2f' % CMT_BEST[i])
        ax.set_xlim(low_lim, high_lim)
        ax.set_ylim(0, 1.45 * counts.max())      # headroom: legend above the bars
        ax.set_title(titles[i], fontsize=13, fontweight='bold')
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=9.5, loc='upper right', framealpha=0.85)
    fig.suptitle('Bootstrap Time and Centroid Uncertainty (histograms of the whole ensemble)',
                 fontsize=17, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(c['plotdir'], _rv('hist_centroid_time.png')), dpi=150)
    plt.close(fig)


def _fig_iso():
    c = _FIG_CTX
    if not _fig_is_full_mt(c):          # ISO histogram only for full MT
        _remove_stale_figure(c, 'hist_iso.png')
        return
    iso = c['SolutionParts'][:, 2]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(iso, bins=25, color='#5b9bd5', edgecolor='white', linewidth=0.5, alpha=0.85)
    _ci_lines(ax, iso, '%.0f')
    ax.set_xlabel('ISO (%)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(fontsize=10, framealpha=0.85)
    ax.set_title('Moment Tensor Decomposition: ISO Component', fontsize=16, fontweight='bold')
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(c['plotdir'], _rv('hist_iso.png')), dpi=150)
    plt.close(fig)


def _fig_tradeoff():
    # trade-off figure of the paper (Fig. 1a and 1c) for CLVD%, always
    _tradeoff_plot(_FIG_CTX, 1, 'CLVD (%)', 'tradeoff_depth_vr.png')


def _remove_stale_figure(c, fname):
    """A figure made only for full moment tensors: remove the file of an earlier
    run (e.g. a previous full-MT revision), so the page of a deviatoric
    solution does not show it."""
    path = os.path.join(c['plotdir'], _rv(fname))
    if os.path.isfile(path):
        os.remove(path)


def _fig_tradeoff_iso():
    # the same figure for ISO%, only for a full moment tensor (deviatoric: ISO = 0)
    if _fig_is_full_mt(_FIG_CTX):
        _tradeoff_plot(_FIG_CTX, 2, 'ISO (%)', 'tradeoff_depth_vr_iso.png')
    else:
        _remove_stale_figure(_FIG_CTX, 'tradeoff_depth_vr_iso.png')


def _tradeoff_plot(c, col, nd_label, fname):
    # (a) VR vs the non-DC component coloured by depth, (b) depth vs the
    # component coloured by VR - Zahradnik et al. (2026), Fig. 1a and 1c
    nd = c['SolutionParts'][:, col]
    depth = np.asarray(c['all_coords'][:, 2], dtype=float)
    vr = np.asarray(c['VR_all'], dtype=float)
    order = np.argsort(vr)            # best-fitting members drawn on top
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    sc1 = ax1.scatter(nd[order], vr[order], c=depth[order], cmap='viridis_r',
                      s=12, alpha=0.7, edgecolors='none')
    ax1.set_xlabel(nd_label, fontsize=11)
    ax1.set_ylabel('Variance Reduction (%)', fontsize=11)
    ax1.set_title('VR vs %s, colour = depth' % nd_label.split()[0], fontsize=14, fontweight='bold')
    fig.colorbar(sc1, ax=ax1).set_label('Depth Z (km)')
    sc2 = ax2.scatter(nd[order], depth[order], c=vr[order], cmap='viridis',
                      s=12, alpha=0.7, edgecolors='none')
    ax2.invert_yaxis()
    ax2.set_xlabel(nd_label, fontsize=11)
    ax2.set_ylabel('Depth Z (km)', fontsize=11)
    ax2.set_title('Depth vs %s, colour = VR' % nd_label.split()[0], fontsize=14, fontweight='bold')
    fig.colorbar(sc2, ax=ax2).set_label('Variance Reduction (%)')
    for ax in (ax1, ax2):
        ax.grid(linestyle='--', linewidth=0.5, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle('Bootstrap ensemble: source type vs fit and depth (N = %d)' % len(vr),
                 fontsize=17, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(c['plotdir'], _rv(fname)), dpi=150)
    plt.close(fig)


def _fig_all_mats(c):
    """(n, 3, 3) stack of every ensemble member's moment tensor."""
    return np.concatenate([res[-1] for res in c['BootstrapAngles']])


def _fig_hudson():
    # Hudson plot of ALL ensemble members (vectorised projection)
    if not HAVE_PYROCKO:
        print("pyrocko not installed - skipping Hudson figure")
        return
    c = _FIG_CTX
    u_vals, v_vals = _hudson_uv(_fig_all_mats(c))
    mt_median = c['mt_median']
    mt_median6 = np.append(mt_median, 0) if len(mt_median) == 5 else mt_median
    median_mt = pmt.MomentTensor(m=build_moment_tensor(np.asarray(mt_median6, dtype=np.float64)))
    _hudson_cloud_plot(u_vals, v_vals, np.asarray(c['VR_all'], dtype=float), median_mt,
                       'Median solution', os.path.join(c['plotdir'], _rv('hudson_median.png')))


def _fig_fuzzy():
    # fuzzy (uncertainty) beachball: deviatoric part of ALL ensemble members
    if not HAVE_PYROCKO:
        print("pyrocko not installed - skipping fuzzy beachball")
        return
    c = _FIG_CTX
    all_mats = _fig_all_mats(c)
    best_mt = pmt.MomentTensor(m=c['result']["Tensor"])
    best_norm = pmt.MomentTensor(m=best_mt.m() / best_mt.scalar_moment()).deviatoric()
    # median moment tensor (same tensor as in the .dat / Hudson / lune plots)
    mt_median = c['mt_median']
    mt_median6 = np.append(mt_median, 0) if len(mt_median) == 5 else mt_median
    med_mt = pmt.MomentTensor(m=build_moment_tensor(np.asarray(mt_median6, dtype=np.float64)))
    med_norm = pmt.MomentTensor(m=med_mt.m() / med_mt.scalar_moment()).deviatoric()
    fig = plt.figure(figsize=(7., 7.))
    ax = fig.add_subplot(1, 1, 1)
    _plot_fuzzy_beachball_all(all_mats, ax, best_norm, size=250, position=(0, 0),
                              color_t='black', edgecolor='black',
                              extra_mts=[(med_norm, 'limegreen')])
    ax.set_xlim(-150, 150); ax.set_ylim(-150., 150)
    ax.set_aspect('equal'); ax.set_axis_off()
    plt.title("Moment Tensor Uncertainty\nBootstrap Ensemble, N = %d"
              % all_mats.shape[0], fontsize=16, fontweight='bold')
    ax.legend(handles=[Line2D([0], [0], color='red', lw=2, label='Best Solution'),
                       Line2D([0], [0], color='limegreen', lw=2, label='Median Solution')],
              loc='upper right', bbox_to_anchor=(1.1, 0.95))
    fig.savefig(os.path.join(c['plotdir'], _rv('beachball_fuzzy.png')), dpi=150,
                bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


def _fig_lune():
    # lune source-type plot of ALL ensemble members (no pyrocko needed)
    c = _FIG_CTX
    mt_median = c['mt_median']
    mt_median6 = np.append(mt_median, 0) if len(mt_median) == 5 else mt_median
    _lune_plot(_fig_all_mats(c), c['VR_all'],
               build_moment_tensor(np.asarray(mt_median6, dtype=np.float64)),
               np.asarray(c['result']['Tensor'], dtype=float),
               os.path.join(c['plotdir'], _rv('lune_median.png')))


# one task per bootstrap figure (each independent: reads _FIG_CTX, writes one PNG)
_BOOTSTRAP_FIG_TASKS = [_fig_fit_waveforms, _fig_decomposition, _fig_magnitudes,
                        _fig_centroid_time, _fig_iso, _fig_tradeoff, _fig_tradeoff_iso,
                        _fig_hudson, _fig_lune, _fig_fuzzy]


def _make_bootstrap_figures(**ctx):
    """All bootstrap figures, sequentially in this process (used by tests and
    as fallback; exportResults runs the same tasks in parallel)."""
    global _FIG_CTX
    _FIG_CTX = ctx
    for fn in _BOOTSTRAP_FIG_TASKS:
        fn()


def _run_fig_task(fn):
    """Worker entry point: run one figure task, return its wall time.
    Module-level so the ProcessPoolExecutor can pickle it."""
    t0 = time.perf_counter()        # monotonic clock (the wall clock can jump under WSL)
    try:
        fn()
    finally:
        plt.close('all')
    return time.perf_counter() - t0


def _run_figures_parallel(tasks):
    """Run independent figure tasks in parallel worker processes (fork: the
    workers inherit config, _BOOT_CTX and _FIG_CTX without copying). Each
    task is submitted separately: a failing task is retried once in this
    process, and a crashed worker does not hang the run. Returns
    {task name: seconds} (retries: time of the retry)."""
    from concurrent.futures.process import BrokenProcessPool
    times = {}
    name = lambda fn: '{}.{}'.format(fn.__module__, fn.__name__)
    pending = list(tasks)
    try:
        ctx = multiprocessing.get_context('fork')
        n_workers = max(1, min(len(tasks), config.n_cores()))
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers,
                                                    mp_context=ctx) as ex:
            futs = {ex.submit(_run_fig_task, fn): fn for fn in tasks}
            for fut in concurrent.futures.as_completed(futs):
                fn = futs[fut]
                try:
                    times[name(fn)] = fut.result()
                    pending.remove(fn)
                except BrokenProcessPool:
                    raise
                except Exception as _e:
                    config.logger.warning('Plot task %s failed in worker: %s', name(fn), _e)
    except Exception as _e:
        config.logger.warning('Parallel plotting interrupted (%s) - running the '
                              'remaining %d task(s) sequentially', _e, len(pending))
    # retry failed / not-run tasks sequentially in this process
    for fn in pending:
        try:
            times[name(fn)] = _run_fig_task(fn)
        except Exception:
            import traceback
            traceback.print_exc()
            print('WARNING: plot {} failed (results unaffected)'.format(name(fn)))
    return times


# =====================================================================
# RESULTS EXPORT + WEBSITE (replaces gatherResults/allplots for the
# Python inversion; reuses the original plot.py / web.py unchanged)
# =====================================================================

def _read_grid(igrid=None):
    if igrid is None:
        igrid = (_BOOT_CTX or {}).get('igrid', '0')
    files = sorted(
        glob.glob(os.path.join(config.workdir, 'sources', 'grid' + str(igrid), '*.dat')),
        key=lambda f: int(''.join(c for c in os.path.basename(f) if c.isdigit())))
    return np.vstack([np.loadtxt(f, usecols=(0, 1, 2, 3), ndmin=2) for f in files])


def doPlots(stage_t=None):
    """Export inversion results and generate all figures.

    timing_summary.dat is written by calculateInversions() after doPlots()
    and renderSite() both complete, so the combined 'export results + plots'
    time is available.  stage_t is accepted here for API compatibility but
    the file is no longer written inside this function.
    """
    exportResults()


def _write_timing_summary(stage_t):
    """Write timing_summary.dat to config.outputdir."""
    try:
        _tl = []
        _tl.append('=' * 62)
        _tl.append(' PIPELINE TIMING SUMMARY')
        _tl.append('=' * 62)
        for _k, _v in stage_t.items():
            _tl.append(' {:<32}: {:8.2f} s'.format(_k, _v))
        _tl.append(' {:<32}: {:8.2f} s'.format('TOTAL', sum(stage_t.values())))
        _tl.append('=' * 62)
        _tp = os.path.join(config.outputdir, _rv('timing_summary.dat'))
        with open(_tp, 'w') as _f:
            _f.write('\n'.join(_tl) + '\n')
        config.logger.info('Timing summary saved to %s', _tp)
    except Exception as _e:
        config.logger.warning('Could not save timing_summary.dat: %s', _e)


def _write_inversion_parameters():
    """Write inversion_parameters.dat to config.outputdir using current in-memory state."""
    try:
        _keyinv      = config.get_keyinv(config.cfg, origin_data=config.org)
        _keyinv_desc = 'Deviatoric (5 MT components, no volumetric)' if _keyinv == 2 \
                       else 'Full moment tensor (6 MT components, isotropic allowed)'
        _freq   = config.freqRules
        _shifts = config.shiftRules
        _grids  = config.gridRules
        _vel    = [r['Filepath'] for r in config.crustalRules]
        _il = []
        _il.append('=' * 62)
        _il.append(' INVERSION PARAMETERS')
        _il.append('=' * 62)
        _il.append('')
        _il.append(' Inversion type (Keyinv={})'.format(_keyinv))
        _il.append('   {}'.format(_keyinv_desc))
        _il.append('   Compute device: {}'.format(config.cfg.get('ComputeDevice', 'CPU')))
        _stf_cfg  = config.cfg.get('Inversion', {}).get('STF', {'type': 'delta'})
        _stf_type = _stf_cfg.get('type', 'delta')
        if _stf_type == 'triangle':
            _stf_dur = float(_stf_cfg.get('duration', 5.0))
            _il.append('   Source time function: triangle  half-duration={:.2f}s  total={:.2f}s'.format(
                _stf_dur, 2 * _stf_dur))
        else:
            _il.append('   Source time function: delta (instantaneous)')
        _il.append('')
        _tw0 = config.windowRules[0] if config.windowRules else 0.0
        _Dt0 = _tw0 / 1024.0
        _nyq0 = 1.0 / (2.0 * _Dt0) if _Dt0 else 0.0
        _il.append(' Time window')
        _il.append('   Window: {:.2f} s'.format(_tw0))
        _il.append('   Dt: {:.4f} s  (Window/1024)'.format(_Dt0))
        _il.append('   Nyquist: {:.4f} Hz  (1/(2·Dt))'.format(_nyq0))
        _il.append('')
        _il.append(' Frequency bands applied (Hz)')
        for _fb in _freq:
            _il.append('   Fmin={:.4f} Hz  Fmax={:.4f} Hz'.format(_fb[0], _fb[-1]))
        _max_filt    = max(_fb[-1] for _fb in _freq)
        _nyq_samples, _freq_mult = _green_freq_settings()
        _setup       = _BOOT_CTX.get('setup', {})
        _nfreq_used  = _setup.get('nfreq', None)
        _tl_val      = float(_setup.get('tl', 1.0))
        _fs          = float(_setup.get('Fs', 1.0))
        # AXITRA frequencies are k/tl, k = 0 .. nfreq-1: the highest is (nfreq-1)/tl;
        # the Nyquist frequency is bin tl*Fs/2 (= 512 for 1024 samples)
        _nfreq_nyq   = int(round(_tl_val * _fs / 2))
        _f_nyq       = _fs / 2
        if _nyq_samples is not None:
            _il.append('   Axitra synthesis mode: NyquistSamples = {:d}'.format(int(_nyq_samples)))
            _nf = _nfreq_used if _nfreq_used is not None else int(_nyq_samples)
            _pct = 100.0 * (_nf - 1) / _nfreq_nyq if _nfreq_nyq else 0
            _il.append('   Synthesis up to: {:.4f} Hz  ({:.1f}% of data Nyquist {:.4f} Hz, nfreq = {:d})'.format(
                (_nf - 1) / _tl_val, _pct, _f_nyq, _nf))
        else:
            _il.append('   Axitra synthesis mode: FreqMultiplier = {} x Fmax = {:.4f} Hz'.format(
                _freq_mult, _freq_mult * _max_filt))
            if _nfreq_used is not None:
                _fmax_used = (_nfreq_used - 1) / _tl_val
                _pct = 100.0 * (_nfreq_used - 1) / _nfreq_nyq if _nfreq_nyq else 0
                _il.append('   nfreq = {:d}  →  {:.4f} Hz  ({:.1f}% of Nyquist {:.4f} Hz)'.format(
                    _nfreq_used, _fmax_used, _pct, _f_nyq))
        _il.append('')
        _il.append(' Time-shift search')
        for _sh in _shifts:
            _il.append('   min={}  step={}  max={}  (samples)'.format(*_sh))
        for _i, _sh in enumerate(_shifts):
            _tw = config.windowRules[_i] if _i < len(config.windowRules) else config.windowRules[-1]
            _Dt = round(_tw / 1024.0, 4)
            _il.append('   min={:.2f}s  step={:.2f}s  max={:.2f}s  (Dt={:.4f}s, window={:.1f}s)'.format(
                _sh[0] * _Dt, _sh[1] * _Dt, _sh[2] * _Dt, _Dt, _tw))
        _il.append('')
        _il.append(' Source grid rules (km)')
        for _gr in _grids:
            _il.append('   Magnitude rule: {}'.format(_gr['Rule']))
            for _dr in _gr['Distance']:
                _il.append('     Horizontal: {} to {} km step {} km'.format(_dr[0], _dr[1], _dr[2]))
            for _zr in _gr['Depth']:
                _il.append('     Depth:      0 to {} km step {} km'.format(_zr[1], _zr[2]))
        _il.append('')
        _il.append(' Bootstrap centroid box')
        _cbi = (_BOOT_CTX or {}).get('centroid_box')
        if _cbi:
            for _line in _centroid_box_lines(_cbi):
                _il.append('   ' + _line)
        if _SUMMARY is not None:
            _sr  = _SUMMARY.get('result', {})
            _xc, _yc, _zc = _sr.get('Xcmt'), _sr.get('Ycmt'), _sr.get('Zcmt')
            if None not in (_xc, _yc, _zc):
                _il.append('   Best centroid: X={:.2f} Y={:.2f} Z={:.2f} km'.format(_xc, _yc, _zc))
            _btmin = _SUMMARY.get('boot_tminb')
            _btmax = _SUMMARY.get('boot_tmaxb')
            _bnsft = _SUMMARY.get('boot_n_shifts')
            _bDt   = _SUMMARY.get('boot_Dt')
            if _btmin is not None and _bDt is not None:
                _il.append('   Time-shift: min={:.2f}s  max={:.2f}s  ({} values, Dt={:.4f}s)'.format(
                    _btmin * _bDt, _btmax * _bDt, _bnsft, _bDt))
        _il.append('')
        _il.append(' Velocity model(s)')
        for _vm in _vel:
            _il.append('   {}'.format(os.path.basename(_vm)))
        _il.append('')
        _il.append('=' * 62)
        _ipar_path = os.path.join(config.outputdir, _rv('inversion_parameters.dat'))
        with open(_ipar_path, 'w') as _ipar_f:
            _ipar_f.write('\n'.join(_il) + '\n')
        config.logger.info('Inversion parameters saved to %s', _ipar_path)
    except Exception as _e:
        config.logger.warning('Could not save inversion_parameters.dat: %s', _e)


def _combo_visualize(stage_t=None):
    """Plots + site + inversion_parameters for the current combo (config.outputdir).
    Accumulates 'export results + plots' in stage_t but does NOT write timing_summary.
    """
    import time as _tm
    _t = _tm.perf_counter()
    doPlots()
    renderSite()
    _write_inversion_parameters()
    _dt = _tm.perf_counter() - _t
    print('[STAGE] {:<38s}: {:7.2f} s'.format('export results + plots', _dt), flush=True)
    if stage_t is not None:
        stage_t['export results + plots'] = \
            stage_t.get('export results + plots', 0.0) + _dt


def _render_home():
    """Render the catalog home page (web.renderHome) once for the current
    event. It lists every event of the year that has <run>/output/event.xml,
    so it must run after the best combination has been promoted to output/."""
    try:
        import web as _webmod
        _webmod.renderHome()
        config.logger.info('Home page rendered')
    except Exception:
        import traceback
        traceback.print_exc()
        print('WARNING: home page rendering failed (results unaffected)')
    finally:
        plt.close('all')


def _notify():
    """
    Notifications once per event, after the final event.xml and index.html are in
    <workdir>/output (as old Gisola's plot.allplots): the e-mail (web.email, if
    Notification.Email.Smtp is set) and Notification.Command, with $bestinvdir
    (name of the best combination folder, config.best_combo) and $outputdir replaced. (Old Gisola
    replaced $bestinvdir and then overwrote it, so only $outputdir was replaced.)
    A failure is reported and does not stop the run.
    """
    notif = config.cfg.get('Notification') or {}
    outdir = os.path.join(config.workdir, 'output')
    if (notif.get('Email') or {}).get('Smtp'):
        _saved = (config.outputdir, config.evt)
        try:
            import web as _webmod
            from obspy import read_events as _read_events
            config.outputdir = outdir
            config.evt = _read_events(os.path.join(outdir, 'event.xml'))[0]
            _webmod.email()
            config.logger.info('Notification e-mail sent')
        except Exception as _e:
            config.logger.warning('Notification e-mail failed: %s', _e)
            print('[WARN] Notification e-mail failed: {}'.format(_e), flush=True)
        finally:
            config.outputdir, config.evt = _saved
    _cmd = notif.get('Command')
    if _cmd:
        _cmd = str(_cmd).replace('$bestinvdir', str(getattr(config, 'best_combo', '') or '')) \
                        .replace('$outputdir', outdir)
        try:
            _proc = subprocess.run(_cmd, shell=True, universal_newlines=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            config.logger.info('Notification command: %s (exit %d)\n%s',
                               _cmd, _proc.returncode, _proc.stdout)
            if _proc.returncode != 0:
                print('[WARN] Notification command exited with {}: {}'.format(
                      _proc.returncode, _cmd), flush=True)
        except Exception as _e:
            config.logger.warning('Notification command failed: %s', _e)
            print('[WARN] Notification command failed: {}'.format(_e), flush=True)


def visualizeResults(stage_t=None):
    """Single-combo visualize + timing summary.
    Used by revise mode (one combo per call). As in old Gisola the revision is
    written into the run's output/: event.xml (reviewed solution preferred)
    and index.html are updated, every other file gets a '.revise' suffix.
    For the multi-combo normal run, calculateInversions() calls _combo_visualize()
    internally per combo and handles timing + promotion itself.
    """
    _combo_visualize(stage_t)
    _render_home()
    _notify()
    if stage_t is not None:
        _write_timing_summary(stage_t)



@config.time
def exportResults():
    """
    Builds everything the original site pipeline consumed from the Fortran
    outputs - but from the in-memory Python inversion (_BOOT_CTX):
      output/solutions, output/correlations, per-station fil/syn files,
      dsretc.lst, and the QuakeML output/event.xml (+ event_sc.xml).
    After this, the ORIGINAL plot.py / web.py run unchanged.
    """
    if _BOOT_CTX is None:
        config.logger.info('No inversion in memory - run getInversions() first')
        return

    G = _BOOT_CTX["G"]
    vr = _BOOT_CTX["vr"]
    shift = _BOOT_CTX["shift"]
    corr = _BOOT_CTX["corr"]
    best_idx = int(_BOOT_CTX["best_idx"])
    Dobs = _BOOT_CTX["Dobs"]
    Dt = _BOOT_CTX["Dt"]
    setup = _BOOT_CTX["setup"]
    result = _BOOT_CTX["result"]
    vr_mat = _BOOT_CTX["vr_mat"]
    mech_mat = _BOOT_CTX["mech_mat"]
    angl = _BOOT_CTX["angl"]

    Timeshifts = np.asarray(setup["Timeshifts"], dtype=np.int64)
    tl = Dt * 1024.0
    n_sources = G.shape[0]

    # QuakeML is built on a fresh copy of the event as it was before any
    # combination (or, in revise mode, the event prepared by
    # prepare_revision_base): solutions of other combinations never accumulate
    import copy as _copy
    evt = _copy.deepcopy(_EVT_BASE if _EVT_BASE is not None else config.evt)
    revise = bool(getattr(config, 'revise', False))
    workdir = config.workdir
    os.makedirs(config.outputdir, exist_ok=True)

    # pseudo "best inversion dir" so the original plot functions find files
    config.bestinvdir = 'python'
    config.inversiondir = os.path.join(workdir, 'inversions')
    config.besttl = tl
    pseudo = os.path.join(config.inversiondir, config.bestinvdir)
    os.makedirs(pseudo, exist_ok=True)

    grid = _read_grid()[:n_sources]
    gid, gx, gy, gz = grid[:, 0], grid[:, 1], grid[:, 2], grid[:, 3]
    bx, by, bz = gx[best_idx], gy[best_idx], gz[best_idx]

    plane1_all, plane2_all = angl[0], angl[1]
    M0_all, DC_all = angl[2], angl[4]

    # ----------------------------------------------------------------
    # 1. solutions file (same csv-ish format the old gatherResults wrote)
    #    [id, x, y, z, shift_samples, corr, moment, dc, s1,d1,r1, s2,d2,r2]
    # ----------------------------------------------------------------
    solutions = []
    for k in range(n_sources):
        solutions.append([float(k + 1), float(gx[k]), float(gy[k]), float(gz[k]),
                          float(shift[k]), float(corr[k]), float(M0_all[k]),
                          float(DC_all[k]),
                          float(plane1_all[k][0]), float(plane1_all[k][1]), float(plane1_all[k][2]),
                          float(plane2_all[k][0]), float(plane2_all[k][1]), float(plane2_all[k][2])])
    with open(os.path.join(config.outputdir, _rv('solutions')), 'w') as f:
        f.writelines(str(x)[1:-1] + '\n' for x in sorted(solutions))
    config.solutions = solutions

    # ----------------------------------------------------------------
    # 2. correlations file: every shift for all depths at best (x, y)
    #    [id, x,y,z, time_s, corr, s1,d1,r1, s2,d2,r2, dc, iso, vard, mom]
    # ----------------------------------------------------------------
    col = np.where((gx == bx) & (gy == by))[0]
    pairs = [(int(k), int(si)) for k in col for si in range(len(Timeshifts))]
    # moment tensors of these pairs from the inversion (as ISOLA's corr00.dat)
    mechs = np.ascontiguousarray(mech_mat[[k for k, _ in pairs], [si for _, si in pairs]])
    pang = silsub(mechs)

    corr_mat = np.sqrt(np.clip(vr_mat, 0.0, None) / 100.0)
    # data power of the used components, as RRORI in ISOLA15
    rr_data = float(np.dot(Dobs, Dobs) * Dt)

    correlations = []
    for idx, (k, si) in enumerate(pairs):
        correlations.append([
            float(k + 1), float(gx[k]), float(gy[k]), float(gz[k]),
            float(Timeshifts[si] * Dt), float(corr_mat[k, si]),
            float(pang[0][idx][0]), float(pang[0][idx][1]), float(pang[0][idx][2]),
            float(pang[1][idx][0]), float(pang[1][idx][1]), float(pang[1][idx][2]),
            float(pang[4][idx]), float(pang[6][idx]),
            (1.0 - float(corr_mat[k, si]) ** 2) * rr_data,      # misfit, as ISOLA's corr00.dat
            float(pang[2][idx])])
    with open(os.path.join(config.outputdir, _rv('correlations')), 'w') as f:
        f.writelines(str(x)[1:-1] + '\n' for x in sorted(correlations))
    config.correlations = correlations

    # ----------------------------------------------------------------
    # 3. quality metrics: stvar / fmvar (same definitions as gatherResults)
    # ----------------------------------------------------------------
    bestcorr = float(corr[best_idx])
    thres = 0.9 * bestcorr
    stvar = float(np.mean(corr_mat >= thres))

    # fmvar (as old Gisola): mean Kagan angle between the best solution and
    # EVERY (grid point, time shift) pair above the threshold - the same set
    # as stvar. The moment tensors of those pairs are the inversion's own
    # (mech_mat), as old Gisola reads them from ISOLA's corr00.dat.
    bp1 = result["BestPlane1"]          # also used for the QuakeML nodal planes below
    fm_mask = corr_mat >= thres
    fmvar = 0.0
    if fm_mask.any():
        fm_planes = silsub(np.ascontiguousarray(mech_mat[fm_mask]))[0]
        fmvar = float(np.mean(_kagan_angles(bp1, fm_planes)))

    # ----------------------------------------------------------------
    # 4. per-station fil/syn files + dsretc.lst (for plot.misfit/text/emsc)
    # ----------------------------------------------------------------
    allstat_lines = setup["allstat_lines"]
    nr = setup["nr"]
    keep_mask = np.ones(nr * 3 * 1024, dtype=np.bool_)
    keep_mask[setup["removeIdx"]] = False
    ch_keep = keep_mask.reshape(nr * 3, 1024)[:, 0]
    kept_rows = np.where(ch_keep)[0]
    row_of = {(int(idx // 3), int(idx % 3)): j for j, idx in enumerate(kept_rows)}

    Dsyn = np.asarray(result["Dsynth"])
    tcol = np.arange(1024) * Dt

    for i, line in enumerate(allstat_lines):
        sta = line.split()[0]
        cols_obs, cols_syn = [tcol], [tcol]
        for rec in range(3):                      # N, E, Z
            j = row_of.get((i, rec))
            if j is None:
                cols_obs.append(np.zeros(1024))
                cols_syn.append(np.zeros(1024))
            else:
                cols_obs.append(Dobs[j * 1024:(j + 1) * 1024])
                cols_syn.append(Dsyn[j * 1024:(j + 1) * 1024])
        np.savetxt(os.path.join(pseudo, sta + 'fil.dat'), np.column_stack(cols_obs), fmt='%.6e')
        np.savetxt(os.path.join(pseudo, sta + 'syn.dat'), np.column_stack(cols_syn), fmt='%.6e')

    bp2 = result["BestPlane2"]
    with open(os.path.join(pseudo, 'dsretc.lst'), 'w') as f:
        f.write('Gisola2 Python inversion\n\n'
                '   NP1: strike {:.0f}  dip {:.0f}  rake {:.0f}\n'
                '   NP2: strike {:.0f}  dip {:.0f}  rake {:.0f}\n'
                '   P axis: azm {:.0f} plunge {:.0f}   T axis: azm {:.0f} plunge {:.0f}\n'
                .format(bp1[0], bp1[1], bp1[2], bp2[0], bp2[1], bp2[2],
                        result["BestP"][0], result["BestP"][1],
                        result["BestT"][0], result["BestT"][1]))

    # ----------------------------------------------------------------
    # 5. QuakeML event (mirrors the original gatherResults, values ours)
    # ----------------------------------------------------------------
    fm = FocalMechanism()
    mt = MomentTensor()
    org = Origin()
    mag = Magnitude()

    horg = event.getOrigin(config.cfg, evt, config.cfg['Watcher']['Historical'])
    org.time = horg.time + float(result["BestShift"]) * (tl / NUM_OF_TIME_SAMPLES)
    _lat, _lon = cart2earth(horg.latitude, horg.longitude, bx, by)
    org.latitude = _lat
    org.longitude = _lon
    org.depth = bz * 1000
    org.depth_type = 'from moment tensor inversion'
    org.time_fixed = False
    org.epicenter_fixed = False
    org.origin_type = 'centroid'
    org.creation_info = CreationInfo(agency_id=config.cfg['Citation']['Agency'],
                                     author=config.cfg['Citation']['Author'],
                                     version=config.cfg['Citation']['Version'],
                                     creation_time=UTCDateTime.now())
    mag.origin_id = org.resource_id
    mag.magnitude_type = 'Mw'
    mag.creation_info = org.creation_info
    mt.creation_info = org.creation_info
    mt.derived_origin_id = org.resource_id
    mt.moment_magnitude_id = mag.resource_id
    mt.category = 'regional'
    # QuakeML: 'zero trace' = deviatoric (5 components), 'general' = full moment
    # tensor (6 components, IsotropicGeoboxes / KEYINV = 1); old Gisola always
    # wrote 'zero trace' (and iso = 0)
    mt.inversion_type = 'general' if G.shape[2] == 6 else 'zero trace'
    org.evaluation_mode = 'automatic'
    # as old Gisola: an automatic run is 'preliminary', a revision 'reviewed'
    org.evaluation_status = 'reviewed' if revise else 'preliminary'

    mag.mag = float(result["BestMw"])
    mt.scalar_moment = float(result["BestMo"])
    # 0.1 % as old Gisola (ISOLA's inv1.dat has one decimal); also used by the
    # quality digit below, so its CLVD limits behave as in old Gisola
    mt.iso = round(float(result["BestISO"]), 1) / 100.0 + 0.0   # + 0.0: no "-0.0"
    mt.double_couple = round(float(result["BestDC"]), 1) / 100.0 + 0.0   # + 0.0: no "-0.0"
    mt.clvd = round(float(result["BestCLVD"]), 1) / 100.0 + 0.0   # + 0.0: no "-0.0"
    fm.nodal_planes = NodalPlanes(
        nodal_plane_1=NodalPlane(strike=round(float(bp1[0])), dip=round(float(bp1[1])), rake=round(float(bp1[2]))),
        nodal_plane_2=NodalPlane(strike=round(float(bp2[0])), dip=round(float(bp2[1])), rake=round(float(bp2[2]))))
    fm.principal_axes = PrincipalAxes(
        p_axis=Axis(azimuth=round(float(result["BestP"][0])), plunge=round(float(result["BestP"][1]))),
        t_axis=Axis(azimuth=round(float(result["BestT"][0])), plunge=round(float(result["BestT"][1]))),
        n_axis=Axis(azimuth=round(float(result["BestB"][0])), plunge=round(float(result["BestB"][1]))))
    mt.variance = float(result["VR"]) / 100.0
    mt.variance_reduction = float(result["VR"])

    # NED (x=N, y=E, z=D) -> RTP/USE (r=Up, t=South, p=East)
    M = np.asarray(result["Tensor"], dtype=float)
    mt.tensor = Tensor(m_rr=M[2, 2], m_tt=M[0, 0], m_pp=M[1, 1],
                       m_rt=M[0, 2], m_rp=-M[1, 2], m_tp=-M[0, 1])

    minsn = float(np.min(result["Singulars"]))
    maxsn = float(np.max(result["Singulars"]))
    conum = float(result["Condition"])

    text = allstat_lines
    # allstat has either 7 tokens (2-value: STA active N E Z fmin fmax)
    # or 9 tokens (4-value: STA active N E Z f1 f2 f3 f4).
    # [-1] is always fmax; fmin is [-2] for 2-value and [-4] for 4-value.
    _aparts = text[0].split()
    _fmax = float(_aparts[-1])
    _fmin = float(_aparts[-4]) if len(_aparts) >= 9 else float(_aparts[-2])
    # stations actually used: station flag set and at least one component.
    # Rows switched off in revise mode (-s/--remove, hand edits) stay in
    # allstat with zero flags; they must not count as used stations.
    used_rows = [l.split() for l in text
                 if int(l.split()[1]) and any(int(c) for c in l.split()[2:5])]
    used_sta = {p[0] for p in used_rows}
    mt.data_used = [DataUsed(wave_type='combined', station_count=len(used_sta),
                             component_count=sum(int(c) for p in used_rows for c in p[2:5]),
                             shortest_period=1 / _fmax if _fmax else None,
                             longest_period=1 / _fmin if _fmin else None)]

    st = read(os.path.join(workdir, 'streams_corrected.mseed'), headonly=True)
    with open(os.path.join(workdir, 'locations'), 'r') as f:
        stationinfo = f.readlines()
    # gap from the used stations only (used_sta above). Old Gisola took the
    # first CHARACTER of each allstat line and kept stations 'not in' that
    # list, i.e. ALL stations in 'locations' - every station that passed QC,
    # before the per-sector cut - so the gap was under-reported.
    stationinfo = list(map(eval, stationinfo))
    stationinfo2 = [s for s in stationinfo if s[0] in used_sta]
    dist = [float(_[4][0]) / 1000 for _ in stationinfo2]
    azm = np.array([float(_[4][1]) for _ in stationinfo2])
    azm.sort()
    azm = azm - np.roll(azm, 1)
    mag.azimuthal_gap = max(np.where(azm <= 0, 360 + azm, azm))

    components = AttribDict()
    components.namespace = 'custom'
    components.value = AttribDict()
    i = 1
    for li, line in enumerate(text):
        w = [0, 0, 0]
        tr = st.select(station=line.split()[0])[0]
        stainfo = [_ for _ in stationinfo if _[0] == tr.stats.station][0]
        sta, sw, w[0], w[1], w[2], *freqs = line.split()
        tr.stats['frequencies'] = freqs
        tr.stats['distance'] = round(float(stainfo[4][0]) / 1000, 1)
        tr.stats['latitude'] = float(stainfo[2])
        tr.stats['longitude'] = float(stainfo[3])
        for j, orient in enumerate(['N', 'E', 'Z']):
            tr.stats['channel'] = tr.stats.channel[:2] + orient
            tr.stats['weight'] = int(int(sw) and int(w[j]))
            rj = row_of.get((li, j))
            if tr.stats['weight'] and rj is not None:
                tr.stats['variance'] = calculateVariance(
                    Dobs[rj * 1024:(rj + 1) * 1024], Dsyn[rj * 1024:(rj + 1) * 1024], tl)
                fm.waveform_id.append(WaveformStreamID(
                    network_code=tr.stats.network, station_code=tr.stats.station,
                    location_code=tr.stats.location, channel_code=tr.stats.channel))
            else:
                tr.stats['variance'] = 'None'
                tr.stats['weight'] = 0
            for element in ['network', 'station', 'location', 'channel', 'latitude',
                            'longitude', 'variance', 'weight', 'distance', 'frequencies']:
                if element == 'network':
                    components.value['component_' + str(i)] = AttribDict()
                    components.value['component_' + str(i)].namespace = 'custom'
                    components.value['component_' + str(i)].value = AttribDict()
                components.value['component_' + str(i)].value[element] = AttribDict()
                components.value['component_' + str(i)].value[element].namespace = 'custom'
                components.value['component_' + str(i)].value[element].type = 'attribute'
                components.value['component_' + str(i)].value[element].value = tr.stats[element]
            i += 1

    fm.extra = AttribDict()
    fm.extra.components = components

    mag.station_count = mt.data_used[0].station_count
    mag.evaluation_mode = org.evaluation_mode
    mag.evaluation_status = org.evaluation_status
    fm.azimuthal_gap = mag.azimuthal_gap
    fm.triggering_origin_id = horg.resource_id
    fm.evaluation_mode = org.evaluation_mode
    fm.evaluation_status = org.evaluation_status
    fm.creation_info = org.creation_info
    fm.moment_tensor = mt
    org.quality = OriginQuality(used_station_count=mt.data_used[0].station_count,
                                azimuthal_gap=mag.azimuthal_gap,
                                minimum_distance=kilometers2degrees(min(dist)),
                                maximum_distance=kilometers2degrees(max(dist)),
                                median_distance=kilometers2degrees(float(np.median(dist))))

    if mt.variance >= 0.6 and mt.data_used[0].station_count > 4:
        quality = 'A'
    elif (mt.variance >= 0.4 and mt.variance < 0.6 and
          mt.data_used[0].station_count >= 4) or (mt.variance >= 0.7 and
          (mt.data_used[0].station_count == 2 or mt.data_used[0].station_count == 3)):
        quality = 'B'
    elif (mt.variance >= 0.15 and mt.variance < 0.4 and
          mt.data_used[0].station_count > 4) or (mt.variance >= 0.2 and
          mt.variance < 0.4 and mt.data_used[0].station_count == 4) or \
         (mt.variance >= 0.2 and mt.variance < 0.7 and
          mt.data_used[0].station_count == 3) or (mt.variance >= 0.3 and
          mt.variance < 0.7 and mt.data_used[0].station_count == 2):
        quality = 'C'
    else:
        quality = 'D'
    # quality digit from the size of the CLVD (CLVD is signed)
    if abs(mt.clvd) <= 0.2:
        quality += '1'
    elif abs(mt.clvd) <= 0.5:
        quality += '2'
    elif abs(mt.clvd) <= 0.8:
        quality += '3'
    else:
        quality += '4'

    mt.extra = OrderedDict()
    mt.extra['correlation'] = {'namespace': 'custom', 'value': bestcorr}
    mt.extra['quality'] = {'namespace': 'custom', 'value': quality}
    mt.extra['min_singular'] = {'namespace': 'custom', 'value': minsn}
    mt.extra['max_singular'] = {'namespace': 'custom', 'value': maxsn}
    mt.extra['condition_number'] = {'namespace': 'custom', 'value': conum}
    mt.extra['stvar'] = {'namespace': 'custom', 'value': stvar}
    mt.extra['fmvar'] = {'namespace': 'custom', 'value': fmvar}

    evt.focal_mechanisms.append(fm)
    evt.origins.append(org)
    evt.magnitudes.append(mag)
    evt.preferred_focal_mechanism_id = fm.resource_id

    evt.write(os.path.join(config.outputdir, 'event.xml'), format="QUAKEML")
    try:
        evt.write(os.path.join(config.outputdir, 'event_sc.xml'), format="SC3ML")
    except Exception:
        config.logger.info('SC3ML export failed (QuakeML written)')

    # Write lightweight summary JSON so renderHome avoids parsing the full event.xml
    try:
        import json as _json
        _summary_path = os.path.join(config.outputdir, 'event_summary.json')
        _summary = {
            'org_time':        str(org.time).split('.')[0].replace('T', ' '),
            'org_latitude':    float(org.latitude),
            'org_longitude':   float(org.longitude),
            'org_depth_m':     float(org.depth),
            'mag_mag':         float(mag.mag),
            'mag_type':        str(mag.magnitude_type),
            'scalar_moment':   float(mt.scalar_moment),
            'strike':          int(round(float(fm.nodal_planes.nodal_plane_1.strike))),
            'dip':             int(round(float(fm.nodal_planes.nodal_plane_1.dip))),
            'rake':            int(round(float(fm.nodal_planes.nodal_plane_1.rake))),
            'quality':         str(quality),
            'evaluation_status': str(org.evaluation_status),
            'hypo_latitude':   float(horg.latitude),
            'hypo_longitude':  float(horg.longitude),
        }
        with open(_summary_path, 'w') as _f:
            _json.dump(_summary, _f)
    except Exception as _je:
        config.logger.warning('Could not write event_summary.json: %s', _je)

    config.logger.info('exportResults: event.xml, solutions, correlations written')

    # ---- ALL figures in parallel: site plots (plot.py), STF response and the
    #      bootstrap figures, one task each, in one process pool ----
    import plot as plotmod
    from obspy.core.event import read_events as _read_evts
    config.evt = _read_evts(os.path.join(config.outputdir, 'event.xml'))[0]
    os.makedirs(os.path.join(config.outputdir, 'plots'), exist_ok=True)

    # the catalog home page (web.renderHome) is NOT rendered here: it reads
    # every event of the year, and must run once per event after the best
    # combination has been promoted to <workdir>/output (see _render_home)
    _tasks = [plotmod.beachball, plotmod.atlas, plotmod.streams, plotmod.misfit,
              plotmod.top, plotmod.northeast, plotmod.time, plotmod.text, plotmod.emsc]

    # bootstrap figures (plot data prepared by getBootstrapping); the workers
    # inherit it through _FIG_CTX
    global _BOOT_PLOT_CTX, _FIG_CTX
    if _BOOT_PLOT_CTX is not None:
        _FIG_CTX = _BOOT_PLOT_CTX
        _tasks += _BOOTSTRAP_FIG_TASKS

    _t_figs = time.perf_counter()
    _fig_times = _run_figures_parallel(_tasks)
    _fig_wall = time.perf_counter() - _t_figs
    _FIG_CTX = None
    _BOOT_PLOT_CTX = None

    # per-figure timing (wall time of each task in its worker)
    for _fname, _ft in sorted(_fig_times.items(), key=lambda kv: -kv[1]):
        config.logger.info('plot %-40s %7.2f s', _fname, _ft)
    if _SUMMARY is not None:
        _SUMMARY['plotsTime'] = _fig_wall


@config.time
def renderSite():
    """
    Renders the HTML site only (all figures already generated by exportResults()).
    Calls web.py to build index.html, then embeds the bootstrap section.
    The catalog home page is rendered separately, once per event, by
    _render_home() after the best combination is promoted to output/.
    """
    import web as webmod

    webmod.renderEvent()

    # Embed bootstrap section into index.html if bootstrapping was run
    if _BOOT_STATS is not None:
        try:
            _renderBootstrapPage(*_BOOT_STATS)
        except Exception:
            import traceback
            traceback.print_exc()
            print('WARNING: bootstrap section embedding failed (index.html still valid)')

    config.logger.info('renderSite: index.html written in ' + config.outputdir)


def _renderBootstrapPage(plotdir, st, result):
    """
    Embeds bootstrap sections directly into index.html (the single final page),
    in the SAME visual style as the event page (Montserrat, centred bold titles
    with a thin grey rule), placed right after the "Best Moment Tensor Solution"
    section and BEFORE the Gisola logo footer. Sections, in order:
       1. Moment Tensor Statistics  - fuzzy uncertainty beachball (left) +
          median-MT planes, Mw, Mo; ensemble size and VR range.
       2. Ensemble Statistics - median, 68% and 95% CI of Mw, DC, CLVD,
          ISO, x/y/z and centroid time + centroid/time, decomposition and
          Mo/Mw histograms.
       3. Uncertainty Visualization  - Hudson, lune, depth/VR trade-off, ISO.
       4. Bootstrap Box - source grid with the bootstrap subset (full width).
    """
    import re
    idx_path = os.path.join(config.outputdir, 'index.html')
    if not os.path.isfile(idx_path):
        print('index.html not found - run renderSite() first; bootstrap page skipped')
        return
    html = open(idx_path, encoding='utf-8').read()
    rel = os.path.basename(plotdir)

    FONT = "'Montserrat',sans-serif"
    BLUE = '#3AAEE0'

    # ---- style helpers that mirror the event-page look ----
    def title(text, size=20):
        # thin grey rule + centred bold h2, exactly like the template panels
        return (
            '<table role="presentation" cellpadding="0" cellspacing="0" width="100%" '
            'border="0" style="font-family:{f}"><tbody><tr>'
            '<td style="padding:5px" align="left">'
            '<table height="0px" align="center" border="0" cellpadding="0" '
            'cellspacing="0" width="100%" style="border-collapse:collapse;'
            'border-top:1px solid #BBBBBB"><tbody><tr style="vertical-align:top">'
            '<td style="font-size:0px;line-height:0px">&#160;</td></tr></tbody>'
            '</table></td></tr></tbody></table>'
            '<h2 style="margin:0;color:#000;line-height:140%;text-align:center;'
            'word-wrap:break-word;font-weight:normal;font-family:{f};font-size:{s}px">'
            '<strong>{t}</strong></h2>').format(f=FONT, t=text, s=size)

    def img(f, css):
        f = _rv(f)
        if not os.path.isfile(os.path.join(plotdir, f)):
            return ''
        return ('<a href="{r}/{f}" target="_blank"><img src="{r}/{f}" alt="{f}" '
                'style="{c}"></a>').format(r=rel, f=f, c=css)

    def kv(label, value):
        return ('<p style="font-size:14px;line-height:140%;margin:5px 0;'
                'font-family:{f}"><strong>{l}:</strong> {v}</p>').format(
            f=FONT, l=label, v=value)

    # ---- section 1: Moment Tensor Statistics (beachball left, stats right) ----
    # planes, M0 and Mw of the median MT (paper, Table 1)
    p1, p2 = st['p1_med'], st['p2_med']

    mt_stats = (
        kv('NP1 strike/dip/rake',
           '{:.0f} / {:.0f} / {:.0f}'.format(p1[0], p1[1], p1[2]))
        + kv('NP2 strike/dip/rake',
             '{:.0f} / {:.0f} / {:.0f}'.format(p2[0], p2[1], p2[2]))
        + kv('Mw', '{:.2f}'.format(st['Mw_mt']))
        + kv('Mo (Nm)', '{:.3e}'.format(st['Mo_mt']))
        + kv('Ensemble members',
             '{} (from {} resamplings)'.format(st['n_solutions'], st['n_iterations']))
        + kv('VR range (%)', '{:.1f} &ndash; {:.1f}'.format(st['vr_min'], st['vr_max'])))

    mt_panel = (
        '<div style="display:flex;flex-wrap:wrap;gap:20px;align-items:center;'
        'justify-content:center;max-width:900px;margin:6px auto">'
        '<div style="flex:1 1 300px;min-width:260px;text-align:center">'
        + img('beachball_fuzzy.png', 'width:100%;max-width:320px;height:auto')
        + '</div>'
        '<div style="flex:1 1 340px;min-width:280px;text-align:left">'
        + mt_stats + '</div></div>')

    # ---- section 2: Location & Time statistics table + plot ----
    # same look as the event page tables: black 14px Montserrat, bold
    # headers, thin grey rules, no coloured background
    TD = ('border-bottom:1px solid #DDDDDD;padding:6px 12px;font-size:14px;'
          'line-height:140%;color:#000000;text-align:center;font-family:' + FONT)
    TH = ('border-bottom:1px solid #BBBBBB;padding:6px 12px;font-size:14px;'
          'color:#000000;font-weight:bold;text-align:center;font-family:' + FONT)

    def row(label, key, fmt):
        med, p16, p84, p025, p975 = st[key]
        return ('<tr><td style="{td};text-align:left"><strong>{l}</strong></td>'
                '<td style="{td}">{m}</td><td style="{td}">[{a}, {b}]</td>'
                '<td style="{td}">[{lo}, {hi}]</td></tr>').format(
            td=TD, l=label, m=_fmt_nz(fmt, med), a=_fmt_nz(fmt, p16), b=_fmt_nz(fmt, p84),
            lo=_fmt_nz(fmt, p025), hi=_fmt_nz(fmt, p975))

    loc_rows = (row('Mw', 'Mw', '{:.2f}')
                + row('DC (%)', 'DC', '{:.0f}')
                + row('CLVD (%)', 'CLVD', '{:.0f}')
                + (row('ISO (%)', 'ISO', '{:.0f}') if st.get('full_mt') else '')
                + row('X (km)', 'x', '{:.1f}')
                + row('Y (km)', 'y', '{:.1f}')
                + row('Z depth (km)', 'z', '{:.1f}')
                + row('Centroid time (s)', 'time', '{:.2f}'))
    loc_table = (
        '<table style="border-collapse:collapse;margin:8px auto;max-width:760px;'
        'width:100%">'
        '<tr><th style="{th};text-align:left">Quantity</th>'
        '<th style="{th}">Median</th><th style="{th}">68% CI</th>'
        '<th style="{th}">95% CI</th></tr>{r}</table>').format(th=TH, r=loc_rows)
    # centroid / time histograms, then decomposition and magnitude histograms
    loc_plot = ('<div style="text-align:center;margin-top:8px">'
                + img('hist_centroid_time.png', 'width:100%;height:auto')
                + '</div>'
                '<div style="text-align:center;margin-top:10px">'
                + img('hist_decomposition.png', 'width:100%;height:auto')
                + '</div>'
                '<div style="text-align:center;margin-top:10px">'
                + img('hist_magnitudes.png', 'width:100%;height:auto')
                + '</div>')

    # ---- section 3: Uncertainty Visualization ----
    # source type (Hudson, trade-off with depth and VR, decomposition / ISO
    # histograms) followed by magnitudes and the grid VR overview
    def _cell(f, basis, maxw, h=None):
        im = img(f, ('height:{}px;width:auto;max-width:100%'.format(h) if h
                     else 'width:100%;max-width:{}px;height:auto'.format(maxw)))
        if not im:                      # figure not made (e.g. ISO for deviatoric)
            return ''
        return ('<div style="flex:1 1 {b}px;min-width:300px;text-align:center">'
                .format(b=basis) + im + '</div>')

    unc = (
        '<div style="display:flex;flex-wrap:wrap;gap:16px;justify-content:center;'
        'align-items:flex-start;max-width:1300px;margin:6px auto">'
        + _cell('hudson_median.png', 380, 440, 480)
        + _cell('lune_median.png', 300, 330, 480)
        + _cell('tradeoff_depth_vr.png', 600, 880)
        + _cell('tradeoff_depth_vr_iso.png', 600, 880)      # below it, same size (full MT)
        + _cell('hist_iso.png', 380, 440)
        + '</div>')

    # ---- section 4: Bootstrap Box - source grid, bootstrap subset, best
    #      solution; full page width ----
    box = ('<div style="text-align:center;margin:6px 0">'
           + img('grid_vr_bootstrap.png', 'width:100%;height:auto')
           + '</div>')

    section = (
        '<div id="bootstrap-results" style="max-width:900px;margin:0 auto;'
        'padding:10px 20px 20px;background:#ffffff;font-family:' + FONT + '">'
        + title('Statistical Uncertainty Analysis', 24)
        + title('Median Moment Tensor') + mt_panel
        + title('Ensemble Statistics') + loc_table + loc_plot
        + title('Uncertainty Visualization') + unc
        + title('Bootstrap Box') + box
        + '</div>')

    # Insert right after the last content section and BEFORE the Gisola logo
    # footer (the final u-row-container in the page), so the logo stays last.
    anchor = html.rfind('<div class="u-row-container"')
    if anchor == -1:
        mm = list(re.finditer(r'</body\s*>', html, re.I))
        anchor = mm[-1].start() if mm else len(html)
    boot_html = html[:anchor] + section + html[anchor:]
    with open(idx_path, 'w', encoding='utf-8') as f:
        f.write(boot_html)
    config.logger.debug('Bootstrap section embedded into: %s', idx_path)


def getGreens():
    """
    All steps required for Greens' Functions calculation
    """
    config.logger.info('Creating sources files based on Grid rules')
    config.logger.info(config.dump(config.gridRules))
    createSources()

    config.logger.info('Selecting crustal model files based on Crustal rules')
    config.logger.info(config.dump(config.crustalRules))
    createCrustals()

    config.logger.info('Creating Greens\' Functions configuration files (grdat.hed)')
    createGrdat()

    config.logger.info('Creating Stations file (station.dat)')
    createStations()

    config.logger.info('Performing Greens\' Functions Computation')
    calculateGreens()

def getInversions():
    """
    Run the inversion kernel using the G matrix already in _BOOT_CTX
    (populated by getElemse()). Times the kernel, writes results_inversion.dat,
    and stores inv_dt in _BOOT_CTX['times']['wall'] for the caller to use.
    """
    config.logger.info(
        'Calculating inversions (kernel only — G already in memory from getElemse)')
    _t_inv = time.perf_counter()
    _run_kernel_only()
    inv_time = time.perf_counter() - _t_inv

    if _BOOT_CTX is not None:
        _BOOT_CTX['times']['kernel'] = inv_time

    # write results_inversion.dat
    try:
        if _BOOT_CTX is not None:
            _r = _BOOT_CTX['result']
            _lines = []
            def _w(line=''):
                _lines.append(line)
            _w()
            _w('================ INVERSION RESULTS ================')
            _w()
            _w('Event ID:    {}'.format(config.evt.resource_id.id.split('/')[-1]))
            _w('Origin time: {}'.format(config.org.time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4] + ' UTC'))
            _w('Hypocenter:  Lat={:.4f}  Lon={:.4f}  Depth={:.1f} km'.format(
                config.org.latitude, config.org.longitude, config.org.depth / 1000))
            _mag = [m for m in config.evt.magnitudes if m.origin_id == config.org.resource_id]
            if _mag:
                _w('Magnitude:   {}={:.1f}'.format(_mag[0].magnitude_type, _mag[0].mag))
            _w()
            _w('VR: {:.2f}'.format(_r['VR']))
            _w('Corr: {:.4f}'.format(_r['Corr']))
            _w('Best shift: {}'.format(_r['BestShift']))
            _w('Inversion time: {:.2f} s'.format(inv_time))
            _w('Best chunk: {}'.format(_r['BestChunk']))
            _w('Best point id: {}'.format(_r['BestIdx'] + 1))
            _w()
            _T = np.asarray(_r['Tensor'])     # 3x3 NED tensor of the best solution
            _w('Moment tensor (NED, N·m):')
            _w('Mxx = {:.4e}  Myy = {:.4e}  Mzz = {:.4e}'.format(_T[0, 0], _T[1, 1], _T[2, 2]))
            _w('Mxy = {:.4e}  Mxz = {:.4e}  Myz = {:.4e}'.format(_T[0, 1], _T[0, 2], _T[1, 2]))
            _w()
            # _fmt_nz: no negative zero (e.g. a rake of -0.2 or an ISO of -1e-16 prints as 0)
            for _ip, _pl in ((1, _r['BestPlane1']), (2, _r['BestPlane2'])):
                _w('Plane {}: Strike = {}, Dip = {}, Rake = {}'.format(
                    _ip, *(_fmt_nz('{:.0f}', _v) for _v in _pl)))
            _w()
            _w('M0: {:.4e}'.format(_r['BestMo']))
            _w('Mw: {:.1f}'.format(_r['BestMw']))
            _w('DC%: ' + _fmt_nz('{:.0f}', _r['BestDC']))
            _w('CLVD%: ' + _fmt_nz('{:.0f}', _r['BestCLVD']))
            _w('ISO: ' + _fmt_nz('{:.0f}', _r['BestISO']))
            _w()
            _w('P-Axis  Azimuth: {:.0f}  Plunge: {:.0f}'.format(_r['BestP'][0], _r['BestP'][1]))
            _w('T-Axis  Azimuth: {:.0f}  Plunge: {:.0f}'.format(_r['BestT'][0], _r['BestT'][1]))
            _w('B-Axis  Azimuth: {:.0f}  Plunge: {:.0f}'.format(_r['BestB'][0], _r['BestB'][1]))
            _w()
            _w('Singulars: {}'.format(np.array2string(_r['Singulars'], formatter={'float_kind': lambda x: '{:.4e}'.format(x)})))
            _w('Condition number: {:.2f}'.format(_r['Condition']))
            if 'Xcmt' in _r:
                _w()
                _w('CMT grid position:  X={:.2f} km  Y={:.2f} km  Depth={:.2f} km'.format(
                    _r['Xcmt'], _r['Ycmt'], _r['Zcmt']))
            if 'CMT_lat' in _r:
                _w('CMT location:       Lat={:.4f}  Lon={:.4f}'.format(
                    _r['CMT_lat'], _r['CMT_lon']))
            _w()
            _w('==================================================')
            _inv_path = os.path.join(config.outputdir, _rv('results_inversion.dat'))
            with open(_inv_path, 'w') as _f:
                _f.write('\n'.join(_lines) + '\n')
            config.logger.info('Inversion results saved to %s', _inv_path)
    except Exception as _e:
        config.logger.warning('Could not save results_inversion.dat: %s', _e)

    return {'inversion (kernel)': inv_time}


def _run_kernel_only():
    """
    Run the inversion kernel using the G matrix already in _BOOT_CTX.

    Called by getInversions() when getElemse() was called first.
    Updates _BOOT_CTX in-place with the full inversion result.
    Also back-fills vr_all in _GRAPH_SEARCH_ALAS dict
    if it was populated before the full inversion ran.
    """
    global _BOOT_CTX, _GRAPH_SEARCH_ALAS

    G          = _BOOT_CTX["G"]
    setup      = _BOOT_CTX["setup"]
    Dobs       = _BOOT_CTX["Dobs"]
    Dt         = _BOOT_CTX["Dt"]
    _grid      = _BOOT_CTX["grid"]

    Timeshifts   = np.ascontiguousarray(setup["Timeshifts"], dtype=np.int64)
    nr           = setup["nr"]
    NFFT         = setup["NFFT"]
    chunk_sizes  = _BOOT_CTX.get("chunk_sizes", [G.shape[0]])
    chunk_bounds = np.cumsum([0] + chunk_sizes)

    valid = _BOOT_CTX.get("valid")

    t_kernel = time.perf_counter()
    if valid is None or valid.all():
        shift, mech, best_idx, vr, corr, singulars, condition, vr_mat, mech_mat = \
            _compute_inversion(G, Dobs, Dt, Timeshifts, nr)
    else:
        # rejected sources (see _check_inversion_inputs) never reach the
        # kernel; they get the kernel's own "no result" values (VR = -1)
        vidx = np.flatnonzero(valid)
        s_shift, s_mech, s_best, s_vr, s_corr, s_sing, s_cond, s_vrmat, s_mechmat = \
            _compute_inversion(np.ascontiguousarray(G[vidx]), Dobs, Dt,
                               Timeshifts, nr)
        n_src = G.shape[0]
        shift     = np.full(n_src, Timeshifts[0], dtype=np.int64)
        mech      = np.zeros((n_src, s_mech.shape[1]))
        vr        = np.full(n_src, -1.0)
        corr      = np.full(n_src, -1.0)
        singulars = np.zeros((n_src, s_sing.shape[1]))
        condition = np.zeros(n_src)
        vr_mat    = np.full((n_src, s_vrmat.shape[1]), -1.0)
        mech_mat  = np.zeros((n_src,) + s_mechmat.shape[1:])
        shift[vidx], mech[vidx], vr[vidx], corr[vidx] = s_shift, s_mech, s_vr, s_corr
        singulars[vidx], condition[vidx], vr_mat[vidx] = s_sing, s_cond, s_vrmat
        mech_mat[vidx] = s_mechmat
        best_idx = int(vidx[s_best])
    kernel_time = time.perf_counter() - t_kernel

    angl = silsub(mech)

    BestVR    = vr[best_idx]
    BestCorr  = corr[best_idx]
    BestMT    = mech[best_idx]
    Bestshift = shift[best_idx]
    BestChunk = int(np.searchsorted(chunk_bounds, best_idx, side='right') - 1)

    BestGF     = G[best_idx]
    n_stations = BestGF.shape[0] // NFFT
    G_reshaped = BestGF.reshape(n_stations, NFFT, BestGF.shape[1])
    G_shifted_mat = np.zeros_like(G_reshaped)

    if Bestshift > 0:
        G_shifted_mat[:, Bestshift:, :] = G_reshaped[:, :-Bestshift, :]
    elif Bestshift < 0:
        s_abs = abs(Bestshift)
        G_shifted_mat[:, :-s_abs, :] = G_reshaped[:, s_abs:, :]
    else:
        G_shifted_mat = G_reshaped

    GB_shift = G_shifted_mat.reshape(BestGF.shape[0], BestGF.shape[1])
    DsynthB  = GB_shift @ BestMT

    result = {
        "VR": BestVR, "Corr": BestCorr,
        "BestIdx": int(best_idx),
        "BestShift": Bestshift, "BestChunk": BestChunk,
        "BestMT": BestMT, "BestGF": BestGF, "Dsynth": DsynthB,
        "BestPlane1": angl[0][best_idx], "BestPlane2": angl[1][best_idx],
        "BestMo":   angl[2][best_idx], "BestMw":   angl[3][best_idx],
        "BestDC":   angl[4][best_idx], "BestCLVD": angl[5][best_idx],
        "BestISO":  angl[6][best_idx],
        "BestP": angl[7][best_idx], "BestT": angl[8][best_idx],
        "BestB": angl[9][best_idx], "Tensor": angl[10][best_idx],
        "Singulars": singulars[best_idx], "Condition": condition[best_idx],
    }

    # CMT location from grid (same as calculateInversions2)
    _bidx = int(best_idx)
    _rows = _grid[_grid[:, 0] == _bidx + 1]
    if len(_rows):
        result["Xcmt"] = float(_rows[0, 1])
        result["Ycmt"] = float(_rows[0, 2])
        result["Zcmt"] = float(_rows[0, 3])
        _cmt_lat, _cmt_lon = cart2earth(
            config.org.latitude, config.org.longitude,
            result["Xcmt"], result["Ycmt"])
        result["CMT_lat"] = _cmt_lat
        result["CMT_lon"] = _cmt_lon

    # Update _BOOT_CTX
    prev_times = _BOOT_CTX.get("times", {})
    _BOOT_CTX.update({
        "vr":       vr,
        "shift":    shift,
        "best_idx": best_idx,
        "result":   result,
        "vr_mat":   vr_mat,
        "mech_mat": mech_mat,   # MT of every (source, shift) pair
        "corr":     corr,
        "angl":     angl,
        "times": {
            "load":   prev_times.get("load",   0.0),
            "elemse": prev_times.get("elemse", 0.0),
            "kernel": kernel_time,
            "total":  (prev_times.get("load", 0.0)
                       + prev_times.get("elemse", 0.0)
                       + kernel_time),
        },
    })

    # Back-fill vr_all in pre-inversion graph search dict
    if _GRAPH_SEARCH_ALAS is not None and _GRAPH_SEARCH_ALAS.get("vr_all") is None:
        _GRAPH_SEARCH_ALAS["vr_all"] = vr


# Sources whose system matrix R = G^T G has cond(R) above this are treated as
# singular. cond(R) = cond(G)^2; real inversions are far below it (ISOLA's
# reported cond(G) is typically < 100), while rank-deficient R reaches ~1e16.
_MAX_R_CONDITION = 1e12


def _check_inversion_inputs(G, Dobs, Dt):
    """
    Validate the inversion inputs before any kernel runs, so the CPU and GPU
    paths never see degenerate numbers (on which they behave differently).

    Raises ValueError if the observed data are unusable (NaN/inf or zero
    energy) or if no source is usable. Returns a boolean mask over sources:
    False for sources whose Green's functions are non-finite or whose system
    matrix R is singular / ill-conditioned.
    """
    if not np.isfinite(Dobs).all():
        raise ValueError('Observed data contain NaN/inf ({} samples) - check '
                         'gaps, merging and response removal'.format(
                             int((~np.isfinite(Dobs)).sum())))
    rr = float(np.dot(Dobs, Dobs) * Dt)
    if not rr > 0.0:
        raise ValueError('Observed data have zero energy (all used channels '
                         'are flat) - nothing to invert')

    n_sources = G.shape[0]
    valid = np.ones(n_sources, dtype=bool)
    reasons = np.full(n_sources, '', dtype=object)

    # chunked over sources on views of G, so no copy of the full matrix
    CHUNK = 256
    for s0 in range(0, n_sources, CHUNK):
        Gc = G[s0:s0 + CHUNK]
        finite = np.isfinite(Gc).all(axis=(1, 2))
        # R_k = G_k^T G_k (scale does not affect the condition number)
        R = np.matmul(Gc.transpose(0, 2, 1), Gc)
        R[~finite] = np.eye(R.shape[1])               # placeholder, rejected below
        eig = np.linalg.eigvalsh(R)                   # ascending, per source
        emin, emax = eig[:, 0], eig[:, -1]
        with np.errstate(divide='ignore', invalid='ignore'):
            cond = np.where(emin > 0.0, emax / emin, np.inf)
        ill = finite & ~(cond <= _MAX_R_CONDITION)

        idx = np.arange(s0, s0 + Gc.shape[0])
        valid[idx[~finite]] = False
        reasons[idx[~finite]] = 'non-finite Green\'s functions'
        valid[idx[ill]] = False
        reasons[idx[ill]] = 'singular / ill-conditioned system matrix'

    n_bad = int((~valid).sum())
    if n_bad == n_sources:
        raise ValueError('No usable source point: all {} sources rejected '
                         '(first reason: {})'.format(n_sources, reasons[0]))
    if n_bad:
        bad_ids = np.flatnonzero(~valid) + 1
        msg = ('{} of {} source points excluded from the inversion: '
               'point ids {}{}'.format(
                   n_bad, n_sources, bad_ids[:20].tolist(),
                   ' ...' if n_bad > 20 else ''))
        for why in sorted(set(reasons[~valid])):
            msg += ' | {}: {}'.format(why, int((reasons == why).sum()))
        config.logger.warning(msg)
        print('[WARN] ' + msg, flush=True)

    return valid


def getElemse(skip_createAllstat=False, hes_files=None, allstat_file=None, inpinv_file=None):
    """
    Build the Green's-function G matrix (elemse) and set a partial _BOOT_CTX
    so that getGraphSearch() can run before the
    full inversion kernel.

    Call order:
        getGreens()                 -- AXITRA Green's functions
        getElemse()                 -- this: createInpinv/createAllstat/createRaw + build G
        getGraphSearch()            -- optional, uses G from _BOOT_CTX
        getInversions()             -- detects G in _BOOT_CTX → runs kernel only
        getBootstrapping()
        exportResults()

    After this call _BOOT_CTX contains:
        G, Dobs, Dt, setup (with Timeshifts), components, grid,
        chunk_sizes, result={}, vr=None
    The result and vr fields are populated later by getInversions().

    When called from calculateInversions(), hes_files / allstat_file / inpinv_file
    are provided — file creation and directory scanning are skipped.
    config.outputdir must already be set to the combination folder by the caller.
    """
    global _BOOT_CTX

    _managed = hes_files is not None  # True when called from calculateInversions()

    if not _managed:
        config.logger.info('Creating Inversions configuration files (inpinv) '
                           'based on Time-shift rules')
        config.logger.info(config.dump(config.shiftRules))
        createInpinv()

        if not skip_createAllstat:
            config.logger.info('Select Stations for inversion (allstat)')
            createAllstat()

            config.logger.info('Creating raw data files for ISOLA parsing')
            createRaw()
        else:
            config.logger.info('Skipping createAllstat/createRaw (revise mode — using pre-written allstat)')

        # ---- resolve greens files -------------------------------------------
        def _chunk_no(fname):
            return int(os.path.basename(fname).split('.')[-2])

        greens_files = sorted(
            [os.path.join(config.greendir, f)
             for f in os.listdir(config.greendir) if f.endswith(".hes")],
            key=_chunk_no)
        greens = [os.path.basename(f) for f in greens_files]
        base   = greens[0]
        _, icrustal, igrdat, igrid, _, _ = base.split('.')

        allstat = sorted(os.listdir(config.allstatdir))[0]
        inpinv  = sorted(os.listdir(config.inpinvdir))[0]

        invdir = '{}.{}.{}.{}.{}'.format(
            allstat.split('.')[0][7:],
            inpinv.split('.')[0][6:],
            icrustal, igrdat, igrid)
        outdir = os.path.join(config.inversiondir, invdir)
        os.makedirs(outdir, exist_ok=True)

        allstat_path = os.path.join(config.allstatdir, allstat)
        inpinv_path  = os.path.join(config.inpinvdir,  inpinv)
    else:
        # caller (calculateInversions) has already set config.outputdir
        # and passes the exact files to use — no scanning or creation needed
        greens_files = hes_files
        greens = [os.path.basename(f) for f in greens_files]
        base   = greens[0]
        _, icrustal, igrdat, igrid, _, _ = base.split('.')

        allstat_path = allstat_file
        inpinv_path  = inpinv_file
        outdir       = config.outputdir

    job = {
        "allstat":      allstat_path,
        "inpinv":       inpinv_path,
        "greens_dir":   config.greendir,
        "greens_files": greens_files,
        "greens":       greens,
        "crustal":      os.path.join(config.crustaldir, f"crustal{icrustal}.dat"),
        "station":      os.path.join(config.workdir, "station.dat"),
        "grdat":        os.path.join(config.grdatdir,   f"grdat{igrdat}.hed"),
        "raw":          os.path.join(config.rawdir, igrdat),
        "outdir":       outdir,
    }

    setup = initialize_inversion(job, config)

    # ---- extract fields needed for G build ----------------------------------
    Dobs      = np.ascontiguousarray(setup["Dobs"], dtype=np.float64)
    removeIdx = setup["removeIdx"]
    nr        = setup["nr"]
    NFFT      = setup["NFFT"]
    Dt        = setup["Dt"]
    sos       = setup["sos"]
    nfreq     = setup["nfreq"]
    CCm       = setup["CCm"]

    keyinv     = config.get_keyinv(config.cfg, origin_data=config.org)
    components = 5 if keyinv == 2 else 6

    keep_mask  = np.ones(nr * 3 * NFFT, dtype=np.bool_)
    keep_mask[removeIdx] = False
    ch_keep    = keep_mask.reshape(nr * 3, NFFT)[:, 0]
    keep_pairs = [(int(idx // 3), int(idx % 3)) for idx in np.where(ch_keep)[0]]

    # ---- load Green's functions (parallel) ----------------------------------
    # each file is read straight into its slice of U_all (sizes from the file
    # sizes), so the Green's functions are copied once, not twice
    t_load = time.perf_counter()
    chunk_sizes   = [_hes_n_sources(fp, nfreq, nr) for fp in greens_files]
    total_sources = sum(chunk_sizes)
    U_all = np.empty((nfreq, 18, nr, total_sources), dtype=np.complex128)
    U_all5 = U_all.reshape(nfreq, 3, 6, nr, total_sources)
    starts = np.cumsum([0] + chunk_sizes)

    def _load_chunk(i):
        load_hes_into(greens_files[i], nfreq, nr,
                      U_all5[..., int(starts[i]):int(starts[i + 1])])

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.n_cores()) as executor:
        list(executor.map(_load_chunk, range(len(greens_files))))

    t_elemse = time.perf_counter()

    # ---- apply STF (if not delta) then build G matrix -----------------------
    U_all = apply_stf(U_all, nfreq, setup["tl"], setup["LOCAL_AW"])
    CC = CCm.ravel()
    G  = build_G_transposed(U_all, CC, sos, Dt, keep_pairs, components, NFFT)
    del U_all
    t_end = time.perf_counter()

    if Dobs.shape[0] != G.shape[1]:
        raise ValueError(
            'Dobs length {} does not match G rows {} (allstat flags and '
            'stream channels disagree)'.format(Dobs.shape[0], G.shape[1]))

    valid = _check_inversion_inputs(G, Dobs, Dt)

    # ---- load source grid (of this combination's grid rule) -----------------
    _grid = _read_grid(igrid)

    # ---- set partial _BOOT_CTX ----------------------------------------------
    _BOOT_CTX = {
        "G":           G,
        "Dobs":        Dobs,
        "Dt":          Dt,
        "setup":       setup,
        "components":  components,
        "grid":        _grid,
        "igrid":       igrid,
        "valid":       valid,    # per-source mask from _check_inversion_inputs()
        "chunk_sizes": chunk_sizes,
        "result":      {},       # populated by getInversions() / _run_kernel_only()
        "vr":          None,     # populated by _run_kernel_only()
        "times": {
            "elemse": t_end - t_load,
            "kernel": 0.0,
        },
    }

    config.logger.info(
        'getElemse done: elemse %.2f s | G shape %s',
        t_end - t_load, G.shape)


def calculateInversions(stage_t=None):
    """
    Run the full inversion pipeline for every combination of:
        (icrustal, igrdat, igrid) × allstat × inpinv

    For each combination:
      - sets config.outputdir to  inversiondir/{ai}.{ii}.{icrustal}.{igrdat}.{igrid}/
      - calls getElemse() → graph search (if enabled) → getInversions()
    Then, for the best combination only (highest VR at full precision):
      getBootstrapping() (if enabled) → doPlots() → renderSite(), and the best
      combination is promoted to <workdir>/output.

    All [GS] and [STAGE] prints that were previously in main live here.
    stage_t dict is updated with cumulative timing for each stage.
    """
    global _GRAPH_SEARCH_ALAS, _GS_SHARED, _EVT_BASE, _BOOT_CTX

    # every combination's QuakeML starts from the event as it is now (before
    # any solution is added), so combinations never see each other's solutions
    import copy as _copy
    _EVT_BASE = _copy.deepcopy(config.evt)

    # ── 1. Create all config files once ──────────────────────────────────────
    config.logger.info('calculateInversions: creating inpinv files from shift rules')
    config.logger.info(config.dump(config.shiftRules))
    createInpinv()

    config.logger.info('calculateInversions: creating allstat + raw files')
    createAllstat()
    createRaw()

    # ── 2. Enumerate allstat / inpinv / hes files ─────────────────────────────
    def _chunk_no(fname):
        return int(os.path.basename(fname).split('.')[-2])

    allstat_files = sorted(
        [os.path.join(config.allstatdir, f)
         for f in os.listdir(config.allstatdir)
         if f.startswith('allstat') and f.endswith('.dat')],
        key=lambda p: int(os.path.basename(p)[7:].split('.')[0]))

    inpinv_files = sorted(
        [os.path.join(config.inpinvdir, f)
         for f in os.listdir(config.inpinvdir)
         if f.startswith('inpinv') and f.endswith('.dat')],
        key=lambda p: int(os.path.basename(p)[6:].split('.')[0]))

    all_hes = sorted(
        [os.path.join(config.greendir, f)
         for f in os.listdir(config.greendir) if f.endswith('.hes')],
        key=_chunk_no)

    # group hes files by (icrustal, igrdat, igrid)
    from collections import defaultdict as _defaultdict
    hes_groups = _defaultdict(list)
    for fp in all_hes:
        parts = os.path.basename(fp).split('.')
        # gr.{icrustal}.{igrdat}.{igrid}.{isource}.hes
        key = (parts[1], parts[2], parts[3])
        hes_groups[key].append(fp)
    for key in hes_groups:
        hes_groups[key].sort(key=_chunk_no)

    groups = sorted(hes_groups.keys())

    total = len(groups) * len(allstat_files) * len(inpinv_files)
    print('[INFO] calculateInversions: {} combination(s)  '
          '[{} grid group(s) × {} allstat × {} inpinv]'.format(
          total, len(groups), len(allstat_files), len(inpinv_files)), flush=True)

    gs_enabled   = config.cfg.get('GraphSearch', {}).get('Enabled', True)
    boot_enabled = config.cfg.get('Bootstrap',   {}).get('Enabled', True)

    # ── 3. Main loop: invert every combination ─────────────────────────────────
    combo_idx = 0
    _best = None
    for (icrustal, igrdat, igrid) in groups:
        group_files = hes_groups[(icrustal, igrdat, igrid)]
        for ai, allstat_path in enumerate(allstat_files):
            for ii, inpinv_path in enumerate(inpinv_files):
                combo_idx += 1

                # ── output folder ─────────────────────────────────────────
                folder = '{}.{}.{}.{}.{}'.format(ai, ii, icrustal, igrdat, igrid)
                config.outputdir = os.path.join(config.inversiondir, folder)
                os.makedirs(config.outputdir, exist_ok=True)

                print('\n[INFO] --- Combination {}/{} : allstat={} inpinv={} '
                      'icrustal={} igrdat={} igrid={} ---'.format(
                      combo_idx, total, ai, ii, icrustal, igrdat, igrid), flush=True)

                _ct = {}  # per-combo timing

                # ── elemse ────────────────────────────────────────────────
                _t = time.perf_counter()
                getElemse(hes_files=group_files,
                          allstat_file=allstat_path,
                          inpinv_file=inpinv_path)
                _dt = time.perf_counter() - _t
                _ct['elementary seismogram calculation'] = _dt
                print('[STAGE] {:<38s}: {:7.2f} s'.format(
                    'elementary seismogram calculation', _dt), flush=True)
                if stage_t is not None:
                    stage_t['elementary seismogram calculation'] = \
                        stage_t.get('elementary seismogram calculation', 0.0) + _dt

                # ── graph search ──────────────────────────────────────────
                _GRAPH_SEARCH_ALAS = None
                _GS_SHARED = None
                if gs_enabled:
                    buildNeighborGraph()
                    if _GS_SHARED is not None:
                        print('[GS] Graph build      : {:7.0f} ms  ({} nodes)'.format(
                              _GS_SHARED['graph_build_time'] * 1000,
                              _GS_SHARED['Total_gridPoints']), flush=True)

                    getGraphSearch()
                    if _GRAPH_SEARCH_ALAS is not None:
                        _mn = _GRAPH_SEARCH_ALAS['max_node']
                        print('[GS] Graph Search     : {:7.0f} ms  '
                              '(x={:.1f}, y={:.1f}, z={:.1f})'.format(
                              _GRAPH_SEARCH_ALAS['gs_time'] * 1000,
                              _mn[0], _mn[1], _mn[2]), flush=True)

                # ── full inversion ────────────────────────────────────────
                getInversions()
                _ctx = _BOOT_CTX
                _inv_dt = (_ctx or {}).get('times', {}).get('kernel', 0.0)
                _ct['full inversion (kernel)'] = _inv_dt
                if stage_t is not None:
                    stage_t['full inversion (kernel)'] = \
                        stage_t.get('full inversion (kernel)', 0.0) + _inv_dt
                if _ctx is not None and 'Xcmt' in _ctx.get('result', {}):
                    _ri = _ctx['result']
                    print('[STAGE] {:<38s}: {:7.2f} s  (x={:.1f}, y={:.1f}, z={:.1f})'.format(
                        'full inversion (kernel)', _inv_dt,
                        _ri['Xcmt'], _ri['Ycmt'], _ri['Zcmt']), flush=True)
                else:
                    print('[STAGE] {:<38s}: {:7.2f} s'.format(
                        'full inversion (kernel)', _inv_dt), flush=True)

                # ── remember the best combination (full-precision VR) ─────
                # the bootstrap, the figures and the event page are made only
                # for the best combination, after all combinations are inverted
                _vr_c = float((_ctx or {}).get('result', {}).get('VR', -np.inf))
                _write_timing_summary(_ct)
                if _best is None or _vr_c > _best['vr']:
                    _best = {'vr': _vr_c, 'dir': config.outputdir, 'ctx': _ctx, 'ct': _ct,
                             'gs': {_n: globals()[_n] for _n in ['_GRAPH_SEARCH_ALAS', '_GS_SHARED']}}

    print('\n[INFO] calculateInversions: all {} combination(s) inverted.'.format(total), flush=True)

    # ── Best combination (highest VR, full precision; first one on a tie) ────
    _best_dir = None
    if _best is not None and np.isfinite(_best['vr']):
        _best_dir = _best['dir']
        config.outputdir = _best_dir
        config.best_combo = os.path.basename(_best_dir)   # $bestinvdir of Notification.Command
        _BOOT_CTX = _best['ctx']
        for _n, _v in _best['gs'].items():
            globals()[_n] = _v
        print('[INFO] Best combo: {}  (VR={:.2f})'.format(config.best_combo, _best['vr']),
              flush=True)
        _ct = _best['ct']

        # ── bootstrap (best combination only) ─────────────────────────────
        if boot_enabled:
            _t = time.perf_counter()
            getBootstrapping()
            _dt = time.perf_counter() - _t
            _ct['bootstrapping'] = _dt
            print('[STAGE] {:<38s}: {:7.2f} s'.format('bootstrapping', _dt), flush=True)
            if stage_t is not None:
                stage_t['bootstrapping'] = stage_t.get('bootstrapping', 0.0) + _dt

        # ── figures, event page, inversion parameters (best only) ─────────
        _combo_visualize(_ct)
        if stage_t is not None:
            stage_t['export results + plots'] = \
                stage_t.get('export results + plots', 0.0) + \
                _ct.get('export results + plots', 0.0)
        _write_timing_summary(_ct)
    else:
        print('[WARN] calculateInversions: no combination produced a solution', flush=True)

    # ── Write timing summary ──────────────────────────────────────────────────
    if stage_t is not None:
        _write_timing_summary(stage_t)

    # ── Promote best combo to workdir/output ─────────────────────────────────
    try:
        import shutil as _shutil
        _target = os.path.join(config.workdir, 'output')
        if _best_dir is not None and _best_dir != _target:
            os.makedirs(_target, exist_ok=True)
            _shutil.copytree(_best_dir, _target, dirs_exist_ok=True)
            config.logger.info('Best combo results promoted to %s', _target)
    except Exception as _e:
        config.logger.warning('Could not promote best combo results: %s', _e)

    # ── Catalog home page: once per event, now that output/event.xml exists ──
    _render_home()

    # ── Notifications (e-mail, command), as old Gisola ──────────────────────
    _notify()


# =========================================================================
# GRAPH SEARCH MODULE
# =========================================================================


@njit(fastmath=True, cache=True)
def compute_vr_single_numba(Gk, obs_data, Dt, shifts_array):
    """
    Compute best VR and corresponding shift for a SINGLE source
    (_point_inversion - the same routine as the full inversion).

    Parameters
    ----------
    Gk          : (n_total, n_comp)  Green's functions for one source
    obs_data    : (n_total,)         observed data vector
    Dt          : float              time-step (seconds)
    shifts_array: (n_shifts,) int64  time-shift indices to test

    Returns
    -------
    best_vr    : float   best variance reduction [0-100]
    best_shift : int64   index of the shift that produced best_vr
    """
    MULTIP   = 1e20
    n_comp   = Gk.shape[1]
    n_shifts = shifts_array.shape[0]
    R      = np.empty((n_comp, n_comp))
    vr_row = np.empty(n_shifts)
    me_row = np.empty((n_shifts, n_comp))
    _point_inversion(Gk, obs_data, shifts_array, Dt * MULTIP, MULTIP,
                     _data_power(obs_data, Dt), R, vr_row, me_row)
    best_vr    = -1.0
    best_shift = np.int64(0)
    for si in range(n_shifts):
        if vr_row[si] > best_vr:
            best_vr    = vr_row[si]
            best_shift = shifts_array[si]
    return best_vr, best_shift


@njit(fastmath=True, cache=True)
def compute_vr_single_with_mech(Gk, obs_data, Dt, shifts_array):
    """
    Like compute_vr_single_numba but also returns moment-tensor amplitudes,
    singular values, and condition number for the best shift.

    Returns
    -------
    best_vr    : float         best VR [0-100]
    best_shift : int64         shift index
    best_a_k   : (n_comp,)    MT amplitudes at best shift
    svals      : (n_comp,)    singular values (same convention as getInversions)
    cond_num   : float         condition number
    """
    MULTIP   = 1e20
    VARDAT   = 2.0e-12
    n_comp   = Gk.shape[1]
    n_shifts = shifts_array.shape[0]
    R      = np.empty((n_comp, n_comp))
    vr_row = np.empty(n_shifts)
    me_row = np.empty((n_shifts, n_comp))
    _point_inversion(Gk, obs_data, shifts_array, Dt * MULTIP, MULTIP,
                     _data_power(obs_data, Dt), R, vr_row, me_row)

    eigvals = np.linalg.eigvalsh(R)
    svals   = np.sqrt(np.abs(eigvals) / VARDAT) / 1e10
    mn = svals.min()
    mx = svals.max()
    cond_num = mx / mn if mn > 0.0 else 1e30

    best_vr    = -1.0
    best_shift = np.int64(0)
    best_a_k   = np.zeros(n_comp)
    for si in range(n_shifts):
        if vr_row[si] > best_vr:
            best_vr    = vr_row[si]
            best_shift = shifts_array[si]
            for n in range(n_comp):
                best_a_k[n] = me_row[si, n]

    return best_vr, best_shift, best_a_k, svals, cond_num


@njit(parallel=True, fastmath=True, cache=True)
def compute_vr_batch_numba(G, pids, obs_data, Dt, shifts_array):
    """
    Compute VR for a batch of sources in parallel, read directly from G by index
    (no copy of their Green's functions).

    Parameters
    ----------
    G           : (n_sources, n_total, n_comp)  Green's functions of all sources
    pids        : (n_batch,) int64              indices of the sources to evaluate
    obs_data    : (n_total,)                    observed data vector
    Dt          : float                         time-step (seconds)
    shifts_array: (n_shifts,) int64             time-shift indices to test

    Returns
    -------
    vr_out : (n_batch,) float64   best VR for each source
    """
    n_batch = pids.shape[0]
    vr_out  = np.zeros(n_batch)
    for j in prange(n_batch):
        vr_out[j], _ = compute_vr_single_numba(G[pids[j]], obs_data, Dt, shifts_array)
    return vr_out


def _gs_idx(pids):
    """Node indices for compute_vr_batch_numba."""
    return np.asarray(pids, dtype=np.int64)


def _gs_get_closest_existing_node(hypothetical_root, df):
    """Snap a hypothetical root point to the nearest existing grid node."""
    coords = df[['x', 'y', 'z']].to_numpy()
    tree = _KDTree(coords)
    _, index = tree.query(hypothetical_root)
    closest_node = tuple(coords[index])
    config.logger.debug("Hypothetical root %s snapped to nearest data point: %s",
                        hypothetical_root, closest_node)
    return closest_node


def _gs_build_neighbor_graph(df):
    """
    Build a 6-connected adjacency graph directly from the actual grid
    coordinates (axis-aligned, uniform or non-uniform).
    Built once before the search — O(N log N).
    """
    from collections import defaultdict

    coords = list(zip(df['x'], df['y'], df['z']))

    yz_groups = defaultdict(list)
    xz_groups = defaultdict(list)
    xy_groups = defaultdict(list)

    for x, y, z in coords:
        yz_groups[(round(y, 6), round(z, 6))].append(x)
        xz_groups[(round(x, 6), round(z, 6))].append(y)
        xy_groups[(round(x, 6), round(y, 6))].append(z)

    for g in (yz_groups, xz_groups, xy_groups):
        for k in g:
            g[k].sort()

    neighbor_graph = {}
    for x, y, z in coords:
        node = (x, y, z)
        nbrs = []

        xs = yz_groups[(round(y, 6), round(z, 6))]
        xi = xs.index(x)
        if xi > 0:            nbrs.append((xs[xi - 1], y, z))
        if xi < len(xs) - 1: nbrs.append((xs[xi + 1], y, z))

        ys = xz_groups[(round(x, 6), round(z, 6))]
        yi = ys.index(y)
        if yi > 0:            nbrs.append((x, ys[yi - 1], z))
        if yi < len(ys) - 1: nbrs.append((x, ys[yi + 1], z))

        zs = xy_groups[(round(x, 6), round(y, 6))]
        zi = zs.index(z)
        if zi > 0:            nbrs.append((x, y, zs[zi - 1]))
        if zi < len(zs) - 1: nbrs.append((x, y, zs[zi + 1]))

        neighbor_graph[node] = nbrs

    return neighbor_graph





def _gs_compute_solution(max_node, coord_to_pointid, G, Dobs, Dt, Timeshifts):
    """
    Compute the full seismological solution for the graph-search best node.

    Calls compute_vr_single_with_mech for the MT amplitudes, singular values,
    and condition number, then runs silsub to get fault planes, Mw, M0,
    DC/CLVD/ISO%, and P/T/B axes.  Also converts grid (x,y) to lat/lon and
    reads the catalog magnitude from config.evt.

    Returns a dict suitable for _gs_format_solution_dat.
    """
    pid = coord_to_pointid[max_node] - 1
    Gk  = np.ascontiguousarray(G[pid])
    Ts  = np.ascontiguousarray(Timeshifts)

    best_vr, best_shift, best_a_k, svals, cond_num = \
        compute_vr_single_with_mech(Gk, Dobs, Dt, Ts)

    corr = float(np.sqrt(max(best_vr, 0.0) / 100.0))

    mech = best_a_k[np.newaxis].copy()
    (plane1, plane2, M0_arr, Mw_arr, DC_arr, CLVD_arr, ISO_arr,
     P_arr, T_arr, B_arr, MT_arr) = silsub(mech)

    M_raw = MT_arr[0]  # raw (unscaled) 3×3 tensor in N·m

    lat, lon = None, None
    try:
        x, y = float(max_node[0]), float(max_node[1])
        lat, lon = cart2earth(config.org.latitude, config.org.longitude, x, y)
    except Exception:
        pass

    mag_str = None
    try:
        pm = config.evt.preferred_magnitude()
        if pm is None:
            mags = config.evt.magnitudes
            pm = mags[0] if mags else None
        if pm is not None:
            mag_str = f'{pm.magnitude_type}={pm.mag:.1f}'
    except Exception:
        pass

    return {
        'vr':       float(best_vr),
        'corr':     corr,
        'shift':    int(best_shift),
        'MT_ned':   np.array([M_raw[0, 0], M_raw[1, 1], M_raw[2, 2],
                              M_raw[0, 1], M_raw[0, 2], M_raw[1, 2]]),
        'plane1':   plane1[0],
        'plane2':   plane2[0],
        'M0':       float(M0_arr[0]),
        'Mw':       float(Mw_arr[0]),
        'DC':       float(DC_arr[0]),
        'CLVD':     float(CLVD_arr[0]),
        'ISO':      float(ISO_arr[0]),
        'P_axis':   P_arr[0],
        'T_axis':   T_arr[0],
        'B_axis':   B_arr[0],
        'svals':    svals,
        'cond_num': float(cond_num),
        'lat':      lat,
        'lon':      lon,
        'mag_str':  mag_str,
    }


def _gs_format_solution_dat(sol, max_node):
    """Format a _gs_compute_solution result dict as a list of .dat lines."""
    lines = ['']
    if sol['mag_str']:
        lines.append(f'Magnitude:   {sol["mag_str"]}')
    lines.append(f'VR: {sol["vr"]:.2f}')
    lines.append(f'Corr: {sol["corr"]:.4f}')
    lines.append(f'Best shift: {sol["shift"]}')
    mt = sol['MT_ned']
    lines.append('MT parameters (Mxx Myy Mzz Mxy Mxz Myz, NED, N·m):')
    lines.append(f'[{mt[0]:.4e} {mt[1]:.4e} {mt[2]:.4e} {mt[3]:.4e} {mt[4]:.4e} {mt[5]:.4e}]')
    p1 = sol['plane1']
    p2 = sol['plane2']
    lines.append(f'Plane 1: Strike = {int(round(p1[0]))}, '
                 f'Dip = {int(round(p1[1]))}, Rake = {int(round(p1[2]))}')
    lines.append(f'Plane 2: Strike = {int(round(p2[0]))}, '
                 f'Dip = {int(round(p2[1]))}, Rake = {int(round(p2[2]))}')
    lines.append(f'M0: {sol["M0"]:.4e}')
    lines.append(f'Mw: {sol["Mw"]:.1f}')
    lines.append(f'DC%: {int(round(sol["DC"]))}')
    lines.append(f'CLVD%: {int(round(sol["CLVD"]))}')
    lines.append(f'ISO: {int(round(sol["ISO"]))}')
    P = sol['P_axis']; T = sol['T_axis']; B = sol['B_axis']
    lines.append(f'P-Axis  Azimuth: {int(round(P[0]))}  Plunge: {int(round(P[1]))}')
    lines.append(f'T-Axis  Azimuth: {int(round(T[0]))}  Plunge: {int(round(T[1]))}')
    lines.append(f'B-Axis  Azimuth: {int(round(B[0]))}  Plunge: {int(round(B[1]))}')
    sv = sorted(sol['svals'])
    lines.append(f'Singulars: [{" ".join(f"{v:.4e}" for v in sv)}]')
    lines.append(f'Condition number: {sol["cond_num"]:.2f}')
    x, y, z = max_node
    lines.append(f'CMT grid position:  X={x:.2f} km  Y={y:.2f} km  Depth={z:.2f} km')
    if sol['lat'] is not None:
        lines.append(f'CMT location:       Lat={sol["lat"]:.4f}  Lon={sol["lon"]:.4f}')
    return lines


def buildNeighborGraph():
    """
    Build the shared neighbor graph and coordinate lookup used by all three
    graph-search method.  Call this once before getGraphSearch().

    Times the graph construction and stores the result in _GS_SHARED so that
    each search method reads the pre-built structures and times only its own
    climb algorithm.

    Populates IsolaPy._GS_SHARED.
    """
    global _GS_SHARED

    if _BOOT_CTX is None:
        config.logger.warning('buildNeighborGraph: no inversion context — skipping')
        return

    G    = _BOOT_CTX["G"]
    grid = _BOOT_CTX["grid"]

    Zhypo            = round(config.org.depth / 1000.0, 1)
    Total_gridPoints = G.shape[0]

    df = _pd.DataFrame({
        'point_id': grid[:Total_gridPoints, 0].astype(int),
        'x':        grid[:Total_gridPoints, 1],
        'y':        grid[:Total_gridPoints, 2],
        'z':        grid[:Total_gridPoints, 3],
    })
    df['x'] = df['x'].round(6)
    df['y'] = df['y'].round(6)
    df['z'] = df['z'].round(6)

    # rejected sources (see _check_inversion_inputs) are left out of the graph
    _valid = _BOOT_CTX.get("valid")
    if _valid is not None and not _valid.all():
        df = df[_valid[:Total_gridPoints]].reset_index(drop=True)

    coord_to_pointid = dict(
        zip(zip(df['x'], df['y'], df['z']), df['point_id'].astype(int))
    )

    Nx = df['x'].nunique()
    Ny = df['y'].nunique()
    Nz = df['z'].nunique()

    root = (0.0, 0.0, Zhypo)

    t0 = time.perf_counter()
    neighbor_graph = _gs_build_neighbor_graph(df)
    start_node     = _gs_get_closest_existing_node(root, df)

    # Pre-compute VR for the start node and its direct neighbours.
    # This loads those G-matrix rows into L3 cache so all three search methods
    # start from warm cache rather than cold RAM.
    G_ctx      = _BOOT_CTX["G"]
    Dobs_ctx   = _BOOT_CTX["Dobs"]
    Dt_ctx     = _BOOT_CTX["Dt"]
    Tshifts    = np.ascontiguousarray(_BOOT_CTX["setup"]["Timeshifts"], dtype=np.int64)

    pid_start    = coord_to_pointid[start_node] - 1
    Gk           = np.ascontiguousarray(G_ctx[pid_start])
    vr_start, _  = compute_vr_single_numba(Gk, Dobs_ctx, Dt_ctx, Tshifts)

    nbr_nodes  = [n for n in neighbor_graph.get(start_node, []) if n in coord_to_pointid]
    nbr_pids   = [coord_to_pointid[n] - 1 for n in nbr_nodes]
    vr_nbrs    = compute_vr_batch_numba(G_ctx, _gs_idx(nbr_pids), Dobs_ctx, Dt_ctx, Tshifts)

    vr_init = {start_node: vr_start}
    for n, vr in zip(nbr_nodes, vr_nbrs):
        vr_init[n] = vr

    graph_build_time = time.perf_counter() - t0

    _GS_SHARED = {
        'df':               df,
        'coord_to_pointid': coord_to_pointid,
        'neighbor_graph':   neighbor_graph,
        'start_node':       start_node,
        'root':             root,
        'Zhypo':            Zhypo,
        'Total_gridPoints': Total_gridPoints,
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
        'vr_init':          vr_init,
        'graph_build_time': graph_build_time,
    }

    config.logger.info('Neighbor graph + VR warmup done in %.3f s | start node %s | %d nodes',
                       graph_build_time, start_node, Total_gridPoints)


def _gs_climb_vr_seeded_adaptive_lookahead(start_node, coord_to_pointid, neighbor_graph,
                                            G, Dobs, Dt, Timeshifts, components, vr_init=None):
    """
    Seeded Adaptive Lookahead hill-climber.

    Runs one full LA-style step from start_node to seed the initial frontier
    with both 1-hop AND 2-hop uphill nodes (including 2-hop paths through
    downhill intermediates).  Then continues with the standard Adaptive
    Lookahead logic: Naive Multipath until the frontier collapses, then
    1-step lookahead, resuming multipath if new nodes are found.

    This fixes the case where the global maximum is only reachable via a
    diagonal 2-hop move from the very first node — the scenario where ALA
    and MP commit to the wrong local branch at step 0.

    Returns (path_to_max, all_visited, max_node, computed_vr).
    """
    computed_vr = {}
    parent      = {start_node: None}

    # Evaluate start node
    if vr_init and start_node in vr_init:
        computed_vr[start_node] = vr_init[start_node]
    else:
        pid = coord_to_pointid[start_node] - 1
        Gk  = np.ascontiguousarray(G[pid])
        vr_start, _ = compute_vr_single_numba(Gk, Dobs, Dt, Timeshifts)
        computed_vr[start_node] = vr_start

    vr_start = computed_vr[start_node]
    queued   = {start_node}

    # ── LA-style seed: evaluate 1-hop and 2-hop from start_node ─────────────
    hop1 = [n for n in neighbor_graph.get(start_node, [])
            if n in coord_to_pointid and n not in computed_vr]
    if hop1:
        pids     = [coord_to_pointid[n] - 1 for n in hop1]
        vr_batch = compute_vr_batch_numba(G, _gs_idx(pids), Dobs, Dt, Timeshifts)
        for n, vr_n in zip(hop1, vr_batch):
            computed_vr[n] = vr_n

    hop2 = list({la
                 for n in neighbor_graph.get(start_node, [])
                 if n in coord_to_pointid
                 for la in neighbor_graph.get(n, [])
                 if la in coord_to_pointid and la != start_node and la not in computed_vr})
    if hop2:
        pids     = [coord_to_pointid[n] - 1 for n in hop2]
        vr_batch = compute_vr_batch_numba(G, _gs_idx(pids), Dobs, Dt, Timeshifts)
        for n, vr_n in zip(hop2, vr_batch):
            computed_vr[n] = vr_n

    # Build initial frontier: uphill 1-hop + uphill 2-hop through any intermediate
    frontier = set()
    for n in neighbor_graph.get(start_node, []):
        if n not in coord_to_pointid:
            continue
        if computed_vr.get(n, -1) > vr_start and n not in queued:
            queued.add(n)
            parent[n] = start_node
            frontier.add(n)
    for n in neighbor_graph.get(start_node, []):
        if n not in coord_to_pointid:
            continue
        for la in neighbor_graph.get(n, []):
            if la not in coord_to_pointid or la == start_node:
                continue
            if computed_vr.get(la, -1) > vr_start and la not in queued:
                queued.add(la)
                parent[la] = start_node
                frontier.add(la)

    if not frontier:
        # start_node is already the local maximum
        return [start_node], list(computed_vr.keys()), start_node, computed_vr

    # ── ALA loop from enriched frontier ─────────────────────────────────────
    last_frontier = set(frontier)

    while frontier:
        last_frontier = set(frontier)

        # Multipath step: evaluate all unvisited 1-hop neighbours
        candidates = list({n for node in frontier
                           for n in neighbor_graph.get(node, [])
                           if n in coord_to_pointid and n not in computed_vr})
        if candidates:
            pids     = [coord_to_pointid[n] - 1 for n in candidates]
            vr_batch = compute_vr_batch_numba(G, _gs_idx(pids), Dobs, Dt, Timeshifts)
            for n, vr_n in zip(candidates, vr_batch):
                computed_vr[n] = vr_n

        next_frontier = set()
        for node in frontier:
            vr_node = computed_vr[node]
            for n in neighbor_graph.get(node, []):
                if n not in coord_to_pointid:
                    continue
                if n not in queued and computed_vr.get(n, -1) > vr_node:
                    queued.add(n)
                    parent[n] = node
                    next_frontier.add(n)

        if next_frontier:
            frontier = next_frontier
            continue

        # Stuck: 1-step lookahead on last_frontier
        la_candidates = list({la
                              for node in last_frontier
                              for n in neighbor_graph.get(node, [])
                              if n in coord_to_pointid
                                 and computed_vr.get(n, -1) <= computed_vr[node]
                              for la in neighbor_graph.get(n, [])
                              if la in coord_to_pointid
                                 and la != node and la not in computed_vr})
        if la_candidates:
            pids     = [coord_to_pointid[n] - 1 for n in la_candidates]
            vr_batch = compute_vr_batch_numba(G, _gs_idx(pids), Dobs, Dt, Timeshifts)
            for n, vr_n in zip(la_candidates, vr_batch):
                computed_vr[n] = vr_n

        la_frontier = set()
        for node in last_frontier:
            vr_node = computed_vr[node]
            for n in neighbor_graph.get(node, []):
                if n not in coord_to_pointid:
                    continue
                if computed_vr.get(n, -1) <= vr_node:
                    for la in neighbor_graph.get(n, []):
                        if la not in coord_to_pointid or la == node:
                            continue
                        if la not in queued and computed_vr.get(la, -1) > vr_node:
                            queued.add(la)
                            parent[la] = node
                            la_frontier.add(la)

        frontier = la_frontier  # empty → exits while loop

    max_node = max(computed_vr, key=lambda n: computed_vr[n])

    path = []
    node = max_node
    while node is not None:
        path.append(node)
        node = parent.get(node)
    path.reverse()

    return path, list(computed_vr.keys()), max_node, computed_vr


def getGraphSearch():
    """
    Seeded Adaptive Lookahead graph-search rapid CMT location.

    Runs one full LA-style seed step from the start node (evaluating both
    1-hop and 2-hop uphill neighbours, including 2-hop paths through downhill
    intermediates), then continues with Adaptive Lookahead (Naive Multipath
    until stuck, then 1-step lookahead).

    Fixes the case where the global maximum is only reachable via a diagonal
    2-hop move from the very first node.

    Writes GraphSearchAdaptiveLookaheadSeeded.dat to config.outputdir.
    Populates IsolaPy._GRAPH_SEARCH_ALAS.
    """
    global _GRAPH_SEARCH_ALAS

    if _BOOT_CTX is None:
        config.logger.warning('getGraphSearch: no inversion context — skipping')
        return
    if _GS_SHARED is None:
        config.logger.warning('getGraphSearch: neighbor graph not built — call buildNeighborGraph() first')
        return

    G          = _BOOT_CTX["G"]
    Dobs       = _BOOT_CTX["Dobs"]
    Dt         = _BOOT_CTX["Dt"]
    Timeshifts = np.ascontiguousarray(
                     _BOOT_CTX["setup"]["Timeshifts"], dtype=np.int64)
    components = _BOOT_CTX["components"]

    df               = _GS_SHARED['df']
    coord_to_pointid = _GS_SHARED['coord_to_pointid']
    neighbor_graph   = _GS_SHARED['neighbor_graph']
    start_node       = _GS_SHARED['start_node']
    root             = _GS_SHARED['root']
    Zhypo            = _GS_SHARED['Zhypo']
    Total_gridPoints = _GS_SHARED['Total_gridPoints']
    graph_build_time = _GS_SHARED['graph_build_time']
    vr_init          = _GS_SHARED['vr_init']

    startGS = time.perf_counter()
    path_nodes, all_visited, max_node, computed_vr = _gs_climb_vr_seeded_adaptive_lookahead(
        start_node, coord_to_pointid, neighbor_graph,
        G, Dobs, Dt, Timeshifts, components, vr_init=vr_init
    )
    try:
        _sol_alas = _gs_compute_solution(max_node, coord_to_pointid, G, Dobs, Dt, Timeshifts)
    except Exception as _se:
        _sol_alas = None
    endGS       = time.perf_counter()
    search_time = endGS - startGS
    total_time  = graph_build_time + search_time

    total_evaluated = len(all_visited)
    Gridpercentage  = (total_evaluated / Total_gridPoints) * 100.0

    result = _BOOT_CTX["result"]
    Xcmt   = float(result.get("Xcmt", 0.0))
    Ycmt   = float(result.get("Ycmt", 0.0))
    Zcmt   = float(result.get("Zcmt", Zhypo))
    vr_all = _BOOT_CTX["vr"]

    _dat_lines = []
    _dat_lines.append(f"Graph Build Time:                       {graph_build_time:.6f}  sec")
    _dat_lines.append(f"Seeded Adaptive Lookahead Search Time:  {search_time:.6f}  sec")
    _dat_lines.append(f"Total Time (build + search):            {total_time:.6f}  sec")
    _dat_lines.append(f"\nFinal Path length: {len(path_nodes)} nodes")
    _dat_lines.append(f"Total Unique Nodes Evaluated: {total_evaluated} nodes")
    _dat_lines.append(f"Grid Nodes: {Total_gridPoints} nodes")
    _dat_lines.append(f"Grid Cube Visited: {Gridpercentage:.4f}%")
    _dat_lines.append("\nFinal Path (xyz, VR):")
    for node in path_nodes:
        _dat_lines.append(f"xyz={node}, VR={computed_vr[node]:.6f}")
    _dat_lines.append(f"\nFinal Maximum Node: xyz={max_node}, VR={computed_vr[max_node]:.6f}")

    if _sol_alas is not None:
        _dat_lines.extend(_gs_format_solution_dat(_sol_alas, max_node))
    else:
        _dat_lines.append('\n[WARN] Full solution computation failed')

    try:
        _gs_dat_path = os.path.join(config.outputdir, _rv('GraphSearchAdaptiveLookaheadSeeded.dat'))
        with open(_gs_dat_path, 'w') as _f:
            _f.write('\n'.join(_dat_lines) + '\n')
        config.logger.info('Seeded adaptive lookahead results saved to %s', _gs_dat_path)
    except Exception as _e:
        print(f'[WARN] Could not save GraphSearchAdaptiveLookaheadSeeded.dat: {_e}', flush=True)

    _GRAPH_SEARCH_ALAS = {
        'gs_time':          search_time,
        'graph_build_time': graph_build_time,
        'total_time':       total_time,
        'path_nodes':       path_nodes,
        'all_visited':      all_visited,
        'max_node':         max_node,
        'computed_vr':      computed_vr,
        'total_evaluated':  total_evaluated,
        'Total_gridPoints': Total_gridPoints,
        'Gridpercentage':   Gridpercentage,
        'root':             root,
        'df':               df,
        'vr_all':           vr_all,
        'Zhypo':            Zhypo,
        'Xcmt':             Xcmt,
        'Ycmt':             Ycmt,
        'Zcmt':             Zcmt,
        'solution':         _sol_alas,
    }

    config.logger.info(
        'Seeded adaptive lookahead done in %.3f s (climb) + %.3f s (build) | path %d nodes | evaluated %d / %d (%.4f%%)',
        search_time, graph_build_time, len(path_nodes),
        total_evaluated, Total_gridPoints, Gridpercentage)


