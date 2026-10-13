#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precompile.py — Compile and cache all Gisola-BaBoo Numba kernels.

Run once after installation (or after updating IsolaPy.py):

    cd src
    python precompile.py

CPU kernels are compiled unconditionally.
GPU kernels are compiled only if the GPU works (the test of check_gpu.py).
All compiled code is written to the Numba cache dir and reused
on every subsequent run without recompilation.
"""

# OpenMP worker threads (Numba's parallel loops) sleep between parallel calls instead of
# spinning: removes a fixed ~10 ms cost of every parallel call (graph-search batches, silsub)
# without changing any result; long calls (inversion, bootstrap) are unaffected. Must be set
# before the OpenMP runtime is loaded; an OMP_WAIT_POLICY set by the user is kept.
import os as _os
_os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')

import sys
import time
import warnings
import numpy as np

# the small test arrays used here make Numba warn about low GPU occupancy: not useful here
try:
    from numba.core.errors import NumbaPerformanceWarning
    warnings.simplefilter('ignore', NumbaPerformanceWarning)
except Exception:
    pass

warnings.filterwarnings('ignore', category=FutureWarning)       # numba-cuda: Python 3.9 notice
import IsolaPy as isola

rng       = np.random.default_rng(0)
NSTATIONS = 2
WIN       = 1024
N_SOURCES = 4

G        = rng.random((N_SOURCES, WIN * NSTATIONS, 5), dtype=np.float64)
obs      = rng.random(WIN * NSTATIONS,                  dtype=np.float64)
Wc       = rng.dirichlet(np.ones(NSTATIONS), size=2)       # bootstrap: one weight per trace
shifts   = np.array([0], dtype=np.int64)
Coords   = rng.random((N_SOURCES, 3),                   dtype=np.float64)
n_keep   = max(1, int(0.2 * N_SOURCES))

N_STEPS = 7
_state = {}

def step(number, name, func):
    """Run one compile step and print its name and time on one line."""
    t0 = time.time()
    func()
    print('  [{}/{}] {:<42s} {:6.2f} s'.format(number, N_STEPS, name, time.time() - t0), flush=True)

print('Precompiling the Numba kernels (one time only)', flush=True)
t_total = time.time()

# ── 1. Inversion kernel ───────────────────────────────────────────────────────
def inversion():
    _state['res'] = isola.compute_inversion_numbaCpu(G, obs, 0.1, shifts, NSTATIONS)
step(1, 'CPU inversion kernel', inversion)

# ── 2. Focal mechanism decomposition ─────────────────────────────────────────
# Also compiles: build_moment_tensor, normalize_moment_tensor,
# eigendecomp_sorted, decompose, angles, cart2trendplunge
def silsub():
    mech = _state['res'][1]
    isola.silsub(mech)
step(2, 'CPU focal mechanism (silsub)', silsub)

# ── 3. Bootstrap kernels ──────────────────────────────────────────────────────
# bootstrap precompute + combine (CPU)
step(3, 'CPU bootstrap kernels',
     lambda: isola._bootstrap_members(G, obs, Wc, 0.1, shifts, Coords, n_keep, use_gpu=False))

# ── 4. Graph search kernels ───────────────────────────────────────────────────
# compute_vr_batch_numba transitively compiles compute_vr_single_numba
step(4, 'CPU graph search, batch kernel',
     lambda: isola.compute_vr_batch_numba(G, np.arange(N_SOURCES, dtype=np.int64), obs, 0.1, shifts))

# compute_vr_single_with_mech (full solution for best node)
step(5, 'CPU graph search, full-solution kernel',
     lambda: isola.compute_vr_single_with_mech(G[0], obs, 0.1, shifts))

# clipping check (modules/clip.py) and mouse check (modules/mouse2.py), preprocessing
def clip_mouse():
    import modules.clip as _clip
    _clip._mark_clipped(np.zeros(16), np.zeros(15), np.zeros(15), 0.98, -1.0, 1.0)
    import modules.mouse2 as _mouse2
    _mouse2.warmup()
step(6, 'CPU clipping and mouse checks', clip_mouse)

# ── 5. GPU kernels (conditional) ──────────────────────────────────────────────
# The GPU test runs in a separate process: if it crashes (for example with a driver that
# this Numba version does not support) the pre-compilation goes on without the GPU kernels.
import check_gpu
_gpu_available, _gpu_message = check_gpu.run_check()

if _gpu_available:
    def gpu():
        isola.compute_inversion_numbaGpu(G, obs, 0.1, shifts, NSTATIONS)
        isola._bootstrap_members(G, obs, Wc, 0.1, shifts, Coords, n_keep, use_gpu=True)
    step(7, 'GPU kernels', gpu)
else:
    print('  [{}/{}] {:<42s} skipped ({})'.format(N_STEPS, N_STEPS, 'GPU kernels', _gpu_message), flush=True)

print('Done in {:.2f} s. The kernels are cached and ready to run.'.format(time.time() - t_total), flush=True)
