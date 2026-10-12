#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mouse.py — Real-record fling-step (mouse) characterisation and detection.

Port of MATLAB my_realmouse_v02.m (Jiří Zahradník, summer 2026).
The previous single-type algorithm (Jiří Vackář, 2015) is preserved in mouse2.py.

──────────────────────────────────────────────────────────────────────────────
ALGORITHM OVERVIEW
──────────────────────────────────────────────────────────────────────────────

A seismic "mouse" (also called fling step) is a long-period artefact in a
broadband seismometer record caused by a permanent ground displacement or
velocity offset at the sensor.  Two physical origins produce two waveform
shapes:

  Mouse I  — acceleration step (e.g. a sudden velocity ramp):
             the instrument sees a linear velocity ramp which integrates to
             a quadratic displacement trend.  After double-integrating the
             instrument's count record, Mouse I appears as the double-integral
             of the seismometer's unit-step response.

  Mouse II — velocity step (e.g. a sudden displacement offset):
             single-integral of the seismometer's unit-step response.

Both types are built analytically from the sensor's pole–zero description
(no scipy needed — the step response is computed via partial-fraction
decomposition of H(s)/s).  The two templates are then fitted jointly to
the integrated displacement record using ordinary least squares, with an
independent time shift for each type.  A 2-D grid search over (shift₁, shift₂)
is fully vectorised with pre-computed dot-product matrices.

──────────────────────────────────────────────────────────────────────────────
PUBLIC API
──────────────────────────────────────────────────────────────────────────────

    check(stream, t_start_origin, t_min, t_max, inventory=None,
          misfit_threshold=0.50, amplitude_threshold=0.20)
        Pipeline entry point — same signature as mouse2.check().
        Returns a list of trace IDs that are flagged as mouse-contaminated.

    characterize(y, zeros, poles, gain, count2ground, instTcorner,
                 dt, endtime, pwavetime, swavetime, timlim=20.0)
        Low-level single-array characterisation; returns a structured dict.

──────────────────────────────────────────────────────────────────────────────
REFERENCES
──────────────────────────────────────────────────────────────────────────────
Zahradník, J. (2026). my_realmouse_v02.m — script for real-record mouse
    identification. Unpublished MATLAB code, Charles University Prague.
Vackář, J. (2015). MouseTrap v0.2.1.
    http://geo.mff.cuni.cz/~vackar/mouse/
"""

import os
import sys
import numpy as np

# sys.path is already set by the caller in the Gisola pipeline.
# We only need config for logging inside the pipeline check() entry point.
try:
    import config
    _HAS_CONFIG = True
except ImportError:
    _HAS_CONFIG = False

# scipy is preferred for the step response (identical to MATLAB's step()).
# If it is not installed the module falls back to the built-in pure-NumPy
# partial-fraction implementation which gives the same numerical result for
# systems with simple (non-repeated) poles.
try:
    from scipy.signal import ZerosPolesGain as _ZPK
    from scipy.signal import step as _scipy_step
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ══════════════════════════════════════════════════════════════════════════════
# STEP-RESPONSE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _step_response_scipy(zeros, poles, gain, t):
    """scipy path — wraps scipy.signal.step for a ZPK continuous-time system."""
    system = _ZPK(zeros, poles, gain)
    _, y = _scipy_step(system, T=np.asarray(t, dtype=float))
    return y


def _step_response_pfrac(zeros, poles, gain, t):
    """
    Pure-NumPy analytical step response via partial-fraction decomposition.

    For H(s) = gain·∏(s−zᵢ)/∏(s−pₖ) the step response y(t) is the inverse
    Laplace transform of H(s)/s:

        H(s)/s  = R₀/s  +  ∑ₖ Rₖ/(s−pₖ)

    where:
        R₀  = H(0)   (DC gain; 0 for the typical seismometer where H(0)=0)
        Rₖ  = [gain · ∏(pₖ−zᵢ) / pₖ] / ∏_{j≠k}(pₖ−pⱼ)

    Time-domain:
        y(t) = R₀ + ∑ₖ Rₖ · exp(pₖ · t)

    Valid for systems with simple (non-repeated) poles.  The Guralp 3T and
    most broadband seismometers have simple poles, so this is exact.
    """
    z = np.asarray(zeros, dtype=complex)
    p = np.asarray(poles, dtype=complex)
    t = np.asarray(t, dtype=float)
    n = len(p)

    # ── DC residue (pole at s = 0 only if H(0) ≠ 0) ─────────────────────────
    num0 = gain * (np.prod(-z) if len(z) else 1.0)
    den0 = np.prod(-p)
    R0 = (num0 / den0) if abs(num0) > 1e-40 else 0.0

    # ── Residues at each finite pole ─────────────────────────────────────────
    # Rₖ = [gain·∏(pₖ−zᵢ) / pₖ] / ∏_{j≠k}(pₖ−pⱼ)
    R = np.zeros(n, dtype=complex)
    for k in range(n):
        pk = p[k]
        if abs(pk) < 1e-30:
            # pole exactly at origin: residue from the s=0 term above
            continue
        num_k  = gain * np.prod(pk - z) / pk
        den_k  = np.prod([pk - p[j] for j in range(n) if j != k])
        R[k]   = num_k / den_k if abs(den_k) > 1e-60 else 0.0

    # ── Synthesise y(t) = R₀ + ∑ₖ Rₖ exp(pₖ t) ─────────────────────────────
    y = np.full(len(t), complex(R0), dtype=complex)
    for k in range(n):
        if abs(R[k]) > 0:
            y += R[k] * np.exp(p[k] * t)
    return np.real(y)


def _step_response(zeros, poles, gain, t):
    """Compute step response; use scipy when available, fall back to pfrac."""
    if _HAS_SCIPY:
        return _step_response_scipy(zeros, poles, gain, t)
    return _step_response_pfrac(zeros, poles, gain, t)


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC MICE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _build_synthetic_mice(zeros, poles, gain, dt, endtime, swavetime):
    """
    Construct normalised synthetic Mouse I and Mouse II templates.

    Both templates are zero before *swavetime* and contain the relevant
    integral of the seismometer's unit-step response from *swavetime* onward.

    Parameters
    ----------
    zeros, poles : array-like of complex
        Poles and zeros of the seismometer's velocity transfer function.
    gain : float
        A₀ normalisation constant (pole-zero gain).
    dt : float
        Sample interval in seconds.
    endtime : float
        Total record length in seconds (= signal duration).
    swavetime : float
        S-wave arrival time in seconds from trace start.  The synthetic step
        is placed at this time (zero before, step response after).

    Returns
    -------
    mouse1 : 1-D ndarray  Mouse I  (double-integrated step response), normalised.
    mouse2 : 1-D ndarray  Mouse II (single-integrated step response), normalised.
    Both have length  round(endtime/dt) + 1  (same as the signal array).
    """
    # ── Step response for the interval [swavetime, endtime] ──────────────────
    trgh   = round((endtime - swavetime) / dt) * dt   # length of step-response segment
    t_step = np.arange(0.0, trgh + dt, dt)            # time axis for step response
    y      = _step_response(zeros, poles, gain, t_step)

    # Normalise by the peak of the raw step response (MATLAB: y = y / max(y)).
    peak_y = y.max()
    if peak_y > 1e-30:
        y /= peak_y

    # ── Prepend zeros for [0, swavetime) ─────────────────────────────────────
    numzeros = round(swavetime / dt)
    n_total  = round(endtime   / dt) + 1      # total samples in the full record
    y_full   = np.concatenate([np.zeros(numzeros), y])[:n_total]

    # ── Mouse II: integrate once and normalise ────────────────────────────────
    # Mouse II represents a velocity step: single integral of the step response.
    mouse2 = np.cumsum(y_full) * dt
    peak2  = mouse2.max()
    if peak2 > 1e-30:
        mouse2 /= peak2

    # ── Mouse I: integrate again and normalise ────────────────────────────────
    # Mouse I represents an acceleration step (velocity ramp): double integral.
    mouse1 = np.cumsum(mouse2) * dt
    peak1  = mouse1.max()
    if peak1 > 1e-30:
        mouse1 /= peak1

    return mouse1, mouse2


# ══════════════════════════════════════════════════════════════════════════════
# VECTORISED 2-D GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def _build_shift_matrix(template, loopvals, numpo, taperlen):
    """
    Build a matrix whose rows are time-shifted and taper-trimmed copies of
    *template*.

    Row k corresponds to the integer shift loopvals[k] (positive = shift right,
    i.e. the template arrives later).  The trim removes the first taperlen−1
    and last taperlen samples to match MATLAB's ``taperlen:numpo-taperlen``
    (1-based, inclusive).

    Parameters
    ----------
    template : 1-D array of length numpo
    loopvals : integer array of shift values
    numpo    : int  (= len(template) = len(signal))
    taperlen : int  (= round(0.05 * numpo))

    Returns
    -------
    out : ndarray, shape (len(loopvals), numpo − 2·taperlen + 1)
    """
    # MATLAB slice taperlen:numpo-taperlen (1-based inclusive)
    # → Python slice [taperlen-1 : numpo-taperlen] (0-based, exclusive end)
    trim_lo = taperlen - 1
    trim_hi = numpo - taperlen
    n_samp  = trim_hi - trim_lo                 # = numpo − 2·taperlen + 1

    out = np.zeros((len(loopvals), n_samp), dtype=np.float64)

    for k, s in enumerate(loopvals):
        # Position i in output receives template[i − s].
        # Valid when both i ∈ [0, numpo) and i−s ∈ [0, numpo).
        i_lo = max(0, s)
        i_hi = min(numpo - 1, numpo - 1 + s) + 1   # exclusive
        if i_lo < i_hi:
            tmp       = np.zeros(numpo, dtype=np.float64)
            tmp[i_lo:i_hi] = template[i_lo - s:i_hi - s]
            out[k]    = tmp[trim_lo:trim_hi]

    return out


def _vectorised_2d_search(D1, D2, d):
    """
    Analytically solve the 2-parameter LS system for every (shift₁, shift₂)
    pair without any Python-level loop.

    For pair (i, j):
        GG = [D1[i], D2[j]],  shape (m, 2)
        solve  GG · [c1, c2]ᵀ ≈ d   in the LS sense.

    The normal equations are:
        ⎡ A11[i]    A12[i,j] ⎤  ⎡c1⎤   ⎡b1[i]⎤
        ⎣ A12[i,j]  A22[j]   ⎦  ⎣c2⎦ = ⎣b2[j]⎦

    with:
        A11 = D1·D1ᵀ (row dot-products), shape (n1,)
        A22 = D2·D2ᵀ,                    shape (n2,)
        A12 = D1·D2ᵀ  (cross dot-product matrix), shape (n1, n2)  ← BLAS
        b1  = D1·d,                       shape (n1,)
        b2  = D2·d,                       shape (n2,)

    Cramer's rule gives c1, c2 analytically as (n1, n2) arrays.

    Misfit = ‖GG·cof − d‖ / ‖d‖ is computed from the same dot products,
    avoiding re-synthesis of the residual.

    Parameters
    ----------
    D1 : (n1, m) array — shifted Mouse I templates
    D2 : (n2, m) array — shifted Mouse II templates
    d  : (m,) array — trimmed signal (integrated displacement)

    Returns
    -------
    c1, c2, misfit : each (n1, n2) array
    """
    d_norm2 = float(d @ d)

    # Pre-compute all necessary dot products in one pass.
    A11 = (D1 * D1).sum(axis=1)   # (n1,)
    A22 = (D2 * D2).sum(axis=1)   # (n2,)
    A12 = D1 @ D2.T                # (n1, n2) — heavy but done once in BLAS
    b1  = D1 @ d                   # (n1,)
    b2  = D2 @ d                   # (n2,)

    # Determinant of the 2×2 normal-equation matrix.
    det  = A11[:, None] * A22[None, :] - A12 ** 2
    safe = np.where(np.abs(det) > 1e-30, det, 1e-30)   # avoid /0

    # Cramer's rule — all operations broadcast to (n1, n2).
    c1 = (A22[None, :] * b1[:, None] - A12 * b2[None, :]) / safe
    c2 = (A11[:, None] * b2[None, :] - A12 * b1[:, None]) / safe

    # Residual energy ‖d − c1·d1 − c2·d2‖²  via the quadratic form.
    # (Expanding the squared norm and substituting the dot products.)
    res2 = (d_norm2
            + c1 ** 2 * A11[:, None]
            + c2 ** 2 * A22[None, :]
            + 2.0 * c1 * c2 * A12
            - 2.0 * c1 * b1[:, None]
            - 2.0 * c2 * b2[None, :])

    misfit = np.sqrt(np.maximum(res2, 0.0) / max(d_norm2, 1e-30))
    return c1, c2, misfit

# ══════════════════════════════════════════════════════════════════════════════
# PER-COMPONENT INVERSION
# ══════════════════════════════════════════════════════════════════════════════

def _invert_component(comp_data, mouse1, mouse2, dt,
                       pwavetime, endtime, timlim, perctaper=0.05,
                       _grid_density=50, _dtype=np.float64):
    """
    Fit Mouse I and Mouse II to a single integrated displacement component.

    Steps (mirrors MATLAB my_realmouse_v02.m exactly):
      1. Demean using the pre-P noise window  [0, 0.9*pwavetime).
      2. Integrate: counts -> raw displacement (cumulative sum x dt).
      3. Normalise to unit peak absolute value; save valmax scaling factor.
      4. Trim 5 % taper from each end of the normalised signal.
      5. Build shifted matrices for both mice (same trim).
      6. Vectorised 2-D LS search over all (shift1, shift2) combinations.
      7. Report best solution (minimum misfit).

    Parameters
    ----------
    _grid_density : int, default 50
        Shift grid is built with step = floor(isetlim / _grid_density), giving
        approximately 2*_grid_density+1 candidate shifts per axis.
        50 reproduces the MATLAB algorithm exactly (101 shifts, ~0.4 s spacing).
        25 halves the grid to 51 shifts, giving ~4× speedup with negligible
        loss of detection accuracy — used by the pipeline check().
    _dtype : numpy dtype, default np.float64
        Floating-point type for shift matrices and the BLAS multiply.
        np.float32 gives an additional ~2× BLAS speedup and is sufficient
        for binary detection (misfit threshold 0.5) — used by check().

    Returns dict: shift1, shift2, coef1, coef2, misfit, iter_no, peak
    """
    # 1. Demean using 90% of P-wave window (pre-noise)
    L_noise = round((pwavetime - 0.1 * pwavetime) / dt)
    y = comp_data.astype(float) - comp_data[:L_noise].mean()

    # 2. Integrate counts to raw displacement
    y_int = np.cumsum(y) * dt

    # 3. Normalise to peak absolute value
    peak = np.abs(y_int).max()
    if peak < 1e-30:
        return {'shift1': 0, 'shift2': 0, 'coef1': 0.0, 'coef2': 0.0,
                'misfit': 1.0, 'iter_no': 1, 'peak': peak}
    y_int /= peak

    # 4. Trim taper ends
    # MATLAB: d = dataobs(taperlen : numpo-taperlen)  [1-based, inclusive]
    # Python:  d = y_int[taperlen-1 : numpo-taperlen]
    numpo    = len(y_int)
    taperlen = round(perctaper * numpo)
    d        = y_int[taperlen - 1:numpo - taperlen]

    # 5. Build shift grid: -isetlim to +isetlim with step floor(isetlim/_grid_density)
    isetlim  = int(np.ceil(timlim / dt))
    loopstep = max(1, int(np.floor(isetlim / _grid_density)))
    loopvals = np.arange(-isetlim, isetlim + 1, loopstep, dtype=int)

    # 6. Build shifted-template matrices (cast to requested dtype for BLAS)
    D1 = _build_shift_matrix(mouse1, loopvals, numpo, taperlen).astype(_dtype)
    D2 = _build_shift_matrix(mouse2, loopvals, numpo, taperlen).astype(_dtype)
    d  = d.astype(_dtype)

    # 7. Vectorised 2-D search: find (shift1, shift2) that minimises misfit
    c1_grid, c2_grid, misfit_grid = _vectorised_2d_search(D1, D2, d)

    flat_idx   = int(np.argmin(misfit_grid))
    idx1, idx2 = np.unravel_index(flat_idx, misfit_grid.shape)

    # iter_no: 1-based counter matching MATLAB's variable j (outer-loop-major)
    n2      = len(loopvals)
    iter_no = int(idx1 * n2 + idx2 + 1)

    return {
        'shift1':   int(loopvals[idx1]),
        'shift2':   int(loopvals[idx2]),
        'coef1':    float(c1_grid[idx1, idx2]),
        'coef2':    float(c2_grid[idx1, idx2]),
        'misfit':   float(misfit_grid[idx1, idx2]),
        'iter_no':  iter_no,
        'peak':     peak,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: LOW-LEVEL CHARACTERISATION
# ══════════════════════════════════════════════════════════════════════════════

def characterize(y, zeros, poles, gain, count2ground, instTcorner,
                 dt, endtime, pwavetime, swavetime, timlim=20.0,
                 _grid_density=50, _dtype=np.float64):
    """
    Fit Mouse I and Mouse II to a single component array.

    Mirrors MATLAB my_realmouse_v02.m for one component.

    Parameters
    ----------
    y            : 1-D array, raw count record (velocity seismogram)
    zeros, poles : array-like of complex, seismometer transfer function
    gain         : float, A0 normalisation constant
    count2ground : float, physical unit per count (e.g. 1e-9 m/s per count)
    instTcorner  : float, long-period instrument corner in seconds
    dt           : float, sample interval (s)
    endtime      : float, record length (s); len(y) should be round(endtime/dt)+1
    pwavetime    : float, P-wave arrival time (s from trace start)
    swavetime    : float, S-wave arrival time (s); synthetic mice start here
    timlim       : float, half-width of shift search window (s); default 20

    Returns
    -------
    dict with keys: shift1, shift2, coef1, coef2, truemax1, truemax2,
                    misfit, iter_no, messages
    """
    zeros  = np.asarray(zeros, dtype=complex)
    poles  = np.asarray(poles, dtype=complex)

    mouse1, mouse2 = _build_synthetic_mice(zeros, poles, gain, dt, endtime, swavetime)

    n_sig  = round(endtime / dt) + 1
    y_seg  = y[:n_sig]

    res = _invert_component(y_seg, mouse1, mouse2, dt, pwavetime, endtime, timlim,
                            _grid_density=_grid_density, _dtype=_dtype)

    # Physical unit conversion:
    # peak x count2ground x 4*pi^2 / instTcorner^2  ->  m/s^2 (acceleration)
    valmax = res['peak'] * count2ground * 4.0 * np.pi**2 * (1.0 / instTcorner)**2
    true1  = res['coef1'] * valmax
    true2  = res['coef2'] * valmax

    isetlim  = int(np.ceil(timlim / dt))
    messages = []

    if abs(res['coef1']) >= 3.0 * abs(res['coef2']):
        messages.append('Mouse I is dominating (> 3-times) over Mouse II')
    elif abs(res['coef2']) >= 3.0 * abs(res['coef1']):
        messages.append('Mouse II is dominating (> 3-times) over Mouse I')

    if abs(res['shift1']) >= isetlim:
        messages.append(
            'WARNING: shift of Mouse I reached limit, check timlim or swavetime')
    if abs(res['shift2']) >= isetlim:
        messages.append(
            'WARNING: shift of Mouse II reached limit, check timlim or swavetime')

    if abs(res['coef1']) <= 0.2 and abs(res['coef2']) <= 0.2:
        messages.append(
            'No large mice found on this component; amplitude < 0.2 -- inversion may be wrong')
    if res['misfit'] > 0.5:
        messages.append(
            'WARNING: Misfit > 0.5; the fit is poor and results may be unreliable')

    return {
        'shift1':   res['shift1'],
        'shift2':   res['shift2'],
        'coef1':    res['coef1'],
        'coef2':    res['coef2'],
        'truemax1': true1,
        'truemax2': true2,
        'misfit':   res['misfit'],
        'iter_no':  res['iter_no'],
        'messages': messages,
    }


# ══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT PARAMETER EXTRACTION (from ObsPy response)
# ══════════════════════════════════════════════════════════════════════════════

def _paz_from_response(response):
    """
    Extract (zeros, poles, gain, count2ground, instTcorner) from an ObsPy
    instrument response object.
    """
    paz          = response.get_paz()
    sensitivity  = response.instrument_sensitivity.value   # counts per m/s

    zeros        = np.asarray(paz.zeros, dtype=complex)
    poles        = np.asarray(paz.poles, dtype=complex)
    gain         = paz.normalization_factor * paz.stage_gain
    count2ground = 1.0 / sensitivity

    # Long-period corner: period of the slowest (smallest magnitude) pole
    abs_p = np.abs(poles)
    abs_p = abs_p[abs_p > 1e-6]
    instTcorner  = (2.0 * np.pi / abs_p.min()) if len(abs_p) else 120.0

    return zeros, poles, gain, count2ground, instTcorner


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: PIPELINE-COMPATIBLE CHECK FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def check(stream, t_start_origin=10, t_min=0, t_max=120, inventory=None,
          misfit_threshold=0.50, amplitude_threshold=0.20):
    """
    Pipeline entry point -- same call signature as mouse2.check().

    A trace is flagged when:
        misfit < misfit_threshold   AND   max(|coef1|, |coef2|) > amplitude_threshold

    Parameters
    ----------
    stream              : obspy.core.stream.Stream
    t_start_origin      : float, seconds of pre-event record (~P-arrival time)
    t_min, t_max        : float, analysis window (s from trace start)
    inventory           : obspy.core.inventory.Inventory, optional
    misfit_threshold    : float (default 0.5)
    amplitude_threshold : float (default 0.2)

    Returns
    -------
    list of str -- trace IDs flagged as mouse-contaminated
    """
    try:
        st = stream.copy()
        if inventory:
            st.attach_response(inventory)
        return _run_check(st, t_start_origin, t_min, t_max,
                          misfit_threshold, amplitude_threshold)
    except Exception as exc:
        if _HAS_CONFIG:
            config.logger.warning('mouse.check() failed globally: %s', exc)
        return [tr.get_id() for tr in stream]


def _run_check(st, t_start_origin, t_min, t_max,
               misfit_threshold, amplitude_threshold):
    """Internal: run per-trace mouse detection on a prepared stream."""
    flagged    = []
    _mice_cache = {}   # reuse synthetic mice across components of the same station

    for tr in st:
        try:
            zeros, poles, gain, count2ground, instTcorner = \
                _paz_from_response(tr.stats.response)

            dt = tr.stats.delta

            # Only analyse up to the end of the requested window, not the full
            # trace.  Traces in the pipeline may be much longer than t_max
            # (e.g. 500 s), but the mouse only needs to be detected within
            # [0, t_start_origin + t_max].  Capping endtime here gives a ~6×
            # reduction in both shift-matrix cost and the 2-D BLAS search.
            full_endtime = (tr.stats.npts - 1) * dt
            endtime      = min(t_start_origin + t_max, full_endtime)
            n_use        = round(endtime / dt) + 1

            pwavetime = t_start_origin
            swavetime = t_start_origin + 0.5 * (t_max - t_start_origin)
            timlim    = 20.0  # fixed at MATLAB default; sufficient for detection

            # Build synthetic mice once per unique response (all components of
            # the same station share the same PAZ, so cache keyed on poles+gain).
            _cache_key = (tuple(np.round(np.concatenate([poles.real, poles.imag]), 6)),
                          round(gain, 2), round(endtime, 4))
            if _cache_key not in _mice_cache:
                _mice_cache[_cache_key] = _build_synthetic_mice(
                    zeros, poles, gain, dt, endtime, swavetime)
            mouse1, mouse2 = _mice_cache[_cache_key]

            # Invert directly (bypass characterize() to avoid rebuilding mice)
            n_sig = round(endtime / dt) + 1
            res   = _invert_component(
                tr.data[:n_sig], mouse1, mouse2, dt, pwavetime, endtime, timlim,
                _grid_density=25, _dtype=np.float32)

            max_amp  = max(abs(res['coef1']), abs(res['coef2']))
            detected = (res['misfit'] < misfit_threshold and
                        max_amp > amplitude_threshold)

            if _HAS_CONFIG:
                config.logger.info(
                    'Mouse check %s for %s: misfit=%.3f amp=%.3f',
                    'DETECTED' if detected else 'OK',
                    tr.get_id(), res['misfit'], max_amp)

            if detected:
                flagged.append(tr.get_id())

        except Exception as exc:
            if _HAS_CONFIG:
                config.logger.warning(
                    'Mouse check failed for %s: %s -- treating as contaminated',
                    tr.get_id(), exc)
            flagged.append(tr.get_id())

    return flagged


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST  (python mouse.py  [/path/to/test/data])
# ══════════════════════════════════════════════════════════════════════════════

def _self_test(datadir):
    """
    Verify against Zahradnik's MATLAB expected output (Mice_Fit_DROunc.dat).

    All 18 quantities (shift1, shift2, mouse1, mouse2, misfit, iter for each
    of 3 components) must match to within 0.01% of the MATLAB reference values.
    """
    import time

    C   = np.loadtxt(os.path.join(datadir, 'Invert.dat'), comments='%')
    endtime, pwavetime, swavetime, timlim = C[0], C[1], C[2], C[3]

    zeros        = np.array([0+0j, 0+0j])
    poles        = np.array([-0.037014+0.037008j, -0.037014-0.037008j,
                               -197.92  +207.35j,   -197.92  -207.35j,
                               -911.06  +0j])
    gain         = 74911164.0
    count2ground = 1e-9
    instTcorner  = 120.0

    A  = np.loadtxt(os.path.join(datadir, 'DROunc.dat'))
    dt = A[1, 0] - A[0, 0]

    # Reference values from MATLAB Mice_Fit_DROunc.dat
    expected = {
        'N': {'shift1': -1040, 'shift2':  240,
              'mouse1': -8.306063e-07, 'mouse2':  3.643758e-05,
              'misfit': 0.023502, 'iter': 2481},
        'E': {'shift1':  -840, 'shift2':  200,
              'mouse1':  6.460848e-07, 'mouse2': -4.710079e-06,
              'misfit': 0.220187, 'iter': 2985},
        'Z': {'shift1': -2000, 'shift2':  200,
              'mouse1':  1.358798e-08, 'mouse2': -4.491238e-05,
              'misfit': 0.017098, 'iter':   56},
    }

    t0     = time.time()
    m1, m2 = _build_synthetic_mice(zeros, poles, gain, dt, endtime, swavetime)
    t_mice = time.time() - t0

    ilft  = 0
    irgh  = round(endtime / dt) + 1   # 50001 samples (0..500 s)

    hdr = ('Comp   shift1   shift2          mouse1          mouse2    misfit   iter')
    print('\n' + hdr)
    print('-' * len(hdr))

    all_ok = True
    for col_idx, cname in enumerate(['N', 'E', 'Z']):
        y   = A[ilft:irgh, col_idx + 1].copy()
        res = _invert_component(y, m1, m2, dt, pwavetime, endtime, timlim)

        valmax = res['peak'] * count2ground * 4.0 * np.pi**2 * (1.0 / instTcorner)**2
        mo1    = res['coef1'] * valmax
        mo2    = res['coef2'] * valmax
        ex     = expected[cname]

        ok = [
            res['shift1']  == ex['shift1'],
            res['shift2']  == ex['shift2'],
            abs((mo1 - ex['mouse1']) / (abs(ex['mouse1']) + 1e-40)) < 1e-4,
            abs((mo2 - ex['mouse2']) / (abs(ex['mouse2']) + 1e-40)) < 1e-4,
            abs(res['misfit'] - ex['misfit']) < 1e-4,
            res['iter_no'] == ex['iter'],
        ]
        mark   = 'OK' if all(ok) else 'FAIL'
        all_ok &= all(ok)

        print(f'{mark} {cname}  {res["shift1"]:>7}  {res["shift2"]:>7}'
              f'  {mo1:>15.6e}  {mo2:>15.6e}  {res["misfit"]:.6f}  {res["iter_no"]}')

        if not all(ok):
            labels = ['shift1', 'shift2', 'mouse1', 'mouse2', 'misfit', 'iter']
            print(f'   EXP {ex["shift1"]:>7}  {ex["shift2"]:>7}'
                  f'  {ex["mouse1"]:>15.6e}  {ex["mouse2"]:>15.6e}'
                  f'  {ex["misfit"]:.6f}  {ex["iter"]}')
            print('   FAILED:', [l for l, o in zip(labels, ok) if not o])

    elapsed = time.time() - t0
    print()
    print(f'Mice construction : {t_mice*1000:.1f} ms')
    print(f'Total (3 comps)   : {elapsed*1000:.0f} ms')
    if all_ok:
        print('ALL CHECKS PASSED -- matches MATLAB output exactly.')
    else:
        print('SOME CHECKS FAILED -- see details above.')
    return all_ok


if __name__ == '__main__':
    _datadir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/for_efthimios_2'
    _ok = _self_test(_datadir)
    sys.exit(0 if _ok else 1)
