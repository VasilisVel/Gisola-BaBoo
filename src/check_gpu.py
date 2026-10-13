#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_gpu.py — check that Numba can use the NVIDIA GPU.

    cd src
    python check_gpu.py

It looks for a CUDA device (the NVIDIA driver) and runs a very small kernel that
also needs the CUDA libraries (NVVM and libdevice), so a missing library is found
here and not in the middle of an event. Exit code 0: the GPU works. Exit code 1: it does not.

run_check() runs this test in a separate process, for precompile.py and GisolaBaBoo.py: on some
systems (for example a driver newer than the Numba version supports) the test crashes the process
(segmentation fault), and that must not stop the installation or the program.
"""
import os
import sys
import math
import subprocess
import warnings


def main():
    warnings.simplefilter('ignore')

    try:
        import numpy as np
        from numba import cuda

        if not cuda.is_available():
            print('No CUDA device found by Numba (no NVIDIA driver, or it is not visible).')
            sys.exit(1)

        device = cuda.get_current_device()
        name = device.name.decode() if isinstance(device.name, bytes) else device.name

        @cuda.jit
        def _kernel(a):
            i = cuda.grid(1)
            if i < a.size:
                a[i] = math.sqrt(4.0) + math.exp(0.0)       # needs libdevice

        a = np.zeros(8)
        _kernel[1, 8](a)
        cuda.synchronize()
        if not np.allclose(a, 3.0):
            print('The GPU test kernel gave a wrong result.')
            sys.exit(1)

        print('GPU works: {} (compute capability {}.{})'.format(name, *device.compute_capability))
        sys.exit(0)

    except SystemExit:
        raise
    except Exception as e:
        message = str(e).strip().splitlines()[0][:200] if str(e).strip() else ''
        print('The GPU does not work: {}: {}'.format(type(e).__name__, message))
        sys.exit(1)


def run_check(script=None, timeout=300):
    """Runs the GPU test in a separate process. Returns (works, message)."""
    script = script or os.path.abspath(__file__)
    try:
        r = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, 'the GPU test did not finish in {} s'.format(timeout)
    except Exception as e:
        return False, 'the GPU test could not be started: {}'.format(e)
    lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
    message = lines[-1] if lines else ''
    if r.returncode == 0:
        return True, message
    if r.returncode < 0 or r.returncode >= 128:        # killed by a signal, e.g. segmentation fault
        return False, ('the GPU test crashed (segmentation fault): the NVIDIA driver is probably newer '
                       'than this version of Numba supports (CUDA 13 drivers)')
    return False, message or 'the GPU test failed'


if __name__ == '__main__':
    main()
