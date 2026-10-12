#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_gpu.py — check that Numba can use the NVIDIA GPU.

    cd src
    python check_gpu.py

It looks for a CUDA device (the NVIDIA driver) and runs a very small kernel that
also needs the CUDA libraries (NVVM and libdevice), so a missing library is found
here and not in the middle of an event. Exit code 0: the GPU works. Exit code 1: it does not.
"""
import sys
import math
import warnings

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
