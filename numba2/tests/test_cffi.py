# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import os
import ctypes
import unittest

from numba2 import jit

try:
    import cffi
    ffi = cffi.FFI()
except ImportError:
    ffi = None

# ______________________________________________________________________

# Test printf for nopython and no segfault
ffi.cdef("int printf(char *, ...);", override=True)
lib = ffi.dlopen(None)
printf = lib.printf

# ______________________________________________________________________

class TestCFFI(unittest.TestCase):

    def test_cffi_calls(self):
        @jit
        def f(value):
            return printf(value)

        f("Hello world!\n")

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()