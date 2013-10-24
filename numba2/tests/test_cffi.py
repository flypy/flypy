# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import math
import unittest

from numba2 import jit

try:
    import cffi
    ffi = cffi.FFI()
except ImportError:
    ffi = None

# ______________________________________________________________________

ffi.cdef("int printf(char *, ...);")
ffi.cdef("double cos(double);")

lib = ffi.dlopen(None)
libm = ffi.dlopen('m')

printf = lib.printf
cos = libm.cos

# ______________________________________________________________________

class TestCFFI(unittest.TestCase):

    def test_call_c_strings(self):
        raise unittest.SkipTest("TODO: unwrap() arguments to FFI calls")
        @jit
        def f(value):
            return printf(value)

        f("Hello world!\n")

    def test_call(self):
        @jit
        def f(value):
            return cos(value)

        f(math.pi)

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()