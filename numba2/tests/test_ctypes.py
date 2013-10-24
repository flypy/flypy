# -*- coding: utf-8 -*-

"""
Test support for ctypes.
"""

from __future__ import print_function, division, absolute_import

import ctypes.util
import math
import unittest

from numba2 import jit, typeof
from numba2.types import char, int32, int64, float32, float64, Function, int8, Pointer

#-------------------------------------------------------------------
# Setup
#-------------------------------------------------------------------

libc = ctypes.CDLL(ctypes.util.find_library('c'))
libm = ctypes.CDLL(ctypes.util.find_library('m'))

libc.printf.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
libc.printf.restype  = None
libm.cos.argtypes = [ctypes.c_double]
libm.cos.restype = ctypes.c_double

printf = libc.printf
cos = libm.cos

#-------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------

class TestCtypes(unittest.TestCase):

    def test_call_c_strings(self):
        raise unittest.SkipTest("TODO: unwrap() arguments to FFI calls")

        @jit
        def f(x):
            printf("hello %s\n", x)

        f(10)

    def test_call(self):
        @jit
        def f(value):
            return cos(value)

        self.assertEqual(f(math.pi), -1.0)

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()
