# -*- coding: utf-8 -*-

"""
Test support for ctypes.
"""

from __future__ import print_function, division, absolute_import

import ctypes.util
import unittest

from numba2 import jit, typeof
from numba2.types import char, int32, int64, float32, float64, Function, int8, Pointer

#-------------------------------------------------------------------
# Setup
#-------------------------------------------------------------------

libc = ctypes.CDLL(ctypes.util.find_library('c'))
libc.printf.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
libc.printf.restype  = None

printf = libc.printf

#-------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------

class TestCtypes(unittest.TestCase):

    def test_call(self):
        @jit
        def f(x):
            printf("hello %s\n", x)

        f(10)


if __name__ == '__main__':
    unittest.main()