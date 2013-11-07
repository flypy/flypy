# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import ctypes.util
import unittest

from numba2 import jit

libc = ctypes.CDLL(ctypes.util.find_library("c"))

libc.printf.argtypes = [ctypes.c_char_p]
libc.restype = ctypes.c_int

#===------------------------------------------------------------------===
# Tests
#===------------------------------------------------------------------===

class TestTypeCheck(unittest.TestCase):

    def test_call_ctypes_mismatch_argno(self):
        @jit
        def simple():
            libc.printf("hello", "world")

        self.assertRaises(TypeError, simple)

    def test_call_ctypes_mismatch_argtypes(self):
        @jit
        def simple():
            libc.printf(10)

        self.assertRaises(TypeError, simple)


if __name__ == '__main__':
    unittest.main()