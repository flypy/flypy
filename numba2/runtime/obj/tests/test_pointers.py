# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import ctypes
import cffi
import unittest

from numba2 import jit, typeof, int32, float64, NULL

ffi = cffi.FFI()
arr = ffi.new('int[3]')
arr[0] = 5
arr[1] = 6
p = ffi.cast('int *', arr)

class TestPointers(unittest.TestCase):

    def test_pointer_getitem(self):
        @jit
        def f(p):
            return p[1]
        self.assertEqual(f(p), 6)

    def test_pointer_setitem(self):
        @jit
        def f(p):
            p[2] = 14
            return p[2]
        self.assertEqual(f(p), 14)

    def test_pointer_compare(self):
        @jit
        def f(p):
            return p == NULL

        p1 = ffi.new('int *')
        p2 = ffi.new('float *')
        p3 = ffi.cast('void *', p1)

        self.assertFalse(f(p1))
        self.assertFalse(f(p2))
        self.assertFalse(f(p3))

        self.assertTrue(f(ctypes.c_void_p(0)))


if __name__ == '__main__':
    unittest.main()