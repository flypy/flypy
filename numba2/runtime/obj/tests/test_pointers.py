# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import cffi
import unittest

from numba2 import jit, typeof, int32, float64

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


if __name__ == '__main__':
    unittest.main()