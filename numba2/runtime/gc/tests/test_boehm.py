# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest

from numba2 import jit, Pointer, float64, typeof
from numba2.runtime.gc import boehm

class TestBoehm(unittest.TestCase):

    def test_boehm(self):
        @jit
        def f(n):
            for i in range(n):
                p = boehm.gc_alloc(1000, Pointer[float64])

        self.assertEqual(f(1000000), 6)

if __name__ == '__main__':
    unittest.main()