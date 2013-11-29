# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from numba2 import jit

import numpy as np

class TestArray(unittest.TestCase):

    def test_array_create(self):
        @jit
        def identity(a):
            return a

        a = np.arange(10)
        result = identity(a)
        self.assertTrue(np.all(a == result))

    def test_array_length(self):
        @jit
        def length(a):
            return len(a)

        self.assertEqual(length(np.arange(10)), 10)
        self.assertEqual(length(np.empty((12, 8))), 12)

    def test_1d_array_index(self):
        @jit
        def index(a):
            return a[6]

        a = np.arange(10)
        self.assertEqual(a[6], index(a))

    def test_2d_array_index(self):
        @jit
        def index(a):
            return a[6, 9]

        a = np.arange(8 * 12).reshape(8, 12)
        self.assertEqual(a[6, 9], index(a))

    def test_nd_array_index(self):
        @jit
        def index(a, t):
            return a[t]

        def test(t, dtype=np.float64):
            shape = tuple(np.array(t) + 5)
            a = np.empty(shape, dtype=dtype)
            a[t] = 6.4

            self.assertEqual(6.4, index(a, t))

        test((2,))
        test((2, 6))
        test((2, 6, 9))
        test((2, 6, 9, 4))
        test((2, 6, 9, 4, 3))


if __name__ == '__main__':
    unittest.main(verbosity=3)
