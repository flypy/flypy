# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from flypy import jit

import numpy as np

class TestArrayAttributes(unittest.TestCase):

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

class TestArrayIndexing(unittest.TestCase):

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

    def test_1d_array_setitem(self):
        @jit
        def index(a):
            a[6] = 14

        a = np.arange(10)
        index(a)
        self.assertEqual(a[6], 14)

    def test_1d_array_setitem_x(self):
        @jit
        def index(a, i):
            a[i] = 14

        for i in [0, 1, 8, 9]:
            a = np.arange(10)
            index(a, i)
            self.assertEqual(a[i], 14, "1D array getitem(%d)" % i)

    def test_2d_array_setitem(self):
        @jit
        def index(a):
            a[6, 9] = 14

        a = np.arange(8 * 12).reshape(8, 12)
        index(a)
        self.assertEqual(a[6, 9], 14)

    def test_2d_array_setitem_x(self):
        @jit
        def index(a, i, j):
            a[i, j] = 14

        x = [0, 1, 6, 7]
        y = [0, 1, 10, 11]
        for i in x:
            for j in y:
                a = np.arange(8 * 12).reshape(8, 12)
                index(a, i, j)
                self.assertEqual(a[i, j], 14, "2D array getitem(%d, %d)" %
                                              (i, j))

    def test_2d_array_setitem_0index(self):
        @jit
        def index(a):
            a[0, 0] = 14

        a = np.arange(8 * 12).reshape(8, 12)
        index(a)
        self.assertEqual(a[0, 0], 14)

    def test_nd_array_setitem(self):
        @jit
        def index(a, t):
            a[t] = 14

        def test(t, dtype=np.float64):
            shape = tuple(np.array(t) + 5)
            a = np.empty(shape, dtype=dtype)
            index(a, t)
            self.assertEqual(a[t], 14)

        test((2,))
        test((2, 6))
        test((2, 6, 9))
        test((2, 6, 9, 4))
        test((2, 6, 9, 4, 3))

    def test_partial_getitem(self):
        @jit
        def index(a):
            return a[6]

        a = np.arange(8 * 12).reshape(8, 12)
        result = index(a)
        self.assertEqual(len(result), 12)
        self.assertTrue(np.all(result == a[6]))

    def test_partial_setitem(self):
        @jit
        def index(a):
            a[6] = 4

        a = np.arange(8 * 12).reshape(8, 12)
        index(a)
        self.assertTrue(np.all(a[6] == 4))


class TestArraySlicing(unittest.TestCase):

    def test_1d_array_slice(self):
        @jit
        def index(a):
            return a[:]

        a = np.arange(10)
        self.assertTrue(np.all(a == index(a)))

    def test_1d_array_slice_bounds(self):
        @jit
        def index(a, start, stop, step):
            return a[start:stop:step]

        def test(start=0, stop=10, step=1):
            a = np.arange(10)
            result = index(a, start, stop, step)
            expected = a[start:stop:step]
            self.assertTrue(np.all(result == expected), (result, expected))

        # Ascending
        test(1)
        test(3)
        test(2, 8, 3)
        test(2, 9, 3)

        # Descending (wrap-around)
        test(-2)
        test(-2, -3)
        test(-2, -3, -1)

        # Wrap around and adjust
        test(-12, 3, 1)
        test(12, 4, -1)
        test(12, -3, -1)
        test(8, -12, -1)


    def test_2d_array_slice(self):
        @jit
        def index(a):
            return a[:, 5]

        a = np.arange(8 * 12).reshape(8, 12)
        result = index(a)
        self.assertTrue(np.all(a[:, 5] == result))


if __name__ == '__main__':
    unittest.main(verbosity=3)