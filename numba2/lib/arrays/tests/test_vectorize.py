# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest

from numba2 import jit
from numba2.lib.arrays.vectorize import broadcast, add

import numpy as np

class TestBroadcasting(unittest.TestCase):

    def _run(self, a, b, shape1, shape2):
        @jit
        def f(a, b):
            return broadcast(a, b)

        a = np.array(a)
        b = np.array(b)
        result1, result2 = f(a, b)
        expected1, expected2 = np.broadcast_arrays(a, b)

        self.assertEqual(result1.shape, shape1)
        self.assertEqual(result2.shape, shape2)

        self.assertTrue(np.all(result1 == expected1), str(result1))
        self.assertTrue(np.all(result2 == expected2), str(result2))

    def test_broadcast_equal(self):
        self._run([1, 2], [3, 4], (2,), (2,))

    def test_broadcast_unequeal(self):
        self._run([[1, 2], [3, 4]], [5, 6], (2, 2), (1, 2))
        self._run([1, 2], [[3, 4], [5, 6]], (1, 2), (2, 2))



class TestVectorize(unittest.TestCase):

    def test_add(self):
        @jit
        def f(a, b):
            return add(a, b)

        a = np.arange(10)
        b = np.arange(10, 20)
        self.assertEqual(f(a, b), a + b)


if __name__ == '__main__':
    unittest.main()