# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from numba2 import jit, int32, int64, float32, float64

import numpy as np


class TestNumpyLib(unittest.TestCase):

    def test_np_empty_like(self):
        @jit
        def empty(shape, dtype):
            return np.empty(shape, dtype)

        def test(shape, type, dtype):
            a = empty(shape, type)
            b = np.empty(shape, dtype)
            self.assertEqual(a.shape, shape)
            self.assertEqual(a.strides, b.strides)

        test((5, 8), float64, np.float64)


if __name__ == '__main__':
    unittest.main(verbosity=3)
