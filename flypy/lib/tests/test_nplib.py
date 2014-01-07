# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from flypy import jit, int32, int64, float32, float64

import numpy as np


class TestNumpyLib(unittest.TestCase):

    def test_np_empty(self):
        @jit
        def empty(shape, dtype):
            return np.empty(shape, dtype)

        def test(shape, type, dtype):
            a = empty(shape, type)
            b = np.empty(shape, dtype)
            self.assertEqual(a.shape, shape)
            self.assertEqual(a.strides, b.strides)

        test((5, 8), float64, np.float64)
        test((8, 5), float64, np.float64)
        test((5, 8), int32, np.int32)
        test((8, 5), int32, np.int32)

    def test_np_empty_like(self):
        @jit
        def empty_like(a, dtype):
            return np.empty_like(a, dtype)

        a = np.empty((10, 12), dtype=np.int64)

        self.assertEqual(empty_like(a, int64).shape, a.shape)
        self.assertEqual(empty_like(a, int64).strides, a.strides)
        self.assertEqual(empty_like(a, float32).shape, a.shape)

    def test_np_zeros_like(self):
        @jit
        def zeros_like(a, dtype):
            return np.zeros_like(a, dtype)

        a = np.zeros((10, 12), dtype=np.int64)

        self.assertEqual(zeros_like(a, int64).shape, a.shape)
        self.assertEqual(zeros_like(a, int64).strides, a.strides)
        self.assertEqual(zeros_like(a, float32).shape, a.shape)
        self.assertTrue(np.all(zeros_like(a, float32) == 0))

    def test_np_empty_like_no_dtype(self):
        @jit
        def empty_like(a):
            return np.empty_like(a)

        a = np.empty((10, 12), dtype=np.int64)

        # Test without dtype
        self.assertEqual(empty_like(a).shape, a.shape)
        self.assertEqual(empty_like(a).dtype, a.dtype)


if __name__ == '__main__':
    unittest.main(verbosity=3)
