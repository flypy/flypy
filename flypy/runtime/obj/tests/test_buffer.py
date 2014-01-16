# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import ctypes
import unittest

from flypy import jit
from flypy.types import int32
from flypy.runtime.obj.bufferobject import Buffer, fromseq

import numpy as np

class TestPointers(unittest.TestCase):

    def test_buffer_getitem(self):
        @jit
        def f(b):
            return b[1]

        b = fromseq([1, 2, 3], int32)
        self.assertEqual(f(b), 2)

    def test_buffer_iter(self):
        @jit
        def f(b):
            sum = 0
            for x in b:
                sum += x
            return sum

        b = fromseq(range(10), int32)
        self.assertEqual(f(b), sum(range(10)))

        b = fromseq(range(11), int32)
        self.assertEqual(f(b), sum(range(11)))

    def test_buffer_setitem(self):
        @jit
        def f(b):
            b[1] = 14

        b = fromseq([1, 2, 3], int32)
        f(b)
        self.assertEqual(b[1], 14)

    def test_buffer_setslice(self):
        @jit
        def f(b):
            b[1:-1:2] = 0

        b = fromseq(range(10), int32)
        f(b)

        x = np.arange(10)
        f.py_func(x)
        self.assertTrue(np.all(list(b) == x))


if __name__ == '__main__':
    unittest.main()