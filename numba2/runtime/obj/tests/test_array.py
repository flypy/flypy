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

    #def test_array_index(self):
    #    @jit
    #    def index(a):
    #        return a[6]
    #
    #    a = np.arange(10)
    #    self.assertEqual(a[6], index(a))


if __name__ == '__main__':
    unittest.main()
