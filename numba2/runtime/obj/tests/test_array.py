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
        self.assertEqual(a, identity(a))

if __name__ == '__main__':
    unittest.main()
