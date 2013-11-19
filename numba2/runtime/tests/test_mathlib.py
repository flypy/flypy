# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import math
import cmath
import unittest

from numba2 import jit
from numba2.runtime import mathlib

import numpy as np

# ______________________________________________________________________

class TestMathLib(unittest.TestCase):

    def test_asin(self):
        @jit
        def f(x):
            return mathlib.asin(x)

        self.assertAlmostEqual(f(0.9), math.asin(0.9))
        #self.assertAlmostEqual(f(0.9+05.j), cmath.asin(0.9+0.5j))

    def test_sin(self):
        @jit
        def f(x):
            return mathlib.sin(x)

        self.assertAlmostEqual(f(0.9), math.sin(0.9))
        self.assertAlmostEqual(f(2), math.sin(2))

    def test_overlay(self):
        @jit
        def f(x):
            return math.asin(x)

        self.assertAlmostEqual(f(0.9), math.asin(0.9))

    def test_np_overlay(self):
        @jit
        def f(x):
            return np.arcsin(x)

        self.assertAlmostEqual(f(0.9), math.asin(0.9))

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()