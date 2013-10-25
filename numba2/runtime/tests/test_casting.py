# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from numba2 import jit, int32, float64, cast

class TestCasting(unittest.TestCase):

    def test_numeric_casting(self):
        @jit
        def f(x, dst_type):
            return cast(x, dst_type)

        self.assertEqual(f(2, float64), 2.0)
        self.assertEqual(f(2.0, int32), 2)

    def test_builtin_int_cast(self):
        @jit
        def f(x):
            return int(x)

        self.assertEqual(f(2.0), 2)


    def test_builtin_float_cast(self):
        @jit
        def f(x):
            return float(x)

        self.assertEqual(f(2), 2.0)


if __name__ == '__main__':
    unittest.main()