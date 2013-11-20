# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest

from numba2 import jit
from numba2.types import floating

class TestFloat(unittest.TestCase):

    def test_float_formatting(self):
        @jit
        def f(x):
            return str(x)

        for i in map(float, range(-300, 300, 100)):
            expected = str(i)
            result = f(i)
            result, trailing = result[:len(expected)], result[len(expected):]
            self.assertEqual(result, expected)
            self.assertEqual(len(trailing.strip("0")), 0, trailing)


if __name__ == '__main__':
    unittest.main()