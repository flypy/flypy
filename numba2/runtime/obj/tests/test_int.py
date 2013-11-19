# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest

from numba2 import jit

class TestInt(unittest.TestCase):

    def test_int_formatting(self):
        @jit
        def f(x):
            return str(x)

        print(f(10))


if __name__ == '__main__':
    unittest.main()