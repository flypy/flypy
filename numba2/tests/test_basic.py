# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from pykit.ir.interp import UncaughtException
from numba2 import jit

class TestTranslation(unittest.TestCase):

    def test_compare(self):
        @jit
        def f(a, b):
            return a < b

        self.assertEqual(f(5, 10), True)
        self.assertEqual(f(10, 5), False)

if __name__ == '__main__':
    TestTranslation('test_compare').debug()
    # unittest.main()