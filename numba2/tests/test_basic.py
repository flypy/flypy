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

    def test_while(self):
        @jit
        def f(a, b):
            while a < b:
                a = a + a
            return a

        self.assertEqual(f(8, 10), 16)

    def test_call(self):
        @jit
        def g(a):
            return a + 2

        @jit
        def f(a):
            return g(a * 3)

        self.assertEqual(f(5), 27)


if __name__ == '__main__':
    #TestTranslation('test_call').debug()
    unittest.main()