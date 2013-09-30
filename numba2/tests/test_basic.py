# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from numba2 import jit

class TestTranslation(unittest.TestCase):

    def test_compare(self):
        @jit
        def f(a, b):
            return a < b

        self.assertEqual(f(5, 10), True)
        self.assertEqual(f(10, 5), False)

    def test_inplace(self):
        @jit
        def f(x):
            x += 5
            return x
        self.assertEqual(f(5), 10)

    def test_while(self):
        @jit
        def f(a, b):
            while a < b:
                a = a + a
            return a

        self.assertEqual(f(8, 10), 16)

    def test_for(self):
        @jit
        def f(L):
            sum = 0
            for x in L:
                sum += x
            return sum

        self.assertEqual(f(5), 17)

    def test_call(self):
        @jit
        def g(a):
            return a + 2
        @jit
        def f(a):
            return g(a * 3)

        self.assertEqual(f(5), 17)


if __name__ == '__main__':
    #TestTranslation('test_while').debug()
    unittest.main()