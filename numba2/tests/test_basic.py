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
        def f(n):
            sum = 0
            for x in range(n):
                sum += x
            return sum

        self.assertEqual(f(6), 15)

    def test_and(self):
        @jit
        def f(x, y):
            if x < y and y > 10:
                return 4
            return 5

        self.assertEqual(f(2, 12), 4)
        self.assertEqual(f(2, 3), 5)
        self.assertEqual(f(3, 2), 5)
        self.assertEqual(f(12, 2), 5)

    def test_or(self):
        @jit
        def f(x, y):
            if x < y or y > 10:
                return 4
            return 5

        self.assertEqual(f(2, 12), 4)
        self.assertEqual(f(20, 12), 4)
        self.assertEqual(f(2, 9), 4)
        self.assertEqual(f(12, 2), 5)

    def test_andor(self):
        @jit
        def f(x, y):
            if x < y or (y > 10 and y < 15) or x > 2:
                return 4
            return 5

        self.assertEqual(f(2, 12), 4)
        self.assertEqual(f(2, 1), 5)

    def test_call(self):
        @jit
        def g(a):
            return a + 2
        @jit
        def f(a):
            return g(a * 3)

        self.assertEqual(f(5), 17)

    def test_call(self):
        @jit('int64 -> int64')
        def g(x):
            return x * 2

        @jit('int32 -> a')
        def f(x):
            return g(x) + 2

        self.assertEqual(f(3), 8)

    def test_tuple_passing(self):
        @jit
        def f():
            return g(1, 2, (1, 2, 3), 4, 5)

        @jit
        def g(a, b, t, c, d):
            return a + b + t[1] + c + d

        self.assertEqual(f(), 14)


if __name__ == '__main__':
    #TestTranslation('test_call').debug()
    unittest.main()