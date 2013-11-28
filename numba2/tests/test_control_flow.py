# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from numba2 import jit

class TestControlFlow(unittest.TestCase):

    def test_loop_carried_dep_promotion(self):
        @jit
        def f(n):
            sum = 0
            for i in range(n):
                sum += float(i)
            return sum

        self.assertEqual(f(10), 45.0)

    def test_nested_rectangular(self):
        @jit
        def f(n):
            sum = 0
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        sum += i * j
            return sum

        self.assertEqual(f(3), f.py_func(3))

    def test_break(self):
        @jit
        def f(n):
            sum = 0
            for i in range(n):
                if i > n - 4:
                    break
                sum += i
            return sum

        self.assertEqual(f(10), f.py_func(10))

    def test_complicated(self):
        @jit
        def f(n):
            sum = 0
            for i in range(n):
                if i % 4 > 2:
                    while i > 0:
                        for j in range(n):
                            i -= 1
                            for k in range(n):
                                while k:
                                    sum += i * j
                                    break
                                else:
                                    continue
                                break
            return sum

        print(f(3))
        #self.assertEqual(f(3), f.py_func(3))


if __name__ == '__main__':
    #TestControlFlow('test_reduction').debug()
    unittest.main()