# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from numba2 import jit

import numpy as np

class TestGenerators(unittest.TestCase):

    def test_simple_generators(self):
        @jit
        def enumerate(it):
            i = 0
            for x in it:
                yield i, x
                i += 1

        @jit
        def consume(it):
            result = 0
            for t in enumerate(it):
                result += t[0] * t[1]
            return result

        result = consume(range(10, 20))
        expected = np.sum(np.arange(10, 20) * np.arange(10))

        self.assertEqual(result, expected)

    def test_nested_generators(self):
        def f(x):
            result = 0
            for x in g(x):
                for y in x:
                    result += y * 2
            return result

        def g(x):
            for i in range(10):
                yield h(i)

        def h(x):
            for i in range(10):
                yield x * i

        expected = f(10)
        f = jit(f); g = jit(g); h = jit(h)
        result = f(10)

        self.assertEqual(result, expected)

    def test_return_generator(self):
        raise unittest.SkipTest(
            "Inline function with generator return types without yield "
            "before fusion")

        def f(x):
            result = 0
            for x in g(x):
                result += x * 2
            return result

        def g(x):
            return h(x * 2)

        def h(x):
            for i in range(10):
                yield x * i

        expected = f(10)
        f = jit(f); g = jit(g); h = jit(h)
        result = f(10)

        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()