# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest

from numba2 import jit

@jit('int32 -> int32 -> int32')
def f(a, b):
    return a + b

@jit('float64 -> float64 -> float64')
def f(a, b):
    return a * b

#===------------------------------------------------------------------===
# Tests
#===------------------------------------------------------------------===

class TestOverloading(unittest.TestCase):

    def test_overloading(self):
        @jit
        def func1():
            f(1, 2)
            return f(1.0, 2.0)

        @jit
        def func2():
            f(1.0, 2.0)
            return f(1, 2)

        self.assertEqual(func1(), 2.0)
        self.assertEqual(func2(), 3)

if __name__ == '__main__':
    unittest.main()