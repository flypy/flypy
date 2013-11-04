# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest

from numba2 import jit, Object, typeof, pyoverload
#from numba2.runtime.obj import object

class C(object):

    def __init__(self, value):
        self.value = value

    __add__ = lambda self, other: self.value + other.value
    __mul__ = lambda self, other: self.value * other.value
    __sub__ = lambda self, other: self.value - other.value

@typeof.case(C)
def typeof(value):
    return Object[()]

class TestObjects(unittest.TestCase):

    def test_add(self):
        @jit
        def f(a, b):
            return a + b
        self.assertEqual(f(C(5), C(6)), 11)

    def test_mul(self):
        @jit
        def f(a, b):
            return a * b
        self.assertEqual(f(C(5), C(6)), 30)

    def test_sub(self):
        @jit
        def f(a, b):
            return a - b
        self.assertEqual(f(C(5), C(6)), -1)


if __name__ == '__main__':
    unittest.main()