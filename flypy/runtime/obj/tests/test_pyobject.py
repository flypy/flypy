# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest

from flypy import jit, typeof, pyoverload
from flypy.types import Object
#from flypy.runtime.obj import object

# -----------------------

class C(object):

    def __init__(self, value):
        self.value = value

    __add__ = lambda self, other: self.value + other.value
    __mul__ = lambda self, other: self.value * other.value
    __sub__ = lambda self, other: self.value - other.value

    def __str__(self):
        return "hihi"

    def __nonzero__(self):
        return bool(self.value)

    __bool__ = __nonzero__

    def __len__(self):
        return 34

@typeof.case(C)
def typeof(value):
    return Object[()]

# -----------------------

class TestObjects(unittest.TestCase):

    def test_get_attribute(self):
        @jit
        def f(obj):
            return obj.value
        self.assertEqual(f(C(5)), 5)

    def test_set_attribute(self):
        @jit
        def f(obj1, obj2):
            obj1.foo = obj2

        obj1 = C(5)
        obj2 = C(6)
        f(obj1, obj2)
        self.assertEqual(obj1.foo.value, 6)


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

    def test_nonzero(self):
        @jit
        def f(obj):
            return bool(obj)

        self.assertEqual(f(C(0)), False)
        self.assertEqual(f(C(1)), True)
        self.assertEqual(f(C(2)), True)

    def test_len(self):
        @jit
        def f(obj):
            return len(obj)
        self.assertEqual(f(C(10)), 34)

    def test_str(self):
        @jit
        def f(x):
            return str(x)
        self.assertEqual(f(C(10)), "hihi")


if __name__ == '__main__':
    unittest.main(verbosity=3)
