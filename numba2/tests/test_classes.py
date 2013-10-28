# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from numba2 import jit, sjit, int32

#===------------------------------------------------------------------===
# Test code
#===------------------------------------------------------------------===

@jit
class C(object):
    layout = {'x': int32}

    @jit
    def __init__(self, x):
        self.x = x

    @jit
    def __add__(self, other):
        return self.x * other.x

    @jit
    def method(self, other):
        return self.x * other.x

# ______________________________________________________________________

@jit
def call_special(x):
    return C(x) + C(2)

@jit
def call_method(x):
    return C(x).method(C(2))

@jit
def return_obj(x):
    return C(x)

# ______________________________________________________________________

@jit('Parameterized[a]')
class Parameterized(object):
    layout = [('x', 'a')]

    @jit
    def __init__(self, x):
        self.x = x

@jit
def call_parameterized(x):
    return Parameterized(x).x

#===------------------------------------------------------------------===
# Tests
#===------------------------------------------------------------------===

class TestClasses(unittest.TestCase):

    def test_special_method(self):
        self.assertEqual(call_special(5), 10)

    def test_methods(self):
        self.assertEqual(call_method(5), 10)

    def test_return_obj(self):
        # TODO: Heap types: allocation, returning
        # TODO: stack allocated types: return by value, pass by pointer,
        # validate immutability in typechecker
        obj = return_obj(10)
        self.assertIsInstance(obj, C)
        self.assertEqual(obj.x, 10)


class TestParameterized(unittest.TestCase):

    def test_parameterized(self):
        self.assertEqual(call_parameterized(2.0), 2.0)


class TestConstructors(unittest.TestCase):

    def test_constructor_promotion(self):
        @jit
        class C(object):
            layout = [('x', 'float64')]

            @jit('a -> float64 -> void')
            def __init__(self, x):
                self.x = x

        @jit
        def f(x):
            return C(x).x

        self.assertEqual(f(10), 10.0)

if __name__ == '__main__':
    #TestClasses('test_special_method').debug()
    unittest.main()
