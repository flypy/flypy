# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from numba2 import jit, sjit, int32

#===------------------------------------------------------------------===
# Test code
#===------------------------------------------------------------------===

@jit
class C(object):
    layout = [('x', int32)]

    @jit
    def __init__(self, x):
        self.x = x

    @jit
    def __add__(self, other):
        return self.x * other.x

    @jit
    def __mul__(self, other):
        return C(self.method(other))

    @jit
    def method(self, other):
        return self.x * other.x

@jit('Composed[a]')
class Composed(object):
    layout = [('obj', 'a')]

    @jit
    def method(self):
        return self.obj.method(self.obj)

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
        class D(object):
            layout = [('x', 'float64')]

            @jit('a -> int64 -> void')
            def __init__(self, x):
                self.x = x

        @jit
        def f(x):
            return C(x).x * D(x).x

        self.assertEqual(f(10), 100.0)

class TestComposition(unittest.TestCase):

    def test_field_composition(self):
        "Test setfield/getfield for 'inlined' fields"
        @jit
        def f(x):
            c = Composed(C(x))
            return c.method()

        self.assertEqual(f(10), 100)

    def test_field_return(self):
        "Test returning an 'inlined' field"
        @jit
        def f(x):
            c = Composed(C(x))
            return c.obj

        self.assertEqual(f(10).x, 10)

    def test_field_apply(self):
        "Test passing around an 'inlined' field"
        @jit
        def f(x):
            c = Composed(C(x))
            return g(c.obj)

        @jit
        def g(x):
            return x * x

        self.assertEqual(f(10).x, 100)


class TestMutability(unittest.TestCase):

    def test_mutable_obj(self):
        @jit
        def f():
            obj = C(4)
            obj2 = g(obj)
            return obj.x, obj2.x

        @jit
        def g(obj):
            obj.x = 6
            return obj

        self.assertEqual(f(), (6, 6))


if __name__ == '__main__':
    #TestClasses('test_special_method').debug()
    unittest.main()