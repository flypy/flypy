# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from numba2 import jit, int32, Type
from numba2.typing import resolve

def _resolve(t, bound):
    return resolve(t, globals(), bound)

@jit('Int[X]')
class Int(object):
    layout = [('x', 'Int[X]'), ('y', 'Float[X]')]

    @jit('Int[X] -> Float[X]')
    def method(self):
        return Float(2.0)

    @jit('a -> Float[X]')
    def method2(self):
        return Float(2.0)


@jit('Float[X]')
class Float(object):

    layout = [('x', 'Int[X]'), ('y', 'Float[X]')]

    @jit('Float[X] -> Int[X]')
    def method(self):
        return Int(2)

#===------------------------------------------------------------------===
# Python tests
#===------------------------------------------------------------------===

class TestTyping(unittest.TestCase):

    def test_type_resolution(self):
        # -------------------------------------------------
        # Test fields
        [signature] = Int.type.fields['method'].signatures # Int[X] -> Float[X]
        signature = _resolve(signature, {})

        self.assertIsInstance(signature.argtypes[0], type(Int.type))
        self.assertIsInstance(signature.restype, type(Float.type))

        [signature] = Float.type.fields['method'].signatures # Float[X] -> Int[X]
        signature = _resolve(signature, {})
        self.assertIsInstance(signature.argtypes[0], type(Float.type))
        self.assertIsInstance(signature.restype, type(Int.type))

        # -------------------------------------------------
        # Test layout

        x = _resolve(Int.type.layout['x'], {})
        self.assertIsInstance(x, type(Int.type))

        # TODO: TypeVar equality differing instances in this context?
        self.assertEqual(str(x), str(Int.type))

    def test_typevar_resolution(self):
        int32 = Int[32]
        bound = int32.bound

        int32 = _resolve(int32, bound)
        self.assertIsInstance(int32, type(Int.type))

        # -------------------------------------------------
        # Test fields
        [signature] = int32.fields['method'].signatures
        self.assertEqual(str(_resolve(signature, bound)),
                         'Int[32] -> Float[32]')

        # -------------------------------------------------
        # Test layout

        x = _resolve(int32.layout['x'], bound)
        self.assertIsInstance(x, type(int32))
        self.assertEqual(int32, x)

#===------------------------------------------------------------------===
# jitted tests
#===------------------------------------------------------------------===

@jit('Type[int32] -> int32')
def match(x):
    return 2

@jit('Type[float32] -> int32')
def match(x):
    return 3

@jit
def type_as_value():
    return match(int32)

# ______________________________________________________________________

@jit('P[a]')
class P(object):
    layout = []

@jit('Type[P[int32]] -> int32')
def indexed(type):
    return 4

@jit
def type_indexing():
    return indexed(P[int32])

# ______________________________________________________________________

#@jit
#def type_return():
#    return Float[int32]

# ______________________________________________________________________

class TestTyping(unittest.TestCase):

    def test_type_as_value(self):
        self.assertEqual(type_as_value(), 2)

    def test_type_indexing(self):
        self.assertEqual(type_indexing(), 4)


if __name__ == '__main__':
    #TestTyping('test_typevar_resolution').debug()
    type_indexing()
    #unittest.main()