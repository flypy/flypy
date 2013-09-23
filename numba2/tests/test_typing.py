# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from numba2 import jit

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


class TestTyping(unittest.TestCase):

    def test_type_resolution(self):
        # -------------------------------------------------
        # Test fields
        signature = Int.type.fields['method'].signature # Int[X] -> Float[X]
        self.assertIsInstance(signature.argtypes[0], type(Int.type))
        self.assertIsInstance(signature.restype, type(Float.type))

        signature = Float.type.fields['method'].signature # Float[X] -> Int[X]
        self.assertIsInstance(signature.argtypes[0], type(Float.type))
        self.assertIsInstance(signature.restype, type(Int.type))

        # -------------------------------------------------
        # Test layout

        x = Int.type.layout['x']
        self.assertIsInstance(x, type(Int.type))

        # TODO: TypeVar equality differing instances in this context?
        self.assertEqual(str(x), str(Int.type))

    def test_typevar_resolution(self):
        int32 = Int[32]
        self.assertIsInstance(int32, type(Int.type))

        # -------------------------------------------------
        # Test fields

        self.assertEqual(str(int32.fields['method'].signature),
                         'Int[32] -> Float[32]')

        # -------------------------------------------------
        # Test layout

        x = int32.layout['x']
        self.assertIsInstance(x, type(int32))
        self.assertEqual(int32, x)

if __name__ == '__main__':
    # TestTyping('test_type_resolution').debug()
    unittest.main()