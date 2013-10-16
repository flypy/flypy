# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from numba2 import jit, typeof, int32
from numba2.runtime.obj.tupleobject import StaticTuple, NoneType
from numba2.runtime.conversion import fromobject, toobject

none = NoneType()

class TestSmallTuple(unittest.TestCase):

    def test_typeof(self):
        self.assertEqual(typeof((10, 20)),
                         StaticTuple[int32, StaticTuple[int32, NoneType[()]]])

    #def test_typeof_constant(self):
    #    t = StaticTuple(10, none)
    #    self.assertEqual(typeof(t), StaticTuple[int32, NoneType[()]])

    def test_py_obj(self):
        t = StaticTuple(10, StaticTuple(11, None))
        self.assertEqual(t.hd, 10)
        self.assertEqual(t[0], 10)
        self.assertEqual(t[1], 11)

    def test_fromobject(self):
        value = (1, 2, 3)
        obj = fromobject(value, typeof(value))
        self.assertEqual(StaticTuple(1, StaticTuple(2, StaticTuple(3, None))), obj)

    def test_toobject(self):
        tup = StaticTuple(1, StaticTuple(2, StaticTuple(3, None)))
        obj = toobject(tup, typeof((1, 2, 3)))
        self.assertEqual(obj, (1, 2, 3))


if __name__ == '__main__':
    unittest.main()