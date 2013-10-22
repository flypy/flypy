# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from numba2 import jit, typeof, int32, float64
#from numba2.compiler.representation import build_ctypes_representation
from numba2.runtime.obj.tupleobject import StaticTuple, NoneType
from numba2.runtime.conversion import fromobject, toobject, toctypes

none = NoneType()

class TestSmallTuple(unittest.TestCase):

    def test_typeof(self):
        "typeof"
        self.assertEqual(typeof((10, 20)),
                         StaticTuple[int32, StaticTuple[int32, NoneType[()]]])

    def test_typeof_constant(self):
        t = StaticTuple(10, none)
        self.assertEqual(typeof(t), StaticTuple[int32, NoneType[()]])
        t2 = StaticTuple(2.0, t)
        self.assertEqual(typeof(t2), StaticTuple[float64, StaticTuple[int32, NoneType[()]]])


    def test_py_obj(self):
        "Test py impl"
        t = StaticTuple(10, StaticTuple(11, None))
        self.assertEqual(t.hd, 10)
        self.assertEqual(t[0], 10)
        self.assertEqual(t[1], 11)

    def test_fromobject(self):
        "object -> tuple"
        value = (1, 2, 3)
        obj = fromobject(value, typeof(value))
        self.assertEqual(StaticTuple(1, StaticTuple(2, StaticTuple(3, None))), obj)

    def test_toobject(self):
        "tuple -> object"
        tup = StaticTuple(1, StaticTuple(2, StaticTuple(3, None)))
        obj = toobject(tup, typeof((1, 2, 3)))
        self.assertEqual(obj, (1, 2, 3))

    def test_representation(self):
        "ctypes"
        ty = typeof((1, 2, 3))
        obj = fromobject((1, 2, 3), ty)
        keepalive = []
        rep = toctypes(obj, ty, keepalive)

        # print(rep) -> { tl:{ tl:{ tl:{ dummy:0 }, hd:3 }, hd:2 }, hd:1 }
        self.assertEqual(rep.hd, 1)
        self.assertEqual(rep.tl.hd, 2)
        self.assertEqual(rep.tl.tl.hd, 3)


class TestJitTuple(unittest.TestCase):

    def test_jit_smalltup(self):
        @jit
        def f(a, b):
            t = (a, b)
            return t[1]

        self.assertEqual(f(5, 6), 6)

if __name__ == '__main__':
    unittest.main()