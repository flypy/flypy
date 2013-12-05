# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from numba2 import jit, typeof, int32, float64
#from numba2.compiler.representation import build_ctypes_representation
from numba2.runtime.obj.tupleobject import StaticTuple, EmptyTuple, NoneType
from numba2.conversion import fromobject, toobject, toctypes
from numba2.support.ctypes_support import CTypesStruct

none = NoneType()

def tonb(tup):
    return fromobject(tup, typeof(tup))

def topy(tup):
    return toobject(tup, typeof(tup))


class TestSmallTuple(unittest.TestCase):

    def test_typeof(self):
        "typeof"
        self.assertEqual(typeof((10, 20)),
                         StaticTuple[int32, StaticTuple[int32, EmptyTuple[()]]])

    def test_typeof_constant(self):
        t = StaticTuple(10, EmptyTuple())
        self.assertEqual(typeof(t), StaticTuple[int32, EmptyTuple[()]])
        t2 = StaticTuple(2.0, t)
        self.assertEqual(typeof(t2), StaticTuple[float64, StaticTuple[int32, EmptyTuple[()]]])

    def test_py_obj(self):
        "Test py impl"
        t = StaticTuple(10, StaticTuple(11, EmptyTuple()))
        self.assertEqual(t.hd, 10)
        self.assertEqual(t[0], 10)
        self.assertEqual(t[1], 11)

    def test_fromobject(self):
        "object -> tuple"
        obj = tonb((1, 2, 3))
        self.assertEqual(StaticTuple(1, StaticTuple(2, StaticTuple(3, EmptyTuple()))), obj)

    def test_toobject(self):
        "tuple -> object"
        self.assertEqual(topy(tonb((1, 2, 3))), (1, 2, 3))

    def test_representation(self):
        "ctypes"
        ty = typeof((1, 2, 3))
        obj = fromobject((1, 2, 3), ty)
        keepalive = []
        rep = toctypes(obj, ty, keepalive)
        rep = CTypesStruct(rep)

        # print(rep) -> { tl:{ tl:{ tl:{ dummy:0 }, hd:3 }, hd:2 }, hd:1 }
        self.assertEqual(rep.hd, 1)
        self.assertEqual(rep.tl.hd, 2)
        self.assertEqual(rep.tl.tl.hd, 3)


class TestJitTuple(unittest.TestCase):

    def test_getitem(self):
        @jit
        def f(a, b):
            t = (a, b)
            return t[1]

        self.assertEqual(f(5, 6), 6)

    def test_getitem_slice(self):
        @jit
        def f(t, s):
            return t[s]

        def test(t, s):
            self.assertEqual(f(t, s), t[s])

        t = (1, 2, 3)

        # Full
        test(t, slice(None, None, None))

        # TODO: implement tuple slicing

        ## Start
        #test(t, slice(1,    None, None))
        #test(t, slice(3,    None, None))
        #test(t, slice(10,   None, None))
        #test(t, slice(-1,   None, None))
        #test(t, slice(-6,   None, None))
        #
        ## Stop
        #test(t, slice(None, 1,    None))
        #test(t, slice(None, 3,    None))
        #test(t, slice(None, 5,    None))
        #test(t, slice(None, -1,   None))
        #test(t, slice(None, -5,   None))
        #
        ## Step
        #test(t, slice(None, None, 1))
        #test(t, slice(None, None, 2))
        #test(t, slice(None, None, 5))
        #test(t, slice(None, None, -1))
        #test(t, slice(None, None, -5))
        #
        ## Combination
        #test(t, slice(1, 2, 1))
        #test(t, slice(-2, None, None))
        #test(t, slice(None, None, 5))
        #test(t, slice(-1, -1, -1))

    def test_len(self):
        raise unittest.SkipTest
        @jit
        def f(t):
            return len(t)
        self.assertTrue(f(()), 0)
        self.assertFalse(f((1, 2, 3)), 3)

    def test_bool(self):
        raise unittest.SkipTest
        @jit
        def f(t):
            return bool(t)
        self.assertTrue(f(()), False)
        self.assertFalse(f((1, 2, 3)), True)


if __name__ == '__main__':
    unittest.main()