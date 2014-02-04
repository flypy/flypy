# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import random
import ctypes
import unittest

from flypy import jit, cast, typeof, sizeof, NULL
from flypy.types import Pointer, float64, int32, int64, void, struct_
from flypy.runtime.gc import semispace as gc

import numpy as np

A_layout =  [
    ('left', 'B[]'),
    ('right', 'int32'),
]

B_layout = [
    ('value', 'int32'),
]

@jit
class A(object):
    layout = A_layout

    @jit
    def __flypy_trace(self, gc):
        self = gc.trace(self, sizeof(self))
        self.left = self.left.__flypy_trace(gc)

@jit
class B(object):
    layout = B_layout

    @jit
    def __flypy_trace(self, gc):
        self = gc.trace(self, sizeof(self))

ptrsize = ctypes.sizeof(ctypes.c_void_p)

type_a = Pointer[struct_(A_layout)]
type_b = Pointer[struct_(B_layout)]

class TestBumpAllocator(unittest.TestCase):

    def test_alloc(self):
        @jit
        def f(heap_size, n):
            space = gc.BumpAllocator(heap_size)

            for i in range(n):
                p = space.alloc(1)
                if p == NULL:
                    return False

            return True

        self.assertEqual(f(80, 80 // ptrsize), True)
        self.assertEqual(f(80, 80 // ptrsize + 1), False)


class TestSemiSpace(unittest.TestCase):

    def test_alloc(self):
        @jit
        def f():
            space = gc.GC(100)

            a = space.alloc(sizeof(A.type))
            b = space.alloc(sizeof(B.type))

            a = cast(a, type_a)
            b = cast(b, type_b)

            a.left = b
            a.right = 9
            b.value = 10



if __name__ == '__main__':
    unittest.main()