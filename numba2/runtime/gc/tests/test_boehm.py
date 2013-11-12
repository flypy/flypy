# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest

from numba2 import jit, Pointer, float64, typeof, int32, cast, void
from numba2.runtime.gc import boehm as gc

class TestBoehm(unittest.TestCase):

    def test_boehm(self):
        @jit
        def f(n):
            for i in range(n):
                p = gc.gc_alloc(1000, Pointer[float64])

        f(1000000)

    def test_boehm_disable(self):
        @jit
        def f(n):
            gc.gc_disable()
            for i in range(n):
                p = gc.gc_alloc(1000, Pointer[float64])
            gc.gc_enable()
            gc.gc_collect()

        f(1000)

    def test_boehm_finalizer(self):
        @jit('Pointer[void] -> Pointer[void] -> void')
        def final(obj, data):
            print("tee hee, finally!")

        @jit
        def f():
            obj = gc.gc_alloc(1, int32)
            gc.gc_add_finalizer(obj, final)

        f()


if __name__ == '__main__':
    TestBoehm('test_boehm_finalizer').debug()
    #unittest.main()