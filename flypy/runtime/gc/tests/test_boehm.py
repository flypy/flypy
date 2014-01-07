# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest

from flypy import jit, Pointer, float64, typeof, int32, cast, void
from flypy.runtime.gc import boehm as gc

class TestBoehm(unittest.TestCase):

    def test_boehm_direct(self):
        """Test direct usage of boehm
        """
        ptr = gc.gc_alloc(1000, Pointer[float64])
        # Make sure we have a valid pointer returned from gc.gc_alloc
        self.assertTrue(ptr.value != 0, str(ptr))

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

        # TODO: flypy.addressof()
        #f()


if __name__ == '__main__':
    unittest.main()