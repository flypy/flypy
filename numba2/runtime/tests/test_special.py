# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import ctypes

import unittest

from numba2 import jit, Type, int32, float64, typeof, addressof

## Test helpers

@jit
class C(object):
    layout = [('ran', 'bool')]

    @jit
    def method(self):
        self.ran = True
        return 12.0

##

class TestSpecial(unittest.TestCase):

    def test_typeof1(self):
        @jit('int32 -> Type[int32]')
        def f(x):
            return typeof(x)

        self.assertEqual(f(10), int32)

    def test_execute_subexpr_typeof(self):
        @jit
        class C(object):
            layout = [('ran', 'bool')]

            @jit
            def method(self):
                self.ran = True
                return 12.0

        @jit
        def f():
            obj = C(False)
            restype = typeof(obj.method())
            return obj.ran, restype

        self.assertEqual(f(), (True, float64))

    #def test_addressof(self):
    #    @jit
    #    def f():
    #        return addressof(g)
    #
    #    @jit('int32 -> int32')
    #    def g(x):
    #        return x * x
    #
    #    p = f()
    #    func = ctypes.cast(p, ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32))
    #    self.assertEqual(func(4), 16)


if __name__ == '__main__':
    unittest.main()