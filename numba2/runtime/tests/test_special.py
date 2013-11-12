# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import ctypes

import unittest

from numba2 import jit, Type
from numba2.runtime.special import typeof, addressof

class TestSpecial(unittest.TestCase):

    #def test_typeof(self):
    #    @jit
    #    def f(x):
    #        return typeof(x)
    #    print(f(10))

    def test_addressof(self):
        @jit
        def f():
            return addressof(g)

        @jit('int32 -> int32')
        def g(x):
            return x * x

        p = f()
        func = ctypes.cast(p, ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32))
        self.assertEqual(func(4), 16)


if __name__ == '__main__':
    unittest.main()