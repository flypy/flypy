# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import math
import unittest

from flypy import jit, types, cast
from flypy.types import int32, float64, Type
from flypy.runtime import ffi

# ______________________________________________________________________

class TestFFI(unittest.TestCase):

    def test_malloc(self):
        @jit
        def f():
            p = ffi.malloc(cast(2, types.int64), types.int32)
            p[0] = 4
            p[1] = 5
            return p

        p = f()
        self.assertEqual(p[0], 4)
        self.assertEqual(p[1], 5)

    def test_sizeof(self):
        def func(x):
            return ffi.sizeof(x)

        def apply(signature, arg):
            return jit(signature)(func)(arg)

        self.assertEqual(apply('int32 -> int64', 10), 4)
        self.assertEqual(apply('Type[int32] -> int64', int32), 4)
        self.assertEqual(apply('float64 -> int64', 10.0), 8)
        self.assertEqual(apply('Type[float64] -> int64', float64), 8)

    def test_ffi_func_as_value(self):
        @jit
        def square(x):
            return x * x

        @jit
        def apply(f, x):
            return f(x)

        c_square, _ = square.translate([int32])
        self.assertEqual(apply(c_square, 10), 100)

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()