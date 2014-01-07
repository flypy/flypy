# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import ctypes
import unittest

from numba2 import jit, int32, float64, cast, Pointer

class TestCasting(unittest.TestCase):

    def test_numeric_casting(self):
        @jit
        def f(x, dst_type):
            return cast(x, dst_type)

        self.assertEqual(f(2, float64), 2.0)
        self.assertEqual(f(2.0, int32), 2)

    def test_builtin_int_cast(self):
        @jit
        def f(x):
            return int(x)

        self.assertEqual(f(2.0), 2)

    def test_builtin_float_cast(self):
        @jit
        def f(x):
            return float(x)

        self.assertEqual(f(2), 2.0)

    def test_pointer_cast(self):
        @jit
        def f(x, dst_type):
            return cast(x, dst_type)

        p = ctypes.pointer(ctypes.c_double(5.0))
        p = ctypes.cast(p, ctypes.c_void_p)
        newp = f(p, Pointer[float64])

        self.assertEqual(p.value, ctypes.cast(newp, ctypes.c_void_p).value)
        self.assertEqual(newp[0], 5.0)

    def test_type_appl(self):
        raise unittest.SkipTest("Casting through type application")

        @jit
        def f(x, dst_type):
            return dst_type(x)

        self.assertEqual(f(2, float64), 2.0)
        self.assertEqual(f(2.0, int32), 2)

    def test_int_to_pointer_cast(self):
        @jit
        def f(x, dst_type):
            return cast(x, dst_type)

        newp = f(200, Pointer[float64])
        self.assertEqual(200, ctypes.cast(newp, ctypes.c_void_p).value)


if __name__ == '__main__':
    unittest.main()