# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import ctypes
import cffi
import unittest

from numba2 import jit, typeof, int32, float64, NULL

class TestStrings(unittest.TestCase):

    def test_pointer_getitem(self):
        @jit
        def f(a, b):
            return a == b

        self.assertEqual(f("foo", "foo"), True)
        self.assertEqual(f("foo", "bar"), False)


if __name__ == '__main__':
    unittest.main()