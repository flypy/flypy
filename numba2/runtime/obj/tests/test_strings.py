# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import ctypes
import cffi
import unittest

from numba2 import jit, typeof, int32, float64, NULL

class TestStrings(unittest.TestCase):

    def test_string_compare(self):
        @jit
        def f(a, b):
            return a == b

        self.assertEqual(f("foo", "foo"), True)
        self.assertEqual(f("foo", "bar"), False)

    def test_string_return(self):
        @jit
        def f(s):
            return s
        self.assertEqual(f("blah"), "blah")

    def test_string_indexing(self):
        @jit
        def f(s1):
            return s1[3]

        self.assertEqual(f("blah"), "h")


if __name__ == '__main__':
    unittest.main()