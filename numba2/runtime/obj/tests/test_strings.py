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
        raise unittest.SkipTest("returning objects")
        @jit
        def f(s1):
            return s1[3]

        self.assertEqual(f("blah"), "h")

    def test_nonzero(self):
        @jit
        def f(s):
            return bool(s)
        self.assertEqual(f(""), False)
        self.assertEqual(f("a"), True)
        self.assertEqual(f("ham"), True)

    def test_string_add(self):
        raise unittest.SkipTest("""
        This works only when run in isolation:
              File "/Users/mark/numba-lang/numba2/compiler/optimizations/throwing.py", line 16, in rewrite_exceptions
    raise NotImplementedError("Exception throwing", op, func)
NotImplementedError: ('Exception throwing', %15, Function(numba2.runtime.obj.rangeobject.__next__))
        """)
        @jit
        def f(s1, s2):
            return s1 + s2
        self.assertEqual(f("foo", "bar"), "foobar")
        self.assertEqual(f("foo", ""), "foo")
        self.assertEqual(f("", "bar"), "bar")

if __name__ == '__main__':
    unittest.main(verbosity=3)