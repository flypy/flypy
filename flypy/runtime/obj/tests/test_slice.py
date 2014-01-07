# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest

from flypy import jit

class TestSlice(unittest.TestCase):

    def test_slice(self):
        @jit
        def f(s):
            return (s.start, s.stop, s.step)

        def test(*args):
            self.assertEqual(f(slice(*args)), args)

        test(1, 5, -2)
        test(None, 5, -2)
        test(1, None, -2)
        test(1, 5, None)

    def test_create_slice(self):
        @jit
        def f(start, stop, step):
            return slice(start, stop, step)

        def test(*args):
            self.assertEqual(f(*args), slice(*args))

        test(1, 5, -2)
        test(None, 5, -2)
        test(1, None, -2)
        test(1, 5, None)

    def test_slice_compare(self):
        @jit
        def f(s1, s2):
            return s1 == s2

        def test(s1, s2):
            self.assertEqual(f(s1, s2), f.py_func(s1, s2))

        test(slice(10), slice(10))
        test(slice(10), slice(11))
        test(slice(10), None)
        test(None, slice(None))
        test(slice(-4, 5, -2), slice(-4, 5, -2))


if __name__ == '__main__':
    unittest.main()
