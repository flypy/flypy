# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from flypy import jit

class TestCallingConventionFromPython(unittest.TestCase):

    def test_varargs(self):
        @jit
        def f(a, b, *args):
            return [a, b, args[1]]

        self.assertEqual(f(1, 2, 0, 3, 0), [1, 2, 3])


class TestCallingFlypyConvention(unittest.TestCase):

    def test_varargs(self):
        @jit
        def g(a, b, *args):
            return [a, b, args[1]]

        @jit
        def f(a, b, c, d, e):
            return g(a, b, c, d, e)

        self.assertEqual(f(1, 2, 0, 3, 0), [1, 2, 3])


if __name__ == '__main__':
    unittest.main()