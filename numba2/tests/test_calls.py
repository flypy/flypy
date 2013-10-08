# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from numba2 import jit

class TestCalls(unittest.TestCase):

    def test_static_call(self):
        @jit
        def g(a):
            return a + 2
        @jit
        def f(a):
            return g(a * 3)

        self.assertEqual(f(5), 17)

    def test_optional_args(self):
        @jit
        def g(a=4):
            return a + 2
        @jit
        def f():
            return g() + 1

        self.assertEqual(f(), 7)

    def test_optional_args_from_python(self):
        @jit
        def f(a=4):
            return a + 1

        self.assertEqual(f(), 5)


if __name__ == '__main__':
    #TestCalls("test_optional_args").debug()
    unittest.main()
