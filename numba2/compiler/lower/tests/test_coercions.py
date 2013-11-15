# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from numba2 import jit, ijit

#===------------------------------------------------------------------===
# Tests
#===------------------------------------------------------------------===

class TestCoercions(unittest.TestCase):

    def test_coerce_phi(self):
        @jit
        def f(x):
            if x > 2:
                y = 3
            else:
                y = 2.0
            return y

        #self.assertEqual(f(8), 3.0)
        #self.assertEqual(f(1), 2.0)

    def test_coerce_application(self):
        @jit('int64 -> float32 -> a')
        def g(a, b):
            return a + b

        @jit
        def f():
            x = 1
            return g(x, x) # convert single constant '1' to int64 and float32

        #self.assertEqual(f(), 2.0)

    def test_coerce_setfield(self):
        @jit
        class C(object):
            layout = [('x', 'float32')]

        @jit
        def f(x):
            return C(x).x

        self.assertEqual(f(5), 5.0)

    def test_coerce_ret(self):
        @jit('int32 -> float32')
        def f(x):
            return x

        #self.assertEqual(f(2), 2.0)


if __name__ == '__main__':
    unittest.main()