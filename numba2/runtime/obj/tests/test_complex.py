# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from numba2 import jit

class TestComplex(unittest.TestCase):

    def test_complex_real(self):
        @jit
        def complex_real(c):
            return c.real

        self.assertEqual(complex_real(123+234j), 123.0)

    def test_complex_imag(self):
        @jit
        def complex_imag(c):
            return c.imag

        self.assertEqual(complex_imag(123+234j), 234.0)

    def test_complex_add(self):
        @jit
        def f(a, b):
            return a + b

        c, d = 123+234j, 321-432j
        self.assertEqual(f(c, d), c + d)

    def test_complex_sub(self):
        @jit
        def f(a, b):
            return a - b

        c, d = 123+234j, 321-432j
        self.assertEqual(f(c, d), c - d)

    def test_complex_mul(self):
        @jit
        def f(a, b):
            return a * b

        c, d = 123+234j, 321-432j
        self.assertEqual(f(c, d), c * d)

    def test_complex_div(self):
        @jit
        def f(a, b):
            return a * b

        c, d = 123+234j, 321-432j
        self.assertEqual(f(c, d), c * d)

    def test_complex_ctor1(self):
        @jit
        def f(real):
            return complex(real)

        self.assertEqual(f(10.0), complex(10.0))

    def test_complex_ctor2(self):
        @jit
        def f(real, imag):
            return complex(real, imag)

        self.assertEqual(f(10.0, 12.0), complex(10.0, 12.0))

    def test_nonzero(self):
        @jit
        def f(c):
            return bool(c)
        self.assertEqual(f(0+0j), False)
        self.assertEqual(f(0+1j), True)
        self.assertEqual(f(1+0j), True)

    def test_tostr(self):
        raise unittest.SkipTest("str.__add__")
        @jit
        def f(x):
            return str(x)

        self.assertEqual(f(1+2j), str(1+2j))


if __name__ == '__main__':
    unittest.main()
