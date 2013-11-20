# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest

from numba2 import jit

class TestBool(unittest.TestCase):

    def test_tostr(self):
        @jit
        def f(x):
            return str(x)
        self.assertEqual(f(False), "False")
        self.assertEqual(f(True), "True")


if __name__ == '__main__':
    unittest.main()