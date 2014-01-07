# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest

from flypy import jit
from flypy.types import integral

class TestInt(unittest.TestCase):

    def test_tostr(self):
        @jit('int32 -> a')
        def f1(x):
            return str(x)

        @jit('int64 -> a')
        def f2(x):
            return str(x)


        for i in range(-300, 300, 100):
            self.assertEqual(f1(i), str(i))
            self.assertEqual(f2(i), str(i))

    def test_bool(self):
        @jit
        def f(x):
            return bool(x)
        self.assertEqual(f(0), False)
        self.assertEqual(f(1), True)
        self.assertEqual(f(10), True)


if __name__ == '__main__':
    unittest.main()
