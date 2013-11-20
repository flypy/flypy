# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest

from numba2 import jit
from numba2.types import integral

class TestInt(unittest.TestCase):

    def test_int_formatting(self):
        @jit('int32 -> a')
        def f1(x):
            return str(x)

        @jit('int64 -> a')
        def f2(x):
            return str(x)


        for i in range(-300, 300, 100):
            for argtype in ('int32', 'int64'):
                self.assertEqual(f1(i), str(i))
                self.assertEqual(f2(i), str(i))


if __name__ == '__main__':
    unittest.main()