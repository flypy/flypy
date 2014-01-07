# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from numba2 import jit, typeof, int32, float64
from numba2.runtime.obj.core import make_variant

int_or_float = make_variant(int32, float64)

class TestVariant(unittest.TestCase):

    def test_py_variant(self):
        raise unittest.SkipTest("Variants are not supported yet")
        v = int_or_float(0, 2, 0.0)
        self.assertEqual(v + v, 4)


if __name__ == '__main__':
    unittest.main()