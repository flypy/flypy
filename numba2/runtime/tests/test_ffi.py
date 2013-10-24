# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import math
import unittest

from numba2 import jit, types
from numba2.runtime import ffi

# ______________________________________________________________________

class TestFFI(unittest.TestCase):

    def test_malloc(self):
        @jit
        def f():
            p = ffi.malloc(2, types.int32)
            p[0] = 4
            p[1] = 5
            return p

        p = f()
        self.assertEqual(p[0], 4)
        self.assertEqual(p[1], 5)

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()