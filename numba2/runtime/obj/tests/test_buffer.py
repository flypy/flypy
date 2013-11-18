# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import ctypes
import unittest

from numba2 import jit, int32
from numba2.runtime.obj.bufferobject import Buffer, fromseq

class TestPointers(unittest.TestCase):

    def test_buffer(self):
        @jit
        def f(b):
            return b[1]

        b = fromseq([1, 2, 3], int32)
        self.assertEqual(f(b), 2)


if __name__ == '__main__':
    unittest.main()