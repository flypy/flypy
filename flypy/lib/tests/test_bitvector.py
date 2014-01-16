# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from flypy import jit
from flypy.lib.bitvector import BitVector

class TestBitVector(unittest.TestCase):

    def test_bitvector(self):
        @jit
        def f(n):
            v = BitVector(100)
            set(v)
            return check(v)

        @jit
        def set(v):
            for i in range(20):
                if i % 7 < 4:
                    v.mark(i)

        @jit
        def check(v):
            for i in range(20):
                if i % 7 < 4:
                    if i not in v:
                        return i
                else:
                    if i in v:
                        return i

            return -1

        self.assertEqual(f(100), -1)


if __name__ == '__main__':
    unittest.main(verbosity=3)
