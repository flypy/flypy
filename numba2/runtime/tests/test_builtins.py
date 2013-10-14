# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from numba2 import jit
from numba2.runtime.builtins import len_range

class TestBuiltins(unittest.TestCase):

    def test_isinstance(self):
        @jit
        def trivial():
            return isinstance(StopIteration(), StopIteration)

        self.assertTrue(trivial())

    def test_len_range(self):
        for start in range(-10, 10):
            for stop in range(-11, 11):
                for step in range(-12, 12):
                    if step == 0:
                        continue
                    self.assertEqual(len_range(start, stop, step),
                                     len(range(start, stop, step)))


if __name__ == '__main__':
    TestBuiltins('test_isinstance').debug()
    #unittest.main()