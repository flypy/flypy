# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from flypy import jit
from flypy.runtime.obj.rangeobject import len_range

class TestBuiltins(unittest.TestCase):

    def test_isinstance(self):
        @jit
        def trivial():
            return isinstance(StopIteration(), StopIteration)

        self.assertTrue(trivial())

    def test_len_range(self):
        # Original loop takes too long and it is unncessarily detailed.
        # Should only necessary to check the boundaries and a few random
        # number in between.
        #
        #for start in range(-10, 10):
        #    for stop in range(-11, 11):
        #        for step in range(-12, 12):
        #            if step == 0:
        #                continue
        start_range = [-10, -1, 0, 1, 10]
        stop_range = [-11, -1, 0, 1, 11]
        step_range = [-12, -1, 1, 12]
        for start in start_range:
            for stop in stop_range:
                for step in step_range:
                    self.assertEqual(len_range(start, stop, step),
                                     len(range(start, stop, step)))


if __name__ == '__main__':
    #TestBuiltins('test_isinstance').debug()
    unittest.main()