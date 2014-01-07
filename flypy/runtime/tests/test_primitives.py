# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from flypy import jit

class TestPrimitives(unittest.TestCase):

    def test_is_none(self):
        @jit
        def f(x):
            return x is None

        self.assertEqual(f(None), True)
        self.assertEqual(f(10), False)


if __name__ == '__main__':
    unittest.main()