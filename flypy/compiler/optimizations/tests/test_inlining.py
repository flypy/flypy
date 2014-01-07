# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from flypy import jit, ijit

#===------------------------------------------------------------------===
# Tests
#===------------------------------------------------------------------===

class TestInlining(unittest.TestCase):

    def test_inline_simple(self):
        @ijit
        def g(x):
            return x * 2
        @ijit
        def f(x):
            return g(x) + 2

        self.assertEqual(f(8), 18)

if __name__ == '__main__':
    unittest.main()