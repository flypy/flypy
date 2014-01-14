# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest

from flypy import jit, debugprint

class TestDebugPrint(unittest.TestCase):

    def test_debugprint(self):
        @jit
        def f(x):
            return x
        @jit
        def test(a, b):
            debugprint(a + b)

        test(10, 12)


if __name__ == "__main__":
    unittest.main()