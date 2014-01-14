# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest
from flypy import jit

# ______________________________________________________________________

class TestLazySignatureParser(unittest.TestCase):

    def test_lazy_function_signatures(self):
        @jit('complete and utter garbage :))')
        def f(x):
            return x

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main(verbosity=3)