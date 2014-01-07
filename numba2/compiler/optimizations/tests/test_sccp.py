# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from numba2 import jit, ijit

#===------------------------------------------------------------------===
# Tests
#===------------------------------------------------------------------===

class TestSCCP(unittest.TestCase):

    def test_sccp_simple(self):
        @jit(specialize_value=('x',))
        def f(x):
            if x > 2:
                return x + 2
            else:
                return x + 3


if __name__ == '__main__':
    unittest.main()