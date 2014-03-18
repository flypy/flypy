# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import os
import unittest
import tempfile

from flypy import jit
from flypy.cppgen import cppgen

@jit('C[x, y]')
class C(object):
    layout = [('a', 'x'), ('b', 'y')]

    @jit('C[x, y] -> x')
    def first(self):
        return self.a

    @jit('C[x, y] -> y')
    def scnd(self):
        return self.b

#===------------------------------------------------------------------===
# Tests
#===------------------------------------------------------------------===

class TestCPPGen(unittest.TestCase):

    def test_cppgen(self):
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".cpp") as f:
            # TODO: Remove compiled code
            # TODO: Portable compilation
            cppgen.generate(C, f.write)
            os.system("g++ -g -Wall -c %s" % (f.name,))


if __name__ == '__main__':
    unittest.main()