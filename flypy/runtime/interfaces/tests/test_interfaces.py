# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from flypy import abstract, jit, parse

class TestInterfaces(unittest.TestCase):

    def test_interface(self):
        @abstract('Int[X, Y]')
        class Int(object):
            layout = []

        self.assertEqual(str(Int.type), 'Int[X, Y]')
        self.assertEqual(str(Int[32, True]), 'Int[32, True]')


if __name__ == '__main__':
    # TestInterfaces('test_interface').debug()
    unittest.main()