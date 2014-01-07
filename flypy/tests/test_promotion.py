# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from flypy import jit

@jit('a -> a -> a')
def mymax(a, b):
    if a > b:
        result = a
    else:
        result = b
    return result

class TestPromotion(unittest.TestCase):

    def test_type_promotion(self):
        self.assertEqual(mymax(5, 10), 10)
        self.assertEqual(mymax(9, 10.0), 10.0)

if __name__ == '__main__':
    #TestPromotion('test_type_promotion').debug()
    unittest.main()