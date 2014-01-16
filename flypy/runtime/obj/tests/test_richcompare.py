# -*- coding: utf-8 -*-

"""
Test rich compare mixin.

Based on http://www.voidspace.org.uk/python/articles/comparison.shtml
"""

from __future__ import print_function, division, absolute_import

import unittest
from flypy import jit

from flypy.runtime.obj.richcompare import RichComparisonMixin

@jit
class RichComparer(RichComparisonMixin):

    layout = [('value', 'int64')]

    @jit
    def __eq__(self, other):
        return self.value == other.value

    @jit
    def __lt__(self, other):
        return self.value < other.value


class RichComparisonMixinTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.comp = RichComparer(6)

    def testDefaultComparison(self):
        self.assertRaises(NotImplementedError,
                          lambda: RichComparisonMixin() == 3)
        self.assertRaises(NotImplementedError,
                          lambda: RichComparisonMixin() != 3)
        self.assertRaises(NotImplementedError,
                          lambda: RichComparisonMixin() < 3)
        self.assertRaises(NotImplementedError,
                          lambda: RichComparisonMixin() > 3)
        self.assertRaises(NotImplementedError,
                          lambda: RichComparisonMixin() <= 3)
        self.assertRaises(NotImplementedError,
                          lambda: RichComparisonMixin() >= 3)

    def testEquality(self):
        self.assertTrue(self.comp == RichComparer(6))
        self.assertFalse(self.comp == RichComparer(7))

    def testInEquality(self):
        self.assertFalse(self.comp != RichComparer(6))
        self.assertTrue(self.comp != RichComparer(7))

    def testLessThan(self):
        self.assertTrue(self.comp < RichComparer(7))
        self.assertFalse(self.comp < RichComparer(5))
        self.assertFalse(self.comp < RichComparer(6))

    def testGreaterThan(self):
        self.assertTrue(self.comp > RichComparer(5))
        self.assertFalse(self.comp > RichComparer(7))
        self.assertFalse(self.comp > RichComparer(6))

    def testLessThanEqual(self):
        self.assertTrue(self.comp <= RichComparer(7))
        self.assertTrue(self.comp <= RichComparer(6))
        self.assertFalse(self.comp <= RichComparer(5))

    def testGreaterThanEqual(self):
        self.assertTrue(self.comp >= RichComparer(5))
        self.assertTrue(self.comp >= RichComparer(6))
        self.assertFalse(self.comp >= RichComparer(7))


if __name__ == '__main__':
    unittest.main()