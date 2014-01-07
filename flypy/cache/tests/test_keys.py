# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from flypy import int32, float64, jit
from flypy.cache import keys
from flypy.cache.tests import mod1, mod2


class TestKeys(unittest.TestCase):

    def test_keys(self):
        def f(x, y):
            print(x, y)

        blob1 = keys.make_code_blob(f, (int32, float64))
        blob2 = keys.make_code_blob(f, (int32, float64))

        self.assertEqual(blob1, blob2)

    def test_key_argtypes(self):
        def f(x):
            print(x)

        blob1 = keys.make_code_blob(f, (mod1.Foo[()],))
        blob2 = keys.make_code_blob(f, (mod2.Foo[()],))

        self.assertNotEqual(blob1, blob2)

if __name__ == '__main__':
    unittest.main()