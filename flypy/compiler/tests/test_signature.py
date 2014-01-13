# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from flypy.compiler.signature import flatargs

class TestArgParse(unittest.TestCase):

    def test_empty(self):
        def f():
            assert False

        self.assertEqual(flatargs(f, (), {}), ())
        self.assertRaises(TypeError, flatargs, f, (1,), {})
        self.assertRaises(TypeError, flatargs, f, (), {'a': 10})

    def test_empty_varargs(self):
        def f(*args):
            assert False

        self.assertEqual(flatargs(f, (), {}), ((),))
        self.assertEqual(flatargs(f, (1, 2, 3), {}), ((1, 2, 3),))
        self.assertRaises(TypeError, flatargs, f, (), {'a': 10})

    def test_empty_keywords(self):
        def f(**kwargs):
            assert False

        self.assertEqual(flatargs(f, (), {}), ({},))
        self.assertEqual(flatargs(f, (), {'a': 10}), ({'a': 10},))
        self.assertRaises(TypeError, flatargs, f, (1, 2, 3), {})

    def test_args(self):
        def f(a, b):
            assert False

        self.assertEqual(flatargs(f, (1, 2), {}), (1, 2))
        self.assertEqual(flatargs(f, (1,), {'b': 2}), (1, 2))
        self.assertEqual(flatargs(f, (), {'a': 1, 'b': 2}), (1, 2))

        self.assertRaises(TypeError, flatargs, f, (1, 2, 3), {})
        self.assertRaises(TypeError, flatargs, f, (1,), {})
        self.assertRaises(TypeError, flatargs, f, (1, 2), {'b': 3})
        self.assertRaises(TypeError, flatargs, f, (), {'a': 1, 'b': 2, 'c': 3})

    def test_defaults(self):
        def f(a, b=3):
            assert False

        self.assertEqual(flatargs(f, (1, 2), {}), (1, 2))
        self.assertEqual(flatargs(f, (1,), {'b': 2}), (1, 2))
        self.assertEqual(flatargs(f, (), {'a': 1, 'b': 2}), (1, 2))
        self.assertEqual(flatargs(f, (), {'a': 1}), (1, 3))
        self.assertEqual(flatargs(f, (1,), {}), (1, 3))

        self.assertRaises(TypeError, flatargs, f, (1, 2, 3), {})
        self.assertRaises(TypeError, flatargs, f, (), {})
        self.assertRaises(TypeError, flatargs, f, (1, 2), {'b': 3})
        self.assertRaises(TypeError, flatargs, f, (), {'a': 1, 'b': 2, 'c': 3})

    def test_all_the_above(self):
        def f(a, b=3, *args, **kwargs):
            assert False

        self.assertEqual(flatargs(f, (1,), {}), (1, 3, (), {}))
        self.assertEqual(flatargs(f, (1, 2), {}), (1, 2, (), {}))
        self.assertEqual(flatargs(f, (1,), {'b': 2}), (1, 2, (), {}))
        self.assertEqual(flatargs(f, (1, 2, 3, 4), {'d': 4}),
                         (1, 2, (3, 4), {'d': 4}))

        self.assertRaises(TypeError, flatargs(f, (1, 2, 3), {'b': 2, 'd': 4}))


if __name__ == '__main__':
    unittest.main()