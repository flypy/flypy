# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from flypy.unification import (TypeConstructor, Type, tvars, typejoin,
                               issubtype, unify, join_constructors)

a, b, c = tvars('a', 'b', 'c')

def maketype(name, parent=object, params=()):
    class Cls(parent):
        pass
    Cls.__name__ = name
    return TypeConstructor(Cls, params)

A = maketype('A')
B = maketype('B', A)
C = maketype('C', B)
D = maketype('D', B)


class TestSubtyping(unittest.TestCase):

    def test_join(self):
        self.assertEqual(join_constructors(A, A), A)
        self.assertEqual(join_constructors(A, B), A)
        self.assertEqual(join_constructors(C, A), A)
        self.assertEqual(join_constructors(B, C), B)
        self.assertEqual(join_constructors(C, D), B)


if __name__ == '__main__':
    unittest.main()