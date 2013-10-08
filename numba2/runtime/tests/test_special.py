# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from numba2 import jit, sizeof

class TestSpecial(unittest.TestCase):

    def test_sizeof(self):
        def func(x):
            return sizeof(x)

        def apply(signature, arg):
            return jit(signature)(func)(arg)

        self.assertEqual(apply('int32 -> int64', 10), 4)
        self.assertEqual(apply('float64 -> int64', 10.0), 8)

@jit
def func(x):
    return sizeof(x)

print (func(10))
#print(func(10.0))

#if __name__ == '__main__':
    #TestSpecial('test_sizeof').debug()
    #unittest.main()