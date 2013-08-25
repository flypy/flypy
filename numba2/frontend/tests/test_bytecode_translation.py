# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import dis
import unittest

from numba2.frontend import translate, interpret

def run(f, expected, args):
    code = translate(f)
    result = interpret(code, args=args)
    assert result == expected, "Got %s, expected %s" % (result, expected)

class TestBytecodeTranslation(unittest.TestCase):

    def test_compare(self):
        run(lambda a, b: a < b, True, [5, 6])

    def test_binop(self):
        run(lambda a, b: a + b, 11, [5, 6])

    def test_while(self):
        def f(a, b):
            while a < b:
                a = a + 1
            return a + b

        # print(translate(f))
        run(f, 20, [0, 10])

    def test_for(self):
        def f(a, b):
            sum = 0
            for i in range(a, b):
                sum += i
            return sum

        # dis.dis(f)
        # print(translate(f))
        run(f, 45, [0, 10])

if __name__ == '__main__':
    # TestBytecodeTranslation('test_for').debug()
    unittest.main()