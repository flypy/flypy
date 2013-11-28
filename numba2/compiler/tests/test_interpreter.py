# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from numba2 import jit
from numba2.pipeline import phase
from numba2.compiler import interpreter

@jit
def f(x, start, stop, step):
    for i in range(start, stop, step):
        x += i
    return x

args1   = (5, 2, 10, 3)
result1 = f.py_func(*args1)

args2   = (5.0, 2, 10, 3)
result2 = f.py_func(*args2)

#===------------------------------------------------------------------===
# Tests
#===------------------------------------------------------------------===

class TestInterpreter(unittest.TestCase):

    def interp(self, phase):
        result = interpreter.interpret(f, phase, args1)
        self.assertEqual(result, result1)

        #result = interpreter.interpret(f, phase, args2)
        #self.assertEqual(result, result2)

    def test_interp_frontend(self):
        self.interp(phase.translation)

    def test_interp_typed(self):
        self.interp(phase.typing)

    def test_interp_optimized(self):
        self.interp(phase.opt)

    def test_interp_lowered(self):
        self.interp(phase.lower)


if __name__ == '__main__':
    unittest.main()