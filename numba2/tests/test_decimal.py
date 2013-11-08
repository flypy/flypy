# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba2 import jit, Decimal
import cdecimal

@jit
def test():
    result = Decimal('0.0')
    for i in range(10000):
        x = Decimal('1.0')
        result = result + x
    return result

def test2():
    result = cdecimal.Decimal('0.0')
    for i in range(10000):
        x = cdecimal.Decimal('1.0')
        result = result + x
    return result

test()
test2()

import timeit
print(timeit.timeit('test()', 'from __main__ import test', number=3))
print(timeit.timeit('test2()', 'from __main__ import test2', number=3))
