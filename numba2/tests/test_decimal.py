# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import numba2
from numba2 import jit
from numba2.lib.extended import decimal
import cdecimal


@jit
def test():
    result = decimal('0.0')
    for i in range(100000):
        x = decimal('1.0')
        result = result + x
    print(result)


@jit
def test2():
    d1 = decimal('1.0')
    d2 = decimal('2.0')
    d3 = decimal('3.0')
    d4 = decimal('4.0')
    d5 = decimal('5.0')
    print(d1 + d2 + d3 + d4 + d5)


@jit
def calculate_mandelbrot(real, imag):
    c_real = real
    c_imag = imag

    for i in range(1000):
        real = (real * real) - (imag * imag) + c_real
        imag = (real * imag) + (real * imag) + c_imag
        if (real * real) + (imag * imag) > decimal('4.0'):
            return False
    return True

@jit
def test_mandelbrot():
    t0 = numba2.bltins.clock()
    calculate_mandelbrot(decimal('0.1'), decimal('0.1'))
    t1 = numba2.bltins.clock()
    return t1 - t0


'''def calculate_mandelbrot2(real, imag):
    c_real = real
    c_imag = imag

    for i in range(1000):
        real = (real * real) - (imag * imag) + c_real
        imag = (real * imag) + (real * imag) + c_imag
        if (real * real) + (imag * imag) > cdecimal.Decimal('4.0'):
            return False
    return True

def test_mandelbrot2():
    t0 = numba2.bltins.clock()
    calculate_mandelbrot2(cdecimal.Decimal('0.1'), cdecimal.Decimal('0.1'))
    t1 = numba2.bltins.clock()
    return t1 - t0'''


#import timeit
#print(timeit.timeit('test()', 'from __main__ import test', number=3))

print(test_mandelbrot())
#print(test_mandelbrot2())
