# -*- coding: utf-8 -*-

"""
Foreign function interface functionality.
"""

from __future__ import print_function, division, absolute_import
import math
import cmath

try:
    import __builtin__ as builtins
except ImportError:
    import builtins


from numba2 import jit, overlay
from numba2.types import Complex
from numba2.compiler import lltype
from numba2.runtime.lowlevel_impls import add_impl
from numba2.typing import TypeSet

import numpy as np

# TODO: Move to `typesets` module
#fcomplex = TypeSet(*types.floating | types.complexes, name='fcomplex')

ufloating = 'a : floating -> a'
ucomplex  = 'Complex[a] -> Complex[a]'
bfloating = 'a : floating -> a -> a'
bcomplex  = 'Complex[a] -> Complex[a] -> Complex[a]'

#----------------------------------------------------------------------------
# Symbols
#----------------------------------------------------------------------------

# sin(double), sinf(float), sinl(long double)
mathsyms = [
    'sin',
    'cos',
    'tan',
    'sqrt',
    'acos',
    'asin',
    'atan',
    'sinh',
    'cosh',
    'tanh',
    'asinh',
    'acosh',
    'atanh',
    'log',
    'log2',
    'log10',
    #'erfc',
    'floor',
    'ceil',
    'exp',
    'exp2',
    'expm1',
    'rint',
    'log1p',
]

n_ary_mathsyms = {
    'hypot'     : 2,
    'atan2'     : 2,
    'logaddexp' : 2,
    'logaddexp2': 2,
    'pow'       : (2, 3),
}

math2ufunc = {
    'asin' : 'arcsin',
    'acos' : 'arccos',
    'atan' : 'arctan',
    'asinh': 'arcsinh',
    'acosh': 'arccosh',
    'atanh': 'arctanh',
    'atan2': 'arctan2',
    'pow'  : 'power',
}

ufunc2math = dict((v, k) for k, v in math2ufunc.items())


#===------------------------------------------------------------------===
# Low-level implementations
#===------------------------------------------------------------------===

def declare(name, signatures, mathname=None):
    """
    Declare a unary math function named `name` for the given signatures.
    """
    mathname = mathname or name.title()

    for signature in signatures:
        @jit(signature, opaque=True)
        def func(*args):
            return getattr(math, name)(*args) # pure python

    def impl(builder, argtypes, *args):
        [ty] = argtypes
        lty = lltype(ty)
        return builder.ret(builder.call_math(lty, mathname, list(args)))


    add_impl(func, "numba_" + name, impl)

    func.__name__ = name
    return func

#===------------------------------------------------------------------===
# Math
#===------------------------------------------------------------------===

asin       = declare('asin'    , [ufloating, ucomplex])
cos        = declare('cos'     , [ufloating, ucomplex])
log2       = declare('log2'    , [ufloating, ucomplex])
log        = declare('log'     , [ufloating, ucomplex])
atan       = declare('atan'    , [ufloating, ucomplex])
tanh       = declare('tanh'    , [ufloating, ucomplex])
exp2       = declare('exp2'    , [ufloating, ucomplex])
atanh      = declare('atanh'   , [ufloating, ucomplex])
log1p      = declare('log1p'   , [ufloating, ucomplex])
asinh      = declare('asinh'   , [ufloating, ucomplex])
sqrt       = declare('sqrt'    , [ufloating, ucomplex])
cosh       = declare('cosh'    , [ufloating, ucomplex])
sinh       = declare('sinh'    , [ufloating, ucomplex])
acosh      = declare('acosh'   , [ufloating, ucomplex])
expm1      = declare('expm1'   , [ufloating, ucomplex])
exp        = declare('exp'     , [ufloating, ucomplex])
acos       = declare('acos'    , [ufloating, ucomplex])
log10      = declare('log10'   , [ufloating, ucomplex])
sin        = declare('sin'     , [ufloating, ucomplex])
tan        = declare('tan'     , [ufloating, ucomplex])

abs        = declare('abs'     , ['int32 -> int32', ufloating, ucomplex])
rint       = declare('rint'    , [ufloating, ucomplex])
ceil       = declare('ceil'    , [ufloating, ucomplex])
trunc      = declare('trunc'   , [ufloating, ucomplex])
floor      = declare('floor'   , [ufloating, ucomplex])

pow             = declare('pow'          , [ufloating, ucomplex])
hypot           = declare('hypot'        , [ufloating])
atan2           = declare('atan2'        , [ufloating])
logaddexp       = declare('logaddexp'    , [ufloating])
logaddexp2      = declare('logaddexp2'   , [ufloating])

#===------------------------------------------------------------------===
# Overlays
#===------------------------------------------------------------------===

def math_overlay(mod, getname):
    """Register all functions listed in mathsyms and n_ary_mathsyms"""
    for symname in mathsyms + list(n_ary_mathsyms):
        if hasattr(mod, getname(symname)):
            py_func = getattr(mod, getname(symname))
            nb_func = globals()[symname]
            overlay(py_func, nb_func)

math_overlay(math,  lambda name: name)
math_overlay(cmath, lambda name: name)
math_overlay(np,    lambda name: math2ufunc.get(name, name))