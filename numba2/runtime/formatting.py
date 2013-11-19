# -*- coding: utf-8 -*-

"""
String formatting functionality for some primitive types. We do this since
it depends on several object implementations at once (e.g. Buffer and String),
which themselves need say, integers.
"""

from __future__ import print_function, division, absolute_import
import math

import numba2
from numba2 import jit

#===------------------------------------------------------------------===
# Utils
#===------------------------------------------------------------------===

@jit #('Buffer[char] -> String[] -> ... -> String[]') # TODO: varargs
def ll_format(buf, format, *args):
    return numba2.libc.snprintf(buf.pointer(), len(buf), format, *args)

#===------------------------------------------------------------------===
# Formatters
#===------------------------------------------------------------------===

@jit #('a -> String[]')
def int_format(self):
    ndigits = int(math.ceil(math.log(self)))
    buf = numba2.newbuffer(numba2.char, ndigits)
    _int_format(buf, numba2.cast(self, numba2.longlong))
    return numba2.String(buf)

@jit('a -> b : integral -> void') # 'Int[a, False] -> void'
def _int_format(buf, x):
    fmt = "%ll".buf.pointer()
    numba2.libc.snprintf(buf.pointer(), len(buf), fmt, x)

# TODO: unsigned!