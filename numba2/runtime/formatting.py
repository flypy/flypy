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
# Formatters
#===------------------------------------------------------------------===

@jit
def sprintf(buf, fmt, x):
    """
    Print `x` to `buf` according to `format`.

    Returns the number of characters written.
    """
    fmt = numba2.runtime.as_cstring(fmt)
    n = len(buf)
    result = numba2.libc.snprintf(buf.pointer(), n, fmt, x)
    #if result >= n:
    #    raise ValueError("Unable to print to buffer:")
    return result