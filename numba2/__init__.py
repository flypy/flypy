# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from .compiler import (annotate, overload, overloadable, convert, promote,
                       typeof, typedef)
from .compiler import T, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10
from .runtime import (jit, ijit)
from .errors import error, InferError, SpecializeError