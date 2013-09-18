# -*- coding: utf-8 -*-

"""
Numba compilation environment.
"""

from __future__ import print_function, division, absolute_import

from .utils import FrozenDict
from .caching import (FrontendCache, InferenceCache, OptimizationsCache,
                      CodegenCache)

root_env = FrozenDict({
    # Caching
    'numba.frontend.cache': FrontendCache(),
    'numba.typing.cache':   InferenceCache(),
    'numba.opt.cache':      OptimizationsCache(),
    'numba.codegen.cache':  CodegenCache(),

    # Typing
    'numba.typing.restype': None,       # Input
    'numba.typing.argtypes': None,      # Input
    'numba.typing.signature': None,     # Output
    'numba.typing.context': None,       # Output
    'numba.typing.constraints': None,   # Output

    # Flags
    'numba.verify':         True,
    'numba.optimize':       True,

    # Codegen
    "codegen.llvm.opt":     None,
    "codegen.llvm.engine":  None,
    "codegen.llvm.module":  None,
    "codegen.llvm.machine": None,
    "codegen.llvm.ctypes":  None,
})