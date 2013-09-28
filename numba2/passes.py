# -*- coding: utf-8 -*-

"""
Numba passes that perform translation, type inference, code generation, etc.
"""

from __future__ import print_function, division, absolute_import

from numba2.compiler.backend import preparation, backend
from .compiler.frontend import translate, simplify_exceptions
from .compiler import simplification
from .compiler.typing import inference
from .compiler.typing.resolution import (resolve_context, resolve_restype,
                                         rewrite_calls)
from .prettyprint import dump, dump_cfg, dump_llvm, dump_optimized

from pykit.analysis import cfa
from pykit.transform import dce
from pykit.codegen.llvm import verify, optimize

#===------------------------------------------------------------------===
# Passes
#===------------------------------------------------------------------===

frontend = [
    translate,
    simplify_exceptions,
    dump_cfg,
    simplification,
    cfa,
]

typing = [
    inference,
    resolve_context,
    resolve_restype,
    rewrite_calls,
]

optimizations = [
    dce,
]

lowering = [
    preparation,
]

backend = [
    backend,
    verify,
    dump_llvm,
    optimize,
    dump_optimized,
]

passes = frontend + typing + optimizations + lowering + backend

all_passes = [frontend, typing, optimizations, lowering, backend, passes]