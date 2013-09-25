# -*- coding: utf-8 -*-

"""
Numba passes that perform translation, type inference, code generation, etc.
"""

from __future__ import print_function, division, absolute_import

from numba2.compiler.backend import preparation, backend
from .compiler.frontend import translate
from .compiler import simplification
from .compiler.typing import inference
from .compiler.typing.resolution import resolve_context, resolve_restype, rewrite_methods
from .prettyprint import dump, dump_cfg, dump_llvm, dump_optimized

from pykit.analysis import cfa
from pykit.transform import dce
from pykit.codegen.llvm import verify, optimize

#===------------------------------------------------------------------===
# Passes
#===------------------------------------------------------------------===

frontend = [
    translate,
    dump_cfg,
    simplification,
    cfa,
]

typing = [
    inference,
    resolve_context,
    resolve_restype,
]

lower = [
    rewrite_methods,
]

optimize = [
    dce,
]

backend = [
    preparation,
    backend,
    verify,
    dump_llvm,
    optimize,
    dump_optimized,
]

passes = frontend + typing + lower + backend