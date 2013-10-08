# -*- coding: utf-8 -*-

"""
Numba passes that perform translation, type inference, code generation, etc.
"""

from __future__ import print_function, division, absolute_import

from numba2.compiler.backend import lltyping, llvm, lowering
from .compiler.frontend import translate, simplify_exceptions
from .compiler import simplification, optimizations as opts, copying, transition
from .compiler.typing import inference
from .compiler.typing.resolution import (resolve_context, resolve_restype,
                                         rewrite_calls, rewrite_constructors,
                                         rewrite_optional_args)
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
    simplification.rewrite_ops,
    simplification.rewrite_overlays,
    cfa,
]

typing = [
    transition.single_copy,
    inference,
    resolve_context,
    resolve_restype,
    rewrite_calls,
    rewrite_constructors,
    rewrite_optional_args,
]

optimizations = [
    dce,
    opts.optimize,
]

backend_init = [
    lltyping,
    lowering.lower_fields,
    llvm.codegen_init,
]

backend_run = [
    llvm.codegen_run,
    llvm.codegen_link,
]

backend_finalize = [
    verify,
    dump_llvm,
    optimize,
    dump_optimized,
    llvm.get_ctypes,
]

passes = frontend + typing + optimizations + backend_init + backend_run

all_passes = [frontend, typing, optimizations, backend_init, backend_run, passes]