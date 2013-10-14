# -*- coding: utf-8 -*-

"""
Numba passes that perform translation, type inference, code generation, etc.
"""

from __future__ import print_function, division, absolute_import

from numba2.compiler.backend import lltyping, llvm, lowering, rewrite_lowlevel_constants
from .compiler.frontend import translate, simplify_exceptions
from .compiler import simplification, transition
from .compiler.typing import inference, typecheck
from .compiler.typing.resolution import (resolve_context, resolve_restype)
from .compiler.optimizations import optimize, inliner
from .compiler.lower import (rewrite_calls, rewrite_raise_exc_type,
                             rewrite_constructors,
                             rewrite_optional_args, rewrite_constants,
                             convert_retval)
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
    typecheck,
    rewrite_calls,
    rewrite_raise_exc_type,
    rewrite_constructors,
    rewrite_optional_args,
    rewrite_constants,
    convert_retval,
]

optimizations = [
    dce,
    inliner,
    cfa,
    optimize,
]

backend_init = [
    lltyping,
    rewrite_lowlevel_constants,
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