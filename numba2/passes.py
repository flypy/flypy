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
from .compiler.optimizations import optimize, inliner, throwing, sccp
from .compiler.lower import (rewrite_calls, rewrite_raise_exc_type,
                             rewrite_constructors,
                             rewrite_optional_args, rewrite_constants,
                             convert_retval)
from .prettyprint import dump, dump_cfg, dump_llvm, dump_optimized

from pykit.analysis import cfa
from pykit.transform import dce
#from pykit.optimizations import local_exceptions
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
    sccp,
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
    #cfa,
    optimize,
    lltyping,
]

lowering = [
    inliner,
    cfa,
    throwing.rewrite_local_exceptions,
    rewrite_lowlevel_constants,
    lowering.lower_fields,
]

backend_init = [
    throwing.rewrite_exceptions,
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

all_passes = [frontend, typing, optimizations, lowering,
              backend_init, backend_run]
passes = sum(all_passes, [])
