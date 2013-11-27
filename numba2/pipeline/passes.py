# -*- coding: utf-8 -*-

"""
Numba passes that perform translation, type inference, code generation, etc.
"""

from __future__ import print_function, division, absolute_import

from numba2.compiler.backend import lltyping, llvm, lowering, rewrite_lowlevel_constants
from numba2.compiler.frontend import translate, simplify_exceptions
from numba2.compiler import simplification, transition
from numba2.compiler.typing import inference, typecheck
from numba2.compiler.typing.resolution import (resolve_context, resolve_restype)
from numba2.compiler.optimizations import optimize, inliner, throwing
from numba2.compiler.lower import (rewrite_calls, rewrite_raise_exc_type,
                                   rewrite_constructors, explicit_coercions,
                                   rewrite_optional_args, rewrite_constants,
                                   convert_retval, rewrite_obj_return, allocator,
                                   rewrite_externs)
from numba2.viz.prettyprint import dump, dump_cfg, dump_llvm, dump_optimized

from pykit.analysis import cfa
from pykit.transform import dce
#from pykit.optimizations import local_exceptions
from pykit.codegen.llvm import (verify, optimize as llvm_optimize,
                                llvm_postpasses)

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
    # numba.compiler.typing.*
    transition.single_copy,
    inference,
    resolve_context,
    resolve_restype,
    typecheck,
    # numba.compiler.lower.*
    rewrite_calls,
    rewrite_raise_exc_type,
    rewrite_constructors,
    allocator,
    rewrite_optional_args,
    explicit_coercions,
    rewrite_externs,
    rewrite_constants,
    rewrite_obj_return,
    convert_retval,
]

optimizations = [
    dce,
    #cfa,
    optimize,
]

prelowering = [
    lltyping,
]

lowering = [
    inliner,
    cfa,
    throwing.rewrite_local_exceptions,
    rewrite_lowlevel_constants,
    #lowering.lower_fields,
]

backend_init = [
    throwing.rewrite_exceptions,
    llvm.codegen_init,
]

backend_run = [
    llvm.codegen_run,
    llvm_postpasses,
    llvm.codegen_link,
]

backend_finalize = [
    verify,
    dump_llvm,
    llvm_optimize,
    dump_optimized,
    llvm.get_ctypes,
]

dpp_backend_run = [
    llvm.codegen_run,
    # llvm_postpasses,  # for math
    #llvm.codegen_link, # do nothing
]

dpp_backend_finalize = [
    verify,
    dump_llvm,
]

all_passes = [frontend, typing, optimizations, lowering,
              backend_init, backend_run]
passes = sum(all_passes, [])

