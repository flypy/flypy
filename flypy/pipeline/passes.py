# -*- coding: utf-8 -*-

"""
flypy passes that perform translation, type inference, code generation, etc.
"""

from __future__ import print_function, division, absolute_import
from functools import partial

from flypy.compiler.backend import lltyping, llvm, lowering, rewrite_lowlevel_constants
from flypy.compiler.frontend import (translate, simplify_exceptions, checker, setup)
from flypy.compiler.backend import (lltyping, llvm, lowering,
                                     rewrite_lowlevel_constants)
from flypy.compiler.analysis import dependence_analysis
from flypy.compiler import simplification, transition
from flypy.compiler.typing import inference, typecheck
from flypy.compiler.typing.resolution import (resolve_context, resolve_restype)
from flypy.compiler.optimizations import (dataflow, optimize, inlining,
                                           throwing, deadblocks, reg2mem)
from flypy.compiler.lower import (rewrite_calls, rewrite_raise_exc_type,
                                   rewrite_getattr, rewrite_setattr,
                                   rewrite_unpacking, rewrite_varargs,
                                   rewrite_constructors, explicit_coercions,
                                   rewrite_optional_args, rewrite_constants,
                                   conversion, rewrite_obj_return, allocator,
                                   rewrite_externs, generators)
from flypy.viz.prettyprint import dump, dump_cfg, dump_llvm, dump_optimized

from pykit.transform import dce
#from pykit.optimizations import local_exceptions
from pykit.codegen.llvm import (verify, optimize as llvm_optimize,
                                llvm_postpasses)

#===------------------------------------------------------------------===
# Passes
#===------------------------------------------------------------------===

initialize = [
    setup,
]

frontend = [
    translate,
    simplify_exceptions,
    dump_cfg,
    simplification.rewrite_ops,
    simplification.rewrite_overlays,
    deadblocks,
    dataflow,
    checker,
]

typing = [
    # flypy.compiler.typing.*
    transition.single_copy,
    inference,
    resolve_context,
    resolve_restype,
    typecheck,
    # flypy.compiler.lower.*
    rewrite_getattr,
    rewrite_setattr,
    rewrite_calls,
    rewrite_unpacking,
    rewrite_varargs,
    rewrite_raise_exc_type,
    reg2mem,
]

generators = [
    generators.generator_fusion,             # generators
    #generators.rewrite_general_generators,  # generators

]

hl_lowering = [
    rewrite_constructors,                   # constructors
    allocator,                              # allocation
    rewrite_optional_args,
    explicit_coercions,
    conversion,
    rewrite_externs,
    rewrite_constants,
    rewrite_obj_return,
    dependence_analysis,
]

optimizations = [
    dce,
    dataflow.dataflow,
    optimize,
    dependence_analysis,
]

prelowering = [
    lltyping,
    dependence_analysis,
]

ll_lowering = [
    inlining,
    dataflow,
    throwing.rewrite_local_exceptions,
    rewrite_lowlevel_constants,
    #lowering.lower_fields,
    dependence_analysis,
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
]

codegen = [
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

all_passes = [
    frontend, typing, generators, hl_lowering, optimizations,
    prelowering,
    ll_lowering, backend_init, backend_run, backend_finalize,
    codegen, dpp_backend_run, dpp_backend_finalize,
]
passes = sum(all_passes, [])