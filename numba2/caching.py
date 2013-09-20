# -*- coding: utf-8 -*-

"""
All caches used during compilation.

The pipeline can roughly be described as follows:

    llvm codegen ∘ high-level optimizations ∘ inference ∘ bytecode translation

A straightforward model could use a single cache from python functions to
compiled LLVM functions. However, there may be one-to-many relations in the
above composition, so we can share results:

    inference:
        we can use the same constraint graph to derive different typings

    optimizations:
        allow experimentation of different optimizations over (copies of) a
        single typed function

    llvm codegen / lowering:
        We can plug in different exception models, garbage collectors etc.

Even with a simple 1-to-1 mapping between stages, we choose to have a cache
at each stage. This enables modularity and allows us to change the
relationship between stages.
"""

from __future__ import print_function, division, absolute_import


class Cache(object):
    def __init__(self):
        self.cached = {}

    def lookup(self, key):
        return self.cached.get(key)

    def insert(self, key, value):
        self.cached[key] = value

#===------------------------------------------------------------------===
# Bytecode translation
#===------------------------------------------------------------------===

class FrontendCache(Cache):
    """
    Cache for bytecode translation

        py_func -> untyped pykit.ir.Function function
    """

#===------------------------------------------------------------------===
# Type inference
#===------------------------------------------------------------------===

class InferenceCache(object):
    """
    Type inference cache.

    Attributes
    ==========

        typings: { (func, argtypes) : (Context, signature) }
            (func, argtypes) tuple mapping to the fully typed function context
            and signature

        ctxs: { func : [Context] }
            Partial typing contexts, containing type templates similar to
            principal type schemes
    """

    def __init__(self):
        self.typings = {}
        self.ctxs = {}

    def lookup(self, func, argtypes):
        return self.typings.get((func, tuple(argtypes)))

    def lookup_ctx(self, func):
        return self.ctxs.get(func)

#===------------------------------------------------------------------===
# Optimizations
#===------------------------------------------------------------------===

class OptimizationsCache(Cache):
    """Cache for high-level optimization passes"""

#===------------------------------------------------------------------===
# Codegen
#===------------------------------------------------------------------===

class CodegenCache(Cache):
    """Cache for LLVM functions obtained from typed optimized functions"""