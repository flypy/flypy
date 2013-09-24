# -*- coding: utf-8 -*-

"""
Numba function wrapper.
"""

from __future__ import print_function, division, absolute_import

from .typing import typeof

from blaze.util import flatargs

# TODO: Reuse numba.numbawrapper.pyx for autojit Python entry points

class FunctionWrapper(object):
    """
    Result of @jit for functions.
    """

    def __init__(self, py_func, signature, abstract=False, opaque=False):
        self.py_func = py_func
        self.signature = signature
        self.abstract = abstract

        self.llvm_funcs = {}
        self.ctypes_funcs = {}
        self.envs = {}

        self.opaque = opaque
        self.implementor = None

    def __call__(self, *args, **kwargs):
        if self.signature is not None:
            restype = self.signature.params[0]
            argtypes = self.signature.params[1:]
            cfunc, lfunc, env = self.translate(argtypes, restype)
        else:
            args = flatargs(self.py_func, args, kwargs)
            argtypes = [typeof(x) for x in args]
            cfunc, lfunc, env = self.translate(argtypes)

        return cfunc(*args)

    def translate(self, argtypes, restype=None):
        from .passes import translate

        key = tuple(argtypes) + (restype,)
        if key in self.ctypes_funcs:
            return self.ctypes_funcs[key]

        # Translate
        llvm_func, env = translate(self.py_func, argtypes)
        cfunc = env["codegen.llvm.ctypes"]

        # Cache
        self.llvm_funcs[key] = llvm_func
        self.ctypes_funcs[key] = cfunc
        self.envs[key] = env

        return cfunc, llvm_func, env

    def __str__(self):
        return "<%s: %s>" % (self.py_func, self.signature)