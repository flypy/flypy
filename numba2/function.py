# -*- coding: utf-8 -*-

"""
Numba function wrapper.
"""

from __future__ import print_function, division, absolute_import

from .compiler import typeof
from .passes import translate

from blaze.util import flatargs

# TODO: Reuse numba.numbawrapper.pyx for autojit Python entry points

class Function(object):
    """
    Result of @jit for functions.
    """

    def __init__(self, py_func, signature, abstract=False):
        self.py_func = py_func
        self.signature = signature
        self.abstract = abstract

        self.llvm_funcs = {}
        self.ctypes_funcs = {}
        self.envs = {}

    def __call__(self, *args, **kwargs):
        if self.signature is not None:
            restype = self.signature.params[0]
            argtypes = self.signature.params[1:]
            cfunc = self.translate(argtypes, restype)
        else:
            args = flatargs(self.py_func, args, kwargs)
            argtypes = [typeof(x) for x in args]
            cfunc = self.translate(argtypes)

        return cfunc(*args)

    def translate(self, argtypes, restype=None):
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

        return llvm_func, env
