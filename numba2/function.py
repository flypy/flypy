# -*- coding: utf-8 -*-

"""
Numba function wrapper.
"""

from __future__ import print_function, division, absolute_import

from .compiler import typeof
from .passes import translate, passes
from .environment import root_env

from blaze.util import flatargs

# TODO: Reuse numba.numbawrapper.pyx for autojit Python entry points

class Function(object):
    """
    Result of @jit for functions.
    """

    def __init__(self, py_func, signature):
        self.py_func = py_func
        self.signature = signature

    def translate(self, args, kwargs, env=root_env):
        args = flatargs(self.py_func, args, kwargs)
        argtypes = [typeof(x) for x in args]
        env = dict(env)
        llvm_func, env = translate(self.py_func, argtypes, env, passes)
        return llvm_func, env

    def __call__(self, *args, **kwargs):
        lfunc, env = self.translate(args, kwargs)
        cfunc = env["codegen.llvm.ctypes"]
        return cfunc(*args)