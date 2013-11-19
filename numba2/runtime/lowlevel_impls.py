# -*- coding: utf-8 -*-

"""
Low-level implementations of opaque methods.
"""

from __future__ import print_function, division, absolute_import
import string

from numba2.compiler import opaque
from pykit import ir, types as ptypes


def add_impl(opaque_func, name, implementation, restype=None, restype_func=None):
    """
    Assign an implementation to an `opaque` function.

    Sets up a pykit function and calls `implementation` to produce the
    function body.
    """
    def impl(py_func, argtypes):
        # TODO: do this better
        from numba2.compiler import representation_type

        ll_argtypes = [representation_type(x) for x in argtypes]
        argnames = list(string.ascii_letters[:len(argtypes)])

        # Determine return type
        if restype_func:
            result_type = restype_func(argtypes)
        else:
            result_type = restype or ll_argtypes[0]

        type = ptypes.Function(result_type, tuple(ll_argtypes), False)
        func = ir.Function(name, argnames, type)
        func.new_block("entry")
        b = ir.Builder(func)
        b.position_at_beginning(func.startblock)
        implementation(b, argtypes, *func.args)
        return func

    opaque.implement_opaque(opaque_func, impl)


def add_impl_cls(cls, name, implementation, restype=None, restype_func=None):
    opaque_func = getattr(cls, name)
    add_impl(opaque_func, name, implementation, restype, restype_func)
