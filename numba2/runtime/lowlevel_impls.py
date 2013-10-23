# -*- coding: utf-8 -*-

"""
Low-level implementations of opaque methods.
"""

from __future__ import print_function, division, absolute_import
import string

from numba2 import representation
from numba2.compiler import opaque
from .interfaces.numbers import Number
from .obj import Pointer

from pykit import ir, types as ptypes

def add_impl(opaque_func, name, implementation, restype=None, restype_func=None):
    def impl(py_func, argtypes):
        # TODO: do this better
        from numba2.compiler.backend.lltyping import ll_type

        ll_argtypes = [ll_type(x) for x in argtypes]
        argnames = string.ascii_letters[:len(argtypes)]

        # Determine return type
        if restype_func:
            result_type = restype_func(argtypes)
        else:
            result_type = restype or ll_argtypes[0]

        type = ptypes.Function(result_type, tuple(ll_argtypes))
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

#===------------------------------------------------------------------===
# Binops
#===------------------------------------------------------------------===

def add_binop(cls, name, restype=None):
    special_name = "__%s__" % name
    impl = lambda b, _, x, y: b.ret(getattr(b, name)(x, y))
    add_impl_cls(cls, special_name, impl, restype)


add_binop(Number, "add")
add_binop(Number, "mul")
add_binop(Number, "sub")
add_binop(Number, "div")
add_binop(Number, "mod")

add_binop(Number, "eq", ptypes.Bool)
add_binop(Number, "ne", ptypes.Bool)
add_binop(Number, "lt", ptypes.Bool)
add_binop(Number, "le", ptypes.Bool)
add_binop(Number, "gt", ptypes.Bool)
add_binop(Number, "ge", ptypes.Bool)

add_binop(Number, "and")
add_binop(Number, "or")
add_binop(Number, "xor")
add_binop(Number, "lshift")
add_binop(Number, "rshift")

#===------------------------------------------------------------------===
# Pointers
#===------------------------------------------------------------------===

def pointer_add(builder, argtypes, ptr, addend):
    builder.ret(builder.ptradd(ptr, addend))

def pointer_load(builder, argtypes, ptr):
    builder.ret(builder.ptrload(ptr))

def pointer_store(builder, argtypes, ptr, value):
    builder.ptrstore(value, ptr)
    builder.ret(None)

# Determine low-level return types

def _getitem_type(argtypes):
    base = argtypes[0].parameters[0]
    return representation.representation_type(base)

# Implement

add_impl_cls(Pointer, "__add__", pointer_add)
add_impl_cls(Pointer, "deref", pointer_load, restype_func=_getitem_type)
add_impl_cls(Pointer, "store", pointer_store, restype=ptypes.Void)