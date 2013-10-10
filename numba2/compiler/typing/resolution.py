# -*- coding: utf-8 -*-

"""
Type resolution and method resolution.
"""

from __future__ import print_function, division, absolute_import

from numba2.environment import fresh_env
from numba2 import promote, unify, is_numba_type
from numba2.functionwrapper import FunctionWrapper
from numba2.runtime.type import Type
from numba2.compiler.overloading import flatargs

from pykit import types
from pykit.ir import OpBuilder, Builder, Const, Function, GlobalValue
from pykit.utils import nestedmap

#===------------------------------------------------------------------===
# Function call typing
#===------------------------------------------------------------------===

# TODO: Move this function to a third module

def is_method(t):
    return type(t).__name__ == 'Method' # hargh

def infer_call(func, func_type, arg_types):
    """
    Infer a single call. We have three cases:

        1) Static receiver function
        2) Higher-order function
            This is already typed
        3) Method. We need to insert 'self' in the cartesian product
    """
    from numba2 import phase

    is_const = isinstance(func, Const)
    is_numba_func = is_const and isinstance(func.const, FunctionWrapper)
    is_class = isinstance(func_type, type(Type.type))

    if is_method(func_type) or is_numba_func:
        # -------------------------------------------------
        # Method call or numba function call

        if is_method(func_type):
            func = func_type.parameters[0]
            arg_types = [func_type.parameters[1]] + list(arg_types)
        else:
            func = func.const

        # Translate # to untyped IR and infer types
        # TODO: Support recursion !

        if len(func.overloads) == 1 and not func.opaque:
            arg_types = fill_missing_argtypes(func.py_func, tuple(arg_types))

        env = fresh_env(func, arg_types)
        func, env = phase.typing(func, env)
        return func, env["numba.typing.restype"]

    elif is_class:
        # -------------------------------------------------
        # Constructor application

        restype = func_type.parameters[0]

        # TODO: Instantiate restype with parameter types
        nargs = restype.parameters

        return func, restype

    elif not isinstance(func, Function):
        # -------------------------------------------------
        # Higher-order function

        restype = func_type.restype
        assert restype
        return func, restype

    else:
        raise NotImplementedError(func, func_type)

def get_remaining_args(func, args):
    newargs = flatargs(func, args, {})
    return newargs[len(args):]

def fill_missing_argtypes(func, argtypes):
    from numba2 import typeof

    remaining = get_remaining_args(func, argtypes)
    return argtypes + tuple(typeof(arg) for arg in remaining)

#===------------------------------------------------------------------===
# Type resolution
#===------------------------------------------------------------------===

def resolve_context(func, env):
    """Reduce typesets in context to concrete types"""
    context = env['numba.typing.context']
    for op, typeset in context.iteritems():
        if typeset:
            typeset = context[op]
            context[op] = reduce(promote, typeset)

def resolve_restype(func, env):
    """Figure out the return type and update the context and environment"""
    context = env['numba.typing.context']
    restype = env['numba.typing.restype']

    typeset = context['return']
    inferred_restype = reduce(promote, typeset)

    if restype is None:
        restype = inferred_restype
    elif inferred_restype != restype:
        restype = unify(inferred_restype, restype)

    env['numba.typing.restype'] = restype
