# -*- coding: utf-8 -*-

"""
Type resolution and method resolution.
"""

from __future__ import print_function, division, absolute_import

from numba2.environment import fresh_env
from numba2.typing import promote, unify_simple
from numba2.functionwrapper import FunctionWrapper

from pykit.ir import OpBuilder, Const, Function

#===------------------------------------------------------------------===
# Function call typing
#===------------------------------------------------------------------===

# TODO: Move this function to a third module

def infer_call(func, func_type, arg_types):
    """
    Infer a single call. We have three cases:

        1) Static receiver function
        2) Higher-order function
            This is already typed
        3) Method. We need to insert 'self' in the cartesian product
    """
    is_method = type(func_type).__name__ == 'Method'
    is_const = isinstance(func, Const)
    is_numba_func = is_const and isinstance(func.const, FunctionWrapper)

    if is_method or is_numba_func:
        # -------------------------------------------------
        # Method call or numba function call

        from numba2 import phase

        if is_method:
            func = func_type.parameters[0]
        else:
            func = func.const

        # Translate # to untyped IR and infer types
        # TODO: Support recursion !

        env = fresh_env(func, arg_types)
        func, env = phase.typing(func, env)
        return func, env["numba.typing.restype"]

    elif not isinstance(func, Function):
        # -------------------------------------------------
        # Higher-order function

        restype = func_type.restype
        assert restype
        return func, restype

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
        restype = unify_simple(inferred_restype, restype)

    env['numba.typing.restype'] = restype

#===------------------------------------------------------------------===
# Call rewrites
#===------------------------------------------------------------------===

def rewrite_calls(func, env):
    """
    Resolve methods calls via static function calls.
    """
    context = env['numba.typing.context']

    b = OpBuilder()
    for op in func.ops:
        if op.opcode == 'call':
            # Retrieve typed function
            f, args = op.args
            argtypes = [context[a] for a in args]
            signature = context[f]
            typed_func, restype = infer_call(f, signature, argtypes)

            # Rewrite call
            newop = b.call(op.type, [typed_func, op.args[1]], op.result)
            op.replace(newop)

    env['numba.state.callgraph'] = None