# -*- coding: utf-8 -*-

"""
Type resolution and method resolution.
"""

from __future__ import print_function, division, absolute_import

from numba2 import errors
from numba2.typing import promote, unify_simple

from pykit.ir import OpBuilder


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

def rewrite_methods(func, env):
    """
    Resolve methods calls via static function calls.
    """
    from .inference import Method

    cache = env['numba.inference.cache']
    context = env['numba.typing.context']

    b = OpBuilder()
    for op in func.ops:
        if op.opcode == 'call':
            f = op.args[0]
            signature = context[f]
            if type(signature) == Method:
                func, self = signature.parameters
                argtypes = [context[arg] for arg in op.args[1]]
                typed_func = cache.lookup(func, tuple(argtypes))
                if typed_func is None:
                    raise errors.CompileError(
                        "Failing to function from cache, this is a bug!")
                ctx, signature = typed_func
                newop = b.call(op.type, [ctx.func, op.args[1]], op.result)
                op.replace(newop)

    env['numba.state.callgraph'] = None


#===------------------------------------------------------------------===
# Calls
#===------------------------------------------------------------------===

def rewrite_calls(func, env):
    """
    Rewrite 'pycall' instructions to other Numba functions to 'call'.

    This recurses into the front-end translator, so this supports recursion
    since the function is already in the cache.
    """
    b = OpBuilder()
    for op in func.ops:
        if op.opcode == 'pycall':
            func, args = op.args[0].const, op.args[1:]
            if isinstance(func, FunctionWrapper):
                translated, env = translate(func.py_func, env)
                newop = b.call(types.Opaque, [translated, args], op.result)
                op.replace(newop)
