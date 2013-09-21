# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function, division, absolute_import

from numba2.typing import promote, coerce, unify_simple

from pykit import types
from pykit.ir import Builder, OpBuilder, Op

#===------------------------------------------------------------------===
# Entrypoint
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

def rewrite_methods(func, env):
    """
    Resolve methods calls via static function calls.
    """
    from .inference import Method

    cache = env['numba.typing.cache']
    context = env['numba.typing.context']

    b = OpBuilder()
    for op in func.ops:
        if op.opcode == 'call':
            f = op.args[0]
            signature = context[f]
            if type(signature) == Method:
                func, self = signature.parameters
                argtypes = [context[arg] for arg in op.args[1]]
                ctx, signature = cache.lookup(func, argtypes)
                newop = b.call(op.type, [ctx.func, op.args[1]], op.result)
                op.replace(newop)