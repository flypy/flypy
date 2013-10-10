# -*- coding: utf-8 -*-

"""
Type resolution and method resolution.
"""

from __future__ import print_function, division, absolute_import

from ..typing.resolution import infer_call, is_method, get_remaining_args

from pykit import types
from pykit.ir import OpBuilder, Builder, Const

#===------------------------------------------------------------------===
# Call rewrites
#===------------------------------------------------------------------===

def rewrite_calls(func, env):
    """
    Resolve methods and function calls (which may be overloaded!) via static
    function calls.
    """
    context = env['numba.typing.context']

    b = OpBuilder()
    for op in func.ops:
        if op.opcode == 'call':
            # Retrieve typed function
            f, args = op.args
            signature = context[f]

            # Retrieve typed function from the given arg types
            argtypes = [context[a] for a in args]
            typed_func, restype = infer_call(f, signature, argtypes)

            if is_method(signature):
                # Insert self in args list
                getfield = op.args[0]
                self = getfield.args[0]
                args = [self] + args

            # Rewrite call
            newop = b.call(op.type, [typed_func, args], op.result)
            op.replace(newop)

    env['numba.state.callgraph'] = None

def rewrite_optional_args(func, env):
    """
    Rewrite function application with missing arguments, which are supplied
    from defaults.

        def f(x, y=4):
            ...

        call(f, [x]) -> call(f, [x, const(4)])
    """
    from numba2 import typeof

    envs = env['numba.state.envs']

    for op in func.ops:
        if op.opcode == 'call':

            # Retrieve function and environment
            f, args = op.args
            f_env = envs[f]

            # Retrieve Python version and opaqueness
            py_func = f_env['numba.state.py_func']
            opaque = f_env['numba.state.opaque']

            if py_func and not opaque:
                # Add any potentially remaining values
                remaining = get_remaining_args(py_func, (None,) * len(args))
                consts = [allocate_const(func, env, op, value, typeof(value))
                              for value in remaining]
                op.set_args([f, args + consts])


def allocate_const(func, env, op, value, type):
    # TODO: Move this elsewhere
    # TODO: Handle complex values (non-pykit constants)
    const = Const(value, types.Opaque)
    context = env['numba.typing.context']
    context[const] = type
    return const