# -*- coding: utf-8 -*-

"""
Type resolution and method resolution.
"""

from __future__ import print_function, division, absolute_import

import numba2
from numba2.compiler.special import SETATTR
from ..typing.resolution import (infer_call, is_method, get_remaining_args,
                                 infer_getattr, make_method)

from pykit import types
from pykit.ir import OpBuilder, Builder, Const, OConst, Function, Op

#===------------------------------------------------------------------===
# Call rewrites
#===------------------------------------------------------------------===

# TODO: Implement rewrite engine

def rewrite_getattr(func, env):
    """
    Resolve missing attributes through __getattr__
    """
    context = env['numba.typing.context']

    b = OpBuilder()
    builder = Builder(func)

    for op in func.ops:
        if op.opcode == 'getfield':
            value, attr = op.args
            obj_type = context[value]
            attr_type = numba2.String[()]

            if attr not in obj_type.fields and attr not in obj_type.layout:
                assert '__getattr__' in obj_type.fields

                op.set_args([value, '__getattr__'])

                # Construct attribute string
                attr_string = OConst(attr)

                # Retrieve __getattr__ function and type
                getattr_func, func_type, restype = infer_getattr(obj_type, env)

                # call(getfield(obj, '__getattr__'), ['attr'])
                call = b.call(op.type, op, [attr_string])
                op.replace_uses(call)
                builder.position_after(op)
                builder.emit(call)

                # Update context
                context[op] = func_type
                context[attr_string] = attr_type
                context[call] = restype


def rewrite_setattr(func, env):
    """
    Resolve missing attributes through __setattr__
    """
    context = env['numba.typing.context']

    b = Builder(func)

    for op in func.ops:
        if op.opcode == 'setfield':
            obj, attr, value = op.args
            obj_type = context[obj]
            attr_type = numba2.String[()]

            if attr not in obj_type.fields and attr not in obj_type.layout:
                assert SETATTR in obj_type.fields, attr

                b.position_after(op)

                # Construct attribute string
                attr_string = OConst(attr)

                # call(getfield(obj, '__setattr__'), ['attr', value])
                method_type = make_method(obj_type, SETATTR)
                method = b.getfield(types.Opaque, obj, SETATTR)
                call = b.call(types.Opaque, method, [attr_string, value])
                op.delete()

                # Update context
                del context[op]
                context[method] = method_type
                context[call] = numba2.Void[()]
                context[attr_string] = attr_type


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
            typed_func, _, _ = infer_call(f, signature, argtypes, env)

            if is_method(signature):
                # Insert self in args list
                getfield = op.args[0]
                self = getfield.args[0]
                args = [self] + args

            # Rewrite call
            newop = b.call(op.type, typed_func, args, result=op.result)
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
            if not isinstance(f, Function):
                continue

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
    const = Const(value, types.Opaque)
    context = env['numba.typing.context']
    context[const] = type
    return const
