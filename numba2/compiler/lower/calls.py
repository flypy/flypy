# -*- coding: utf-8 -*-

"""
Type resolution and method resolution.
"""

from __future__ import print_function, division, absolute_import

from ..typing.resolution import infer_call, is_method, get_remaining_args

from pykit import types
from pykit.ir import OpBuilder, Builder, Const, Function, Op

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

#------------------------------------------------------------------------
# Coercions -> Conversions
#------------------------------------------------------------------------

def explicit_coercions(func, env):
    """
    Turn implicit coercions into explicit conversion operations.
    """
    context = env["numba.typing.context"]
    envs = env["numba.state.envs"]

    # Conversion cache, { (Op, dsttype) : convert Op }. A single value may
    # otherwise be converted multiple times in different contexts
    conversions = {}
    b = Builder(func)

    for op in func.ops:
        if op.opcode != 'call':
            continue

        # -------------------------------------------------

        f, args = op.args
        # TODO: Signatures should always be in the context !
        if f in context:
            argtypes = context[f].parameters[:-1]
        else:
            argtypes = envs[f]["numba.typing.argtypes"]
        replacements = {} # { arg : replacement_conversion }

        # -------------------------------------------------

        for arg, param_type in zip(args, argtypes):
            isconst = isinstance(arg, Const)
            arg_type = context[arg]

            if arg_type != param_type:
                # Argument type does not match parameter type, convert
                conversion = conversions.get((arg, param_type))
                if not conversion:
                    # Create conversion and update typing context with new Op
                    conversion = Op('convert', types.Opaque, [arg])
                    context[conversion] = param_type

                    if isconst:
                        b.position_before(op)
                    else:
                        b.position_after(arg)
                        conversions[arg, param_type] = conversion

                    b.emit(conversion)

                replacements[arg] = conversion

        # -------------------------------------------------

        op.replace_args(replacements)
