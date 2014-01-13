# -*- coding: utf-8 -*-

"""
Type resolution and method resolution.
"""

from __future__ import print_function, division, absolute_import

import flypy
from flypy.compiler.utils import callmap, jitcallmap
from flypy.compiler.special import SETATTR
from flypy.compiler.signature import get_remaining_args, flatargs
from flypy.compiler.utils import Caller
from flypy.compiler.typing.resolution import (infer_call, is_method,
                                              infer_getattr, make_method)
from flypy.compiler.signature import compute_missing
from flypy.runtime import primitives
from flypy.runtime.obj.core import EmptyTuple, StaticTuple, Constructor, extract_tuple_eltypes

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
    context = env['flypy.typing.context']

    b = OpBuilder()
    builder = Builder(func)

    for op in func.ops:
        if op.opcode == 'getfield':
            value, attr = op.args
            obj_type = context[value]
            attr_type = flypy.String[()]

            if attr not in obj_type.fields and attr not in obj_type.layout:
                assert '__getattr__' in obj_type.fields

                op.set_args([value, '__getattr__'])

                # Construct attribute string
                attr_string = OConst(attr)

                # Retrieve __getattr__ function and type
                getattr_func, func_type, restype = infer_getattr(
                    obj_type, op, env)

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
    context = env['flypy.typing.context']

    b = Builder(func)

    for op in func.ops:
        if op.opcode == 'setfield':
            obj, attr, value = op.args
            obj_type = context[obj]
            attr_type = flypy.String[()]

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
                context[call] = flypy.Void[()]
                context[attr_string] = attr_type


def rewrite_calls(func, env):
    """
    Resolve methods and function calls (which may be overloaded!) via static
    function calls.
    """
    b = OpBuilder()

    def f(context, op):
        f, args = op.args
        signature = context[f]

        # Retrieve typed function from the given arg types
        argtypes = [context[a] for a in args]
        typed_func, _, _ = infer_call(f, op, signature, argtypes, env)

        if is_method(signature):
            # Insert self in args list
            getfield = op.args[0]
            self = getfield.args[0]
            args = [self] + args

        # Rewrite call
        newop = b.call(op.type, typed_func, args, result=op.result)
        op.replace(newop)

    callmap(f, func, env)


def rewrite_optional_args(func, env):
    """
    Rewrite function application with missing arguments, which are supplied
    from defaults.

        def f(x, y=4):
            ...

        call(f, [x]) -> call(f, [x, const(4)])
    """
    from flypy import typeof

    def f(context, py_func, f_env, op):
        f, args = op.args

        # Add any potentially remaining values
        remaining = get_remaining_args(py_func, (None,) * len(args))
        consts = [allocate_const(func, env, op, value, typeof(value))
                      for value in remaining]
        op.set_args([f, args + consts])

    jitcallmap(f, func, env)


@flypy.jit
def slicetuple(t, n):
    return t[n:]

def rewrite_unpacking(func, env):
    """
    Rewrite argument unpacking:

        f(x, *args)
              ^^^^^

        temp = tuple(args)
        f(x, temp[0], temp[1])
    """
    from flypy.pipeline import phase

    b = Builder(func)
    caller = Caller(b, env['flypy.typing.context'], env)
    call_flags = env['flypy.state.call_flags']

    def f(context, py_func, f_env, op):
        f, args = op.args
        flags = call_flags.get(op, {})

        if not flags.get('varargs'):
            return

        # Unpack positional and varargs argument
        # For f(x, *args): positional = [x], varargs = args
        positional, varargs = args[:-1], args[-1]
        varargs_type = context[varargs]

        # Missing number of positional arguments (int)
        missing = compute_missing(py_func, positional)

        # Types in the tuple, may be heterogeneous
        eltypes = extract_tuple_eltypes(varargs_type, missing)

        # Now supply the missing positional arguments
        b.position_before(op)
        for i, argty in zip(range(missing), eltypes):
            idx = OConst(i)
            context[idx] = flypy.int32

            #hd = b.getfield(types.Opaque, varargs, 'hd')
            #tl = b.getfield(types.Opaque, varargs, 'tl')

            positional_arg = caller.call(phase.typing,
                                         primitives.getitem,
                                         [varargs, idx])
            positional.append(positional_arg)

        # TODO: For GenericTuple unpacking, assure that
        #       len(remaining_tuple) == missing

        if len(eltypes) > missing:
            # In case we have more element types than positional parameters,
            # we must have a function that takes varargs, e.g.
            #
            #   def f(x, y, *args):
            #       ...
            #
            # That we call as follows (for example):
            #
            #       f(x, *args)
            idx = OConst(missing)
            context[idx] = flypy.int32

            argstup = caller.call(phase.typing,
                                  slicetuple,
                                  [varargs, idx])
            positional.append(argstup)

        # Update with new positional arguments
        op.set_args([f, positional])

    jitcallmap(f, func, env)


def rewrite_varargs(func, env):
    """
    Rewrite function application with arguments that go in the varargs:

        def f(x, *args):
            ...

        call(f, [x, y, z]) -> call(f, [x, (y, z)])
    """
    b = Builder(func)
    caller = Caller(b, env['flypy.typing.context'], env)
    caller = Caller(b, env['flypy.typing.context'], env)
    call_flags = env['flypy.state.call_flags']

    def f(context, py_func, f_env, op):
        f, args = op.args
        flags = call_flags.get(op, {})

        if flags.get('varargs'):
            return

        # Retrieve any remaining arguments meant for *args
        flattened = flatargs(py_func, args, {})
        if flattened.have_varargs:
            b.position_before(op)
            #print(py_func, flattened.varargs)

            # -- Build the tuple -- #
            result = caller.apply_constructor(EmptyTuple)
            for item in flattened.varargs:
                result = caller.apply_constructor(StaticTuple,
                                                  args=[item, result])

            # -- Patch callsite -- #
            args = list(flattened.positional) + [result]
            op.set_args([f, args])

    jitcallmap(f, func, env)

def allocate_const(func, env, op, value, type):
    const = Const(value, types.Opaque)
    context = env['flypy.typing.context']
    context[const] = type
    return const
