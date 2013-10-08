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
from pykit.ir import OpBuilder, Builder, Const, Function

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


def rewrite_constructors(func, env):
    """
    Rewrite constructor application to object allocation followed by
    cls.__init__:

        call(C, x, y) -> call(C.__init__, x, y)
    """
    from numba2 import phase

    context = env['numba.typing.context']
    b = OpBuilder()

    for op in func.ops:
        if op.opcode == 'call':
            cls, args = op.args
            if isinstance(cls, Const) and is_numba_type(cls.const):
                cls = cls.const
                f = cls.__init__
                type = context[op]
                argtypes = [type] + [context[arg] for arg in op.args[1]]

                # TODO: implement this on Type.__call__ when we support *args
                e = fresh_env(f, argtypes)
                __init__, _ = phase.typing(f, e)

                alloc = b.alloca(types.Pointer(types.Opaque))
                call = b.call(types.Void, [__init__, [alloc] + args])

                op.replace_uses(alloc)
                op.replace([alloc, call])

                context[alloc] = type

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