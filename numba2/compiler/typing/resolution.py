# -*- coding: utf-8 -*-

"""
Type resolution and method resolution.
"""

from __future__ import print_function, division, absolute_import
import inspect
from blaze.error import UnificationError

from numba2.pipeline import fresh_env
from numba2 import promote, unify, typejoin
from numba2.functionwrapper import FunctionWrapper
from numba2.types import Type, Constructor, ForeignFunction, Function
from numba2.compiler.overloading import flatargs
from numba2.rules import infer_type_from_layout

from pykit import ir

#===------------------------------------------------------------------===
# Function call typing
#===------------------------------------------------------------------===

# TODO: Move this function to a third module

def is_method(t):
    return type(t).__name__ == 'Method' # hargh

def infer_call(func, func_type, argtypes):
    """
    Infer a single call. We have three cases:

        1) Static receiver function
        2) Higher-order function
            This is already typed
        3) Method. We need to insert 'self' in the cartesian product
    """
    is_const = isinstance(func, ir.Const)
    is_numba_func = is_const and isinstance(func.const, FunctionWrapper)
    is_class = isinstance(func_type, (type(Type.type), type(Constructor.type)))

    if is_method(func_type) or is_numba_func:
        return infer_function_call(func, func_type, argtypes)
    elif is_class:
        return infer_class_call(func, func_type, argtypes)
    elif not isinstance(func, ir.Function):
        return infer_foreign_call(func, func_type, argtypes)
    else:
        raise NotImplementedError(func, func_type)

def infer_function_call(func, func_type, argtypes):
    """
    Method call or numba function call.
    """
    from numba2.pipeline import phase

    #print(('inferring %s' % func).center(80, '!'))

    if is_method(func_type):
        func = func_type.parameters[0]
        argtypes = [func_type.parameters[1]] + list(argtypes)
    else:
        func = func.const

    # TODO: Support recursion !

    if len(func.overloads) == 1 and not func.opaque:
        argtypes = fill_missing_argtypes(func.py_func, tuple(argtypes))

    env = fresh_env(func, argtypes)
    func, env = phase.typing(func, env)
    # env["numba.typing.restype"]
    if func_type is None:
        func_type = env["numba.typing.signature"]
    return func, func_type, env["numba.typing.restype"]

def infer_class_call(func, func_type, argtypes):
    """
    Constructor application.
    """
    classtype = func_type.parameters[0] # extract T from Type[T]
    freevars = classtype.parameters
    argtypes = [classtype] + list(argtypes)
    if freevars:
        classtype = infer_constructor_application(classtype, argtypes)

    # TODO: Return a Constructor?
    return func, Function[tuple(argtypes) + (classtype,)], classtype

def infer_foreign_call(func, func_type, argtypes):
    """
    Higher-order or foreign function call.
    """

    if isinstance(func_type, type(ForeignFunction.type)):
        restype = func_type.parameters[-1]
    else:
        restype = func_type.restype
    assert restype

    expected_argtypes = func_type.parameters[:-1]

    if len(argtypes) != len(expected_argtypes):
        if not (func_type.varargs and len(argtypes) >= len(expected_argtypes)):
            raise TypeError("Function %s requires %d argument(s), got %d" % (
                                func, len(expected_argtypes), len(argtypes)))

    try:
        # Make sure we have compatible types
        unify(zip(argtypes, expected_argtypes))
    except UnificationError:
        raise TypeError(
            "Mismatching signature for function %s with argument types %s" % (
                func, ", ".join(map(str, argtypes))))

    return func, Function[expected_argtypes + (restype,)], restype


def infer_constructor_application(classtype, argtypes):
    """
    Resolve the free type variables of a constructor given the argument types
    we instantiate the class with.

    This means we need to match up argument types with variables from the
    class layout. In the most general case this means we need to fixpoint
    infer all methods called from the constructor.

    For now, we do the dumb thing: Match up argument names with variable names
    from the layout.
    """
    # Figure out the list of argtypes
    cls = classtype.impl
    init = cls.__init__.py_func
    argtypes = fill_missing_argtypes(init, tuple(argtypes))

    # Determine __init__ argnames
    argspec = inspect.getargspec(init)
    assert not argspec.varargs
    assert not argspec.keywords
    argnames = argspec.args
    assert len(argtypes) == len(argnames)

    return infer_type_from_layout(classtype, zip(argnames, argtypes))


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
            try:
                ty = reduce(typejoin, typeset)
            except TypeError, e:
                raise TypeError(
                    "Cannot type-join types for op %s: %s" % (op, e))

            context[op] = ty

def resolve_restype(func, env):
    """Figure out the return type and update the context and environment"""
    context = env['numba.typing.context']
    restype = env['numba.typing.restype']
    signature = env['numba.typing.signature']

    typeset = context['return']
    inferred_restype = signature.restype

    if restype is None:
        restype = inferred_restype
    elif inferred_restype != restype:
        try:
            [restype] = unify([(inferred_restype, restype)])
        except UnificationError, e:
            raise TypeError(
                "Annotated result type %s does not match inferred "
                "type %s for function %r: %s" % (
                    restype, inferred_restype, func.name, e))

    env['numba.typing.restype'] = restype
