# -*- coding: utf-8 -*-

"""
Type resolution and method resolution.
"""

from __future__ import print_function, division, absolute_import
import inspect

from flypy.errors import UnificationError
from flypy.pipeline import fresh_env
from flypy import promote, unify, typejoin
from flypy.functionwrapper import FunctionWrapper
from flypy.types import Type, Constructor, ForeignFunction, Function, void
from flypy.typing import TypeConstructor
from flypy.compiler.overloading import fill_missing_argtypes
from flypy.rules import infer_type_from_layout

from pykit import ir, types
from pykit.ir import ops

#===------------------------------------------------------------------===
# Function call typing
#===------------------------------------------------------------------===

# Method(func, self_type)
Method = TypeConstructor("Method", 2, [{'coercible': True}] * 2)

def is_method(t):
    return isinstance(t, Method)

def make_method(type, attr):
    value = type.fields[attr]
    func, self = value, type
    return Method(func, self)


def infer_call(func, func_type, argtypes, env, flags={}):
    """
    Infer a single call. We have three cases:

        1) Static receiver function
        2) Higher-order function
            This is already typed
        3) Method. We need to insert 'self' in the cartesian product

    NOTE: This must only be called during the type inference phase !

    Parameters
    ==========
    func: ir.Value
        IR representation of the callee
    func_type: Function, ForeignFunction or Method
        signature of callee
    flags: dict
        dict containing info regarding the presence of varargs and keyword args
    """
    is_const = isinstance(func, ir.Const)
    is_flypy_func = is_const and isinstance(func.const, FunctionWrapper)
    is_class = isinstance(func_type, (type(Type.type), type(Constructor.type)))

    if is_method(func_type) or is_flypy_func:
        return infer_function_call(func, func_type, argtypes, env, flags)
    elif is_class:
        return infer_class_call(func, func_type, argtypes, flags)
    elif not isinstance(func, ir.Function):
        return infer_foreign_call(func, func_type, argtypes, flags)
    else:
        raise NotImplementedError(func, func_type)


def infer_function_call(func, func_type, argtypes, env, flags):
    """
    Method call or flypy function call.
    """
    from flypy.pipeline import phase

    #print(('inferring %s' % func).center(80, '!'))

    if is_method(func_type):
        func = func_type.parameters[0]
        argtypes = [func_type.parameters[1]] + list(argtypes)
    else:
        func = func.const

    # TODO: Support recursion !

    env = env['flypy.fresh_env'](func, argtypes, **flags)
    func, env = phase.typing(func, env)
    # env["flypy.typing.restype"]
    if func_type is None:
        func_type = env["flypy.typing.signature"]
    return func, func_type, env["flypy.typing.restype"]


def infer_class_call(func, func_type, argtypes, flags):
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


def infer_foreign_call(func, func_type, argtypes, flags):
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


#===------------------------------------------------------------------===
# Attribute resolution
#===------------------------------------------------------------------===

def infer_getattr(type, env):
    """
    Infer a call of obj.__getattr__(attr)
    """
    from flypy.runtime.obj.core import String

    func_type = make_method(type, '__getattr__')
    func = func_type.parameters[0]

    attr_type = String[()]
    return infer_call(func, func_type, [attr_type], env)


#===------------------------------------------------------------------===
# Type resolution
#===------------------------------------------------------------------===

def resolve_context(func, env):
    """Reduce typesets in context to concrete types"""
    context = env['flypy.typing.context']

    for op, typeset in context.iteritems():
        if typeset:
            typeset = context[op]
            try:
                ty = reduce(typejoin, typeset)
            except TypeError, e:
                raise TypeError(
                    "Cannot type-join types for op %s: %s" % (op, e))

            context[op] = ty

        elif isinstance(op, ir.Op) and not ops.is_void(op.opcode):
            print(func)
            raise TypeError("op %s has no type in function %s" % (op, func.name))


def resolve_restype(func, env):
    """Figure out the return type and update the context and environment"""
    context = env['flypy.typing.context']
    restype = env['flypy.typing.restype']
    signature = env['flypy.typing.signature']

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

    if isinstance(restype, set):
        raise TypeError(
            "Undetermined return type for function %s" % (func.name,))

    env['flypy.typing.restype'] = restype

    if restype == void or env['flypy.state.generator']:
        _, argtypes, varargs = func.type
        func.type = types.Function(types.Void, argtypes, varargs)
