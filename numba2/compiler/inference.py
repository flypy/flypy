# -*- coding: utf-8 -*-

"""
Compute the most general type unifier and a set of constraints for functions.

Note that promotion is handled through overloading, e.g.:

    @overload('α -> β -> γ')
    def __add__(self, other):
        return other.__radd__(self)

    @overload('α -> α -> β')
    def __add__(self, other):
        # Guard against infinite recursion
        raise NotImplementedError("__add__")

    @overload('α -> β -> γ')
    def __radd__(self, other):
        type = promote(typeof(self), typeof(other))
        return convert(other, type) + convert(self, type)

These are implemented in a trait which can be implemented by a user-defined
type (like Int). If there is no more specific overload, __radd__ will do the
promotion, triggering either an error or a call to a real implementation.
"""

from __future__ import print_function, division, absolute_import
import collections
from functools import partial

from .typing import Typevar, Constraint
from .types import Type, Function
from .. import InferError

import pykit.types
from pykit.ir import visit

def compute_principal_type(func, context=None):
    """Compute the principal type scheme of a function"""
    if context is None:
        context = {} # Γ: term -> type

    assign_type_variables(func, context)
    constraints = list(generate_constraints(func, context))
    context = unify(constraints)
    mgu = substitute(context, func.type)
    return (mgu, context, constraints)

def assign_type_variables(func, context):
    """Assign fresh type variables to the function body."""
    for arg in func.args:
        arg.type = context.get(arg) or Typevar()
    for op in func.ops:
        if op.type != pykit.types.Void:
            op.type = context.get(op) or Typevar()

def generate_constraints(func, context=None):
    """
    Generate constraints for untyped IR in SSA form.
    """
    context = context or {}
    context[func] = monotype(func.type)
    gen = ConstraintGenerator(func, context)
    visit(gen)
    return context, gen.constraints


class ConstraintGenerator(object):

    def __init__(self, func, context):
        self.func = func
        self.constraints = []
        self.context = context

    def constraint(self, op, c_op, args):
        self.constraints.append(Constraint(op.lineno, c_op, args))

    def op_phi(self, op):
        self.constraint(op, 'unify', op.args[1])

    def op_getfield(self, op):
        arg, attr = op.args
        self.constraint(op, 'hasfield', [arg.type, attr])

    def _call(self, op, func_type, args):
        expected = Function(op.type, [a.type for a in args])
        self.constraint(op, '=', [func_type, expected])

    def op_callmethod(self, op):
        method, args = op.args
        self._call(op, method.type, args)

    def op_call(self, op):
        func, args = op.args
        if func in self.context:
            func_type = self.context[func]
        else:
            func_type = monotype(func.type)

        self._call(op, func_type, args)

    def op_ret(self, op):
        functype = self.context[self.func].type
        self.constraint(op, '=', [op.args[0], functype.parameters[0]])

# ______________________________________________________________________
# Unification

# TODO: Union types

def unify_single(context, left, right):
    """Unify a single type equation"""
    get = lambda x: context.get(x, x)

    if isinstance(left, Typevar):
        context[left] = get(right)
    elif isinstance(right, Typevar):
        context[right] = get(left)
    else:
        if left.name != right.name:
            raise InferError(
                "Cannot unify types with constructors %s and %s" % (left,
                                                                    right))

        args1, args2 = left.parameters, right.parameters
        if len(args1) != len(args2):
            raise InferError("%s got %d and %d arguments" % (
                            left.name, len(args1), len(args2)))

        for a, b in zip(args1, args2):
            unify_single(context, get(a), get(b))

    return get(left)

def unify(constraints):
    """Unify a set of constraints"""
    constraints = collections.deque(constraints)
    context = {}
    get = lambda x: context.get(x, x)
    while constraints:
        c = constraints.popleft()
        if c.op == '=':
            unify_single(context, *c.args)
        elif c.op == 'unify':
            # TODO: Call user-defined unify()
            result, args = c.args[0], c.args[1:]
            context[result] = reduce(partial(unify_single, context), args)

        # TODO: the other constraints

    return context

def substitute(context, ty):
    if isinstance(ty, Typevar):
        return context[ty]
    else:
        ty.parameters = tuple(substitute(p) for p in ty.parameters)
        return ty