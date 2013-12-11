# -*- coding: utf-8 -*-

"""
Type coercions.
"""

from __future__ import print_function, division, absolute_import

import numba2

from pykit import types
from pykit.ir import OpBuilder, Builder, Const, Function, Op, Undef, ops

#------------------------------------------------------------------------
# Coercions -> Conversions
#------------------------------------------------------------------------

# TODO: In a later pass, we need to call builtins like int() and bool() or the
# TODO: 'cast function to perform the coercions

def explicit_coercions(func, env):
    """
    Turn implicit coercions into explicit conversion operations.
    """
    if env['numba.state.opaque']:
        return # TODO: @no_opaque decorator...

    context = env["numba.typing.context"]
    envs = env["numba.state.envs"]

    # Conversion cache, { (Op, dsttype) : convert Op }. A single value may
    # otherwise be converted multiple times in different contexts
    conversions = {}
    b = Builder(func)
    coercer = Coercion(func, b, context, envs, conversions, env)

    for op in func.ops:
        if op.opcode == 'call':
            coercer.coerce_to_parameters(op)
        elif op.opcode == 'store':
            coercer.coerce_to_var(op)
        elif op.opcode == 'setfield':
            coercer.coerce_to_field_setting(op)
        elif op.opcode == 'phi':
            coercer.coerce_to_phi(op)
        elif op.opcode == 'ret':
            coercer.coerce_to_restype(op)
        elif op.opcode == 'cbranch':
            coercer.coerce_to_conditional(op)


class Coercion(object):

    def __init__(self, f, builder, context, envs, conversions, env):
        self.f = f
        self.builder = builder
        self.context = context
        self.envs = envs
        self.conversions = conversions
        self.env = env

    def coerce_to_parameters(self, op):
        """
        Promote arguments to match parameter types.
        """
        # -------------------------------------------------

        f, args = op.args

        # TODO: Signatures should always be in the context !
        if f in self.context:
            argtypes = self.context[f].parameters[:-1]
        else:
            argtypes = self.envs[f]["numba.typing.argtypes"]

        # -------------------------------------------------
        # Promote arguments to match parameter types

        newargs = self.promote_args(args, argtypes, op)

        # Append any extrananeous arguments for varargs...
        newargs.extend(args[len(newargs):])

        op.set_args([f, newargs])

    def coerce_to_var(self, op):
        val, var = op.args
        if self.context[val] != self.context[var]:
            if isinstance(val, Undef):
                self.context[val] = self.context[var]
            else:
                newval = self.convert(val, self.context[var], op)
                op.set_args([newval, var])

    def coerce_to_field_setting(self, op):
        """
        Promote values for field setting.
        """
        obj, attr, value = op.args

        obj_type = self.context[obj]
        field_type = obj_type.resolved_layout[attr]
        value_type = self.context[value]

        if field_type != value_type:
            newval = self.convert(value, field_type, op)
            op.set_args([obj, attr, newval])

    def coerce_to_restype(self, op):
        """
        Coerce return value to return type.
        """
        restype = self.env['numba.typing.restype']
        [retval] = op.args
        if retval is not None and self.context[retval] != restype:
            retval = self.convert(retval, restype, op)
            op.set_args([retval])

    def coerce_to_conditional(self, op):
        restype = numba2.bool_
        cond, trueblock, falseblock = op.args
        if self.context[cond] != restype:
            retval = self.convert(cond, restype, op)
            op.set_args([retval, trueblock, falseblock])

    def coerce_to_phi(self, op):
        """
        Coerce incoming phi values to the type of the `phi` node.
        """
        blocks, vals = map(list, op.args) # copy blocks, vals
        ty = self.context[op]

        # Promote constants in previous blocks
        for i, (pred, val) in enumerate(zip(blocks, vals)):
            if isinstance(val, Const) and self.context[val] != ty:
                self.builder.position_before(pred.tail)
                newval = self.builder.convert(types.Opaque, val)
                self.context[newval] = ty
                vals[i] = newval
            elif isinstance(val, Undef) and self.context[val] != ty:
                newval = Undef(types.Opaque)
                self.context[newval] = ty
                vals[i] = newval

        newargs = self.promote_args(vals, [ty] * len(vals), op)
        op.set_args([blocks, newargs])

    def promote_args(self, args, argtypes, op):
        """
        Promote values from `args` to types listed by `argtypes`, and place
        them before `op` in the function.
        """
        newargs = []
        for arg, param_type in zip(args, argtypes):
            if self.context[arg] != param_type:
                # Argument type does not match parameter type, convert
                arg = self.convert(arg, param_type, op)
            newargs.append(arg)

        return newargs

    def convert(self, arg, ty, op):
        """
        Create conversion and update typing context with new conversion Op.

        Parameters
        ==========
        b: Builder
        arg: Op
            op to convert
        ty: Type
            Type to convert arg to
        op: Op
            insert conversion before this op
        """
        conversion = self.conversions.get((arg, ty))
        if not conversion:
            isconst = isinstance(arg, (Undef, Const))

            conversion = Op('coerce', types.Opaque, [arg])
            self.context[conversion] = ty

            if isconst:
                self.builder.position_before(op)
            else:
                self.builder.position_after(arg)
                self.conversions[arg, ty] = conversion

            self.builder.emit(conversion)

        return conversion