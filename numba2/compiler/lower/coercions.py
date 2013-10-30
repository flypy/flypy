# -*- coding: utf-8 -*-

"""
Type coercions.
"""

from __future__ import print_function, division, absolute_import

from ..typing.resolution import infer_call, is_method, get_remaining_args

from pykit import types
from pykit.ir import OpBuilder, Builder, Const, Function, Op

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
    coercer = Coercion(b, context, envs, conversions)

    for op in func.ops:
        if op.opcode == 'call':
            coercer.coerce_to_parameters(op)
        elif op.opcode == 'setfield':
            coercer.coerce_to_field_setting(op)

class Coercion(object):

    def __init__(self, builder, context, envs, conversions):
        self.builder = builder
        self.context = context
        self.envs = envs
        self.conversions = conversions

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
        replacements = {} # { arg : replacement_conversion }

        # -------------------------------------------------
        # Promote arguments to match parameter types

        for arg, param_type in zip(args, argtypes):
            arg_type = self.context[arg]

            if arg_type != param_type:
                # Argument type does not match parameter type, convert
                conversion = self.convert(arg, param_type, op)
                replacements[arg] = conversion

        op.replace_args(replacements)


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
            isconst = isinstance(arg, Const)

            conversion = Op('convert', types.Opaque, [arg])
            self.context[conversion] = ty

            if isconst:
                self.builder.position_before(op)
            else:
                self.builder.position_after(arg)
                self.conversions[arg, ty] = conversion

            self.builder.emit(conversion)

        return conversion