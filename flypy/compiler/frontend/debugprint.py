# -*- coding: utf-8 -*-

"""
Handle calls to debugprint(), see runtime/special.py.
"""

from __future__ import print_function, division, absolute_import

from pykit.ir import Builder, Op, Const
from pykit import types as ptypes

def run(func, env):
    from flypy.runtime.special import debugprint

    for op in func.ops:
        if op.opcode == 'call':
            f, args = op.args
            if isinstance(f, Const) and f.const == debugprint:
                env["flypy.state.have_debugprint"] = True
                op.replace(Op("debugprint", ptypes.Void, args,
                              result=op.result))

                [expr] = args
                debugprint_expr(expr, "frontend")
                #op.delete()


def debugprint_typed(func, env, current_stage="typing"):
    if not env["flypy.state.have_debugprint"]:
        return

    for op in func.ops:
        if op.opcode == 'debugprint':
            [expr] = op.args
            debugprint_expr(expr, "typing")
            context = env["flypy.typing.context"]
            print("    %s :: %s" % (expr, context[expr]))
            op.delete()


def debugprint_expr(expr, stage):
    print("debugprint from %s: %s" % (stage, expr))
