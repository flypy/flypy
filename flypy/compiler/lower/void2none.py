# -*- coding: utf-8 -*-

"""
Replace uses of Void with None. This works simultaneously for flypy and foreign
functions.
"""

from __future__ import print_function, division, absolute_import

import flypy
from pykit.ir import OConst

def void2none(func, env):
    if env['flypy.state.opaque']:
        return

    context = env['flypy.typing.context']
    none = OConst(None)
    for op in func.ops:
        if context[op] == flypy.void:
            if func.uses[op]:
                op.replace_uses(none)
                context[none] = flypy.typeof(None)