# -*- coding: utf-8 -*-

"""
Run sparse conditional constant propagation from pykit using the flypy
constant folder.
"""

from __future__ import print_function, division, absolute_import
from .constfolding import ConstantFolder

from pykit.optimizations import sccp

def run(func, env):
    folder = ConstantFolder(env['flypy.typing.context'], env)
    sccp.run(func, env, folder)