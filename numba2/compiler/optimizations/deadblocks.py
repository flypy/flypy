# -*- coding: utf-8 -*-

"""
Remove dead basic blocks.
"""

from __future__ import print_function, division, absolute_import

from pykit.analysis import cfa

def run(func, env):
    cfg = cfa.cfg(func)
    deadblocks = cfa.find_dead_blocks(func, cfg)
    cfa.delete_blocks(func, cfg, deadblocks)