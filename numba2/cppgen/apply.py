# -*- coding: utf-8 -*-

"""
Apply a numba function over a tuple of arguments.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit

def applier(f, argtypes):
    # TODO: Make `apply` ingest a tuple/array of arguments
    @jit
    def apply(args):
        return f(*args)
    return apply