# -*- coding: utf-8 -*-

"""
Numba function wrapper.
"""

from __future__ import print_function, division, absolute_import

# TODO: Reuse numba.numbawrapper.pyx for autojit Python entry points

class Function(object):
    """
    Result of @jit.
    """

    def __init__(self, py_func, signature):
        self.py_func = py_func
        self.signature = signature