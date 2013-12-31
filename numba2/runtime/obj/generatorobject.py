# -*- coding: utf-8 -*-

"""
Generator implementation.
"""

from __future__ import print_function, division, absolute_import

import numba2
from numba2 import jit, sjit, typeof
from ..interfaces import Iterator

@sjit('Generator[produce, consume]')
class Generator(Iterator):
    """
    Generator object with dummy methods. The `produce` parameter specifies
    the type of values we produce, and the `consume` parameter specifies
    the values of objects that can be sent into a generator.
    """

    layout = [('dummy', 'produce')]

    # -- Dummy Methods -- #

    @jit('a -> a')
    def __iter__(self):
        return self

    @jit('Generator[p, c] -> p')
    def __next__(self):
        return self.dummy # NOTE: This is only to pass type checking !

    @jit('Generator[p, c] -> c -> p')
    def send(self, value):
        return self.dummy # NOTE: This is only to pass type checking !

    @jit('a -> Exception[] -> void')
    def throw(self, exc):
        raise NotImplementedError
