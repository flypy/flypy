# -*- coding: utf-8 -*-

"""
Generator implementation.
"""

from __future__ import print_function, division, absolute_import

import flypy
from flypy import jit, sjit, typeof
from ..interfaces import Iterator

@sjit('Generator[produce, consume, state, func]')
class Generator(Iterator):
    """
    Generator object with dummy methods.

    The `produce` parameter specifies the type of values we produce.

    The `consume` parameter specifies the values of objects that can be
    sent into a generator.
    """

    layout = [('state', 'state'), ('tok', 'int32')]

    # -- Dummy Methods -- #

    @jit('a -> a')
    def __iter__(self):
        return self

    # TODO: Omit type parameters, e.g. Generator[p, c]

    @jit('Generator[p, c, s, f] -> p')
    def __next__(self):
        return self.dummy # NOTE: This is only to pass type checking !

    @jit('Generator[p, c, s, f] -> c -> p')
    def send(self, value):
        return self.dummy # NOTE: This is only to pass type checking !

    @jit('a -> Exception[] -> void')
    def throw(self, exc):
        raise NotImplementedError
