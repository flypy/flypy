# -*- coding: utf-8 -*-

"""
Iterable and related interfaces.
"""

from __future__ import print_function, division, absolute_import
from functools import wraps
from ... import abstract, jit

__all__ = ['Number', 'Real', 'Complex', 'Rational', 'Irrational',
           'Integer', 'Floating']

@abstract('Iterable[X]')
class Iterable(object):
    """Interface for iterables"""

    @abstract('Iterable[X] -> Iterator[X]')
    def __iter__(self):
        raise NotImplementedError

@abstract('Iterator[X]')
class Iterator(Iterable):
    """Inferface for iterators"""

    @abstract('Iterator[X] -> Iterator[X]')
    def __next__(self):
        raise NotImplementedError
