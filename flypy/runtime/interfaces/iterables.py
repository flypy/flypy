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
    def __iter__(self):
        return self

    @abstract('Iterator[X] -> Iterator[X]')
    def __next__(self):
        raise NotImplementedError

    @jit
    def next(self):
        return next(self)


@abstract('Sequence[X]')
class Sequence(Iterable):
    """Interface for iterables"""

    @abstract('Sequence[X] -> int64')
    def __contains__(self, value):
        for item in self:
            if item == value:
                return True
        return False

    @abstract('Sequence[X] -> int64 -> X')
    def __getitem__(self, item):
        raise NotImplementedError

    @abstract('Sequence[X] -> int64')
    def __len__(self):
        raise NotImplementedError

    @abstract('Sequence[X] -> Iterator[X]')
    def __reversed__(self):
        for i in range(len(self) - 1, -1 , -1):
            yield self[i] # TODO: Generators