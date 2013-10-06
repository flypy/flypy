# -*- coding: utf-8 -*-

"""
tuple implementation.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit, typeof, overload
from ..interfaces import Number, implements

T = TypeVar()

@abstract
class Tuple(object):
    pass


@jit('GenericTuple[T]')
class Tuple(object):
    layout = [('items', 'List[T]')]

    @jit('a -> T')
    def __getitem__(self, item):
        return self.items[item]

    @jit('a -> Iterator[T]')
    def __iter__(self):
        return iter(self.items)

    @jit('a -> int64')
    def __len__(self):
        return len(self.items)

    @jit('a -> Tuple[T]')
    def __add__(self, other):
        return Tuple(self.items + other.items)


@jit('StaticTuple[T1, T2 : <StaticTuple, None>]', Number)
class StaticTuple(object):
    layout = [('hd', 'T1'), ('tl', 'T2 | None')]

    @jit('a -> b : integral -> c')
    def __getitem__(self, item):
        if item == 0:
            return self.hd
        else:
            return self.tl[item - 1]

    @jit('a -> Iterator[T]')
    def __iter__(self):
        yield self.hd
        for x in self.tl:
            yield x

    @jit('a -> int64')
    def __len__(self):
        if self.hd is None:
            return 0
        elif self.tl is None:
            return 1
        else:
            return len(self.tl) + 1

    @overload('a -> StaticTuple[t1, t2] -> c')
    def __add__(self, other):
        for i in unroll(range(len(self) - 1, -1, -1)):
            item = self[i]
            other = StaticTuple(item, other)
        return other

    @overload('a -> Tuple[T] -> Tuple[T]')
    def __add__(self, other):
        result = List[T]()
        for x in self:
            result.append(x)
        return tuple(result) + other

    def element_type(self):
        if self.hd is None:
            raise TypeError("Cannot compute element type of empty tuple!")
        else:
            type = typeof(self.hd)
            if self.tl is not None:
                type = promote(type, self.tl.element_type())
            return type



@typeof.case(tuple)
def typeof(pyval):
    valtypes = tuple(map(typeof, pyval))
    if len(pyval) <= 4:
        return StaticTuple[valtypes]
    return GenericTuple[reduce(promote, valtypes)]