# -*- coding: utf-8 -*-

"""
tuple implementation.
"""

from __future__ import print_function, division, absolute_import

from flypy import jit, sjit, ijit, cjit, abstract, typeof
from flypy.rules import typematch
from flypy.conversion import fromobject, toobject
from .noneobject import NoneType
from .iterators import counting_iterator
from .listobject import List

STATIC_THRESHOLD = 8

jit = cjit

@abstract
class Tuple(object):

    @jit('a -> bool')
    def __nonzero__(self):
        return bool(len(self))

    __bool__ = __nonzero__

@sjit('GenericTuple[T]')
class GenericTuple(Tuple):
    layout = [('_items', 'List[T]')]

    @jit('a -> T')
    def __getitem__(self, item):
        return self._items[item]

    @jit('a -> Iterator[T]')
    def __iter__(self):
        return iter(self._items)

    @jit('a -> int64')
    def __len__(self):
        return len(self._items)

    @jit('a -> a -> a')
    def __add__(self, other):
        return GenericTuple(self._items + other._items)

    @jit #('Iterable[a] -> GenericTuple[a]')
    def from_iterable(self, it):
        # TODO: class methods
        self._items.extend(it)


@sjit('StaticTuple[a, b]')
class StaticTuple(Tuple):
    layout = [('hd', 'a'), ('tl', 'b')]

    @jit
    def __init__(self, hd, tl):
        self.hd = hd
        self.tl = tl

    @jit # slice
    def __getitem__(self, item):
        # TODO: implement
        # TODO: variants
        return self

    @jit('a -> b : integral -> c')
    def __getitem__(self, item):
        if item == 0:
            return self.hd
        else:
            return self.tl[item - 1]

    @jit #('a -> Iterator[T]')
    def __iter__(self):
        return counting_iterator(self)

    @jit('a -> int64')
    def __len__(self):
        if self.hd is None:
            return 0
        elif self.tl is None:
            return 1
        else:
            return len(self.tl) + 1

    @jit('a -> StaticTuple[t1, t2] -> c')
    def __add__(self, other):
        if self.tl is None:
            return StaticTuple(self.hd, other)
        else:
            return StaticTuple(self.hd, self.tl + other)

    @jit('a -> Tuple[T] -> Tuple[T]')
    def __add__(self, other):
        result = List[T]()
        for x in self:
            result.append(x)
        return tuple(result) + other

    @jit('a -> a -> bool')
    def __eq__(self, other):
        return self.hd == other.hd and self.tl == other.tl

    @jit('a -> str')
    def __repr__(self):
        result = []
        t = self
        while not isinstance(t.tl, EmptyTuple):
            result.append(t.hd)
            t = t.tl

        result.append(t.hd)
        return repr(tuple(result))
        #return '(%s)' % ", ".join(map(str, self))

    # -------------------------

    def element_type(self):
        if self.hd is None:
            raise TypeError("Cannot compute element type of empty tuple!")
        else:
            type = typeof(self.hd)
            if self.tl is not None:
                type = promote(type, self.tl.element_type())
            return type

    @staticmethod
    def fromobject(tuple, type):
        head, tail = type.parameters
        hd = fromobject(tuple[0], head)
        if tuple[1:]:
            tl = fromobject(tuple[1:], tail)
        else:
            tl = EmptyTuple()

        return StaticTuple(hd, tl)

    @staticmethod
    def toobject(value, type):
        head, tail = type.parameters
        hd = toobject(value.hd, head)
        if isinstance(value.tl, EmptyTuple):
            return (hd,)
        return (hd,) + toobject(value.tl, tail)


@sjit
class EmptyTuple(Tuple):
    layout = []

    @jit
    def __getitem__(self, item):
        return 0
        #raise IndexError

    @jit
    def __iter__(self):
        return self

    @jit
    def next(self):
        raise StopIteration

    @jit('a -> b -> bool')
    def __eq__(self, other):
        return isinstance(other, EmptyTuple)

    @jit('a -> int64')
    def __len__(self):
        return 0

    # -------------------------

    @staticmethod
    def toobject(value, type):
        return ()


@jit('StaticTuple[a, b] -> a')
def head(t):
    return t.hd

@jit('StaticTuple[a, b] -> b')
def tail(t):
    return t.tl



def make_tuple_type(tup):
    """
    Create a flypy tuple type from a python tuple of element types.
    """
    if len(tup) < STATIC_THRESHOLD:
        result = EmptyTuple[()]
        for ty in reversed(tup):
            result = StaticTuple[ty, result]
        return result
    return GenericTuple[reduce(promote, tup)]


def extract_statictuple_eltypes(tuple_type):
    if typematch(tuple_type, EmptyTuple):
        return ()

    hd, tl = tuple_type.parameters
    return (hd,) + extract_statictuple_eltypes(tl)


def extract_tuple_eltypes(tuple_type, n):
    """
    Extract the element types from the (static) tuple type and return them
    in a tuple.
    """
    if typematch(tuple_type, StaticTuple):
        return extract_statictuple_eltypes(tuple_type)
    else:
        assert typematch(tuple_type, GenericTuple), tuple_type
        base_type = tuple_type.parameters[0]
        return (base_type,) * n


@typeof.case(tuple)
def typeof(pyval):
    valtypes = tuple(map(typeof, pyval))
    return make_tuple_type(valtypes)