# -*- coding: utf-8 -*-

"""
List implementation.
"""

from __future__ import print_function, division, absolute_import

from numba2 import jit, typeof
from numba2.conversion import toobject, fromobject, toctypes
from .typeobject import Type
from .pointerobject import Pointer
from .bufferobject import newbuffer, Buffer
from .iterators import counting_iterator

keepalive = []

#===------------------------------------------------------------------===
# List Constants
#===------------------------------------------------------------------===

INITIAL_BUFSIZE = 10
SHRINK = 1.5
GROW = 2

#===------------------------------------------------------------------===
# List Implemention
#===------------------------------------------------------------------===

@jit('List[a]')
class List(object):

    layout = [('base_type', 'Type[a]'),
              ('buf', 'Buffer[a]'),
              ('size', 'int64')]

    @jit('List[a] -> Type[a] -> Buffer[a] -> int64 -> void')
    def __init__(self, base_type, buf, size=0):
        self.base_type = base_type
        self.buf = buf
        self.size = size

    # ---- Special methods ---- #

    # TODO: Slicing

    @jit('List[a] -> int64 -> a')
    def __getitem__(self, key):
        if not (0 <= key < self.size):
            # TODO: Exceptions !
            # raise IndexError(key)
            pass

        return self.buf[key]

    @jit('List[a] -> int64 -> a -> void')
    def __setitem__(self, key, value):
        if not (0 <= key < self.size):
            # TODO: Exceptions !
            # raise IndexError(key)
            pass

        self.buf[key] = value

    @jit #('a -> Iterator[T]')
    def __iter__(self):
        return counting_iterator(self)

    @jit
    def __len__(self):
        return self.size

    @jit
    def __repr__(self):
        buf = ", ".join([str(self.buf[i]) for i in range(self.size)])
        return "[" + buf + "]"

    @jit('List[a] -> List[a] -> List[a]')
    def __add__(self, other):
        buf = newbuffer(self.base_type, len(self) + len(other))
        result = List(self.base_type, buf)
        result.extend(self)
        result.extend(other)
        return result

    @jit('List[a] -> EmptyList[] -> List[a]')
    def __add__(self, other):
        return self

    # ---- List Methods ---- #

    @jit('List[a] -> a -> void')
    def append(self, value):
        self._grow()
        self.buf[self.size] = value
        self.size += 1

    @jit #('List[a] -> Iterable[a] -> void')
    def extend(self, iterable):
        for obj in iterable:
            self.append(obj)

    @jit('List[a] -> a -> int64')
    def index(self, value):
        buf = self.buf
        for i in range(self.size):
             if buf[i] == value:
                 return i

        # raise ValueError
        return -1

    @jit('List[a] -> a -> int64')
    def count(self, value):
        count = 0
        for i in range(self.size):
             if self.buf[i] == value:
                 count += 1

        return count

    @jit('List[a] -> a')
    def pop(self):
        # TODO: Optional argument 'index'
        size = self.size - 1
        if size < 0:
            #raise IndexError
            pass

        item = self.buf[size]
        self.size = size
        self._shrink()
        return item

    @jit('List[a] -> int64 -> a -> void')
    def insert(self, index, value):
        size = self.size
        self._grow()

        if index > size:
            self.append(value)
        else:
            current = self.buf[index]
            self.buf[index] = value
            for i in range(index + 1, size + 1):
                next = self.buf[i]
                self.buf[i] = current
                current = next

        self.size = size + 1

    @jit('List[a] -> a -> void')
    def remove(self, value):
        size = self.size
        position = 0
        found = False

        while position < size and not found:
            if self.buf[position] == value:
                found = True
            else:
                position += 1

        if found:
            for i in range(position, size):
                self.buf[i] = self.buf[i + 1]

            self.size = size - 1
            # raise ValueError 'not in list'

    @jit
    def reverse(self):
        buf = self.buf
        size = self.size - 1
        for i in range(self.size / 2):
            tmp = buf[i]
            buf[i] = buf[size - i]
            buf[size - i] = tmp

    @jit
    def sort(self):
        # TODO: optional arguments cmp, key, reverse
        raise NotImplementedError

    # ---- Helpers ---- #

    @jit
    def _shrink(self):
        if INITIAL_BUFSIZE < self.size < len(self.buf) / 2:
            self.buf.resize(int(SHRINK * self.size))

    @jit
    def accommodate(self, n):
        """Accommodate for an additional N objects"""
        if self.size + n >= len(self.buf):
            self.buf.resize(int(self.size + n))

    @jit
    def _grow(self):
        if self.size >= len(self.buf):
            self.buf.resize(int(self.size * GROW))

    # ---- Python <-> Numba ---- #

    @staticmethod
    def fromobject(lst, type):
        [base_type] = type.parameters

        # Allocate and populate buffer
        bufsize = len(lst) + 2
        buf = newbuffer(type.parameters[0], bufsize)
        for i, item in enumerate(lst):
            buf[i] = item # TODO: Convert item to ctypes representation

        result = List(base_type, buf)
        result.size = len(lst)
        return result

    @staticmethod
    def toobject(obj, type):
        [base_type] = type.parameters
        return [toobject(obj[i], base_type) for i in xrange(len(obj))]

#===------------------------------------------------------------------===
# Empty List
#===------------------------------------------------------------------===

@jit
class EmptyList(List):
    layout = []

    @jit('a -> a -> a')
    def __add__(self, other):
        return self

    @jit('a -> List[b] -> List[b]')
    def __add__(self, other):
        return other

    @staticmethod
    def fromobject(lst, type):
        return EmptyList()

    @staticmethod
    def toobject(obj, type):
        return []

#===------------------------------------------------------------------===
# typeof
#===------------------------------------------------------------------===

@typeof.case(list)
def typeof(pyval):
    if pyval:
        types = [typeof(x) for x in pyval]
        if len(set(types)) != 1:
            raise TypeError("Got multiple types for elements, %s" % set(types))
        return List[types[0]]
    else:
        return EmptyList[()]