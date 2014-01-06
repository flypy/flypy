# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest

from numba2 import jit

range = lambda *args: list(xrange(*args))

class TestListCreation(unittest.TestCase):

    def test_empty_builtin(self):
        raise unittest.SkipTest("list() call")

        @jit
        def empty():
            return list()

        self.assertEqual(empty(), [])

    def test_empty(self):
        @jit
        def empty():
            return []

        self.assertEqual(empty(), [])

    def test_creation(self):
        @jit
        def create():
            return [1, 2, 3]

        self.assertEqual(create(), [1, 2, 3])


class TestListSpecialMethods(unittest.TestCase):

    def test_getitem(self):
        @jit
        def getitem(lst):
            return lst[3]

        self.assertEqual(getitem([4, 5, 6, 7, 8, 9]), 7)

    def test_setitem(self):
        @jit
        def setitem(lst):
            lst[4] = 19
            return lst

        result = setitem(range(10))
        expected = setitem.py_func(range(10))
        self.assertEqual(result, expected)

    def test_len(self):
        @jit
        def length(lst):
            return len(lst)

        self.assertEqual(length(range(10)), 10)

    def test_add(self):
        @jit
        def add(lst1, lst2):
            return lst1 + lst2

        self.assertEqual(add(range(10), []), range(10))
        self.assertEqual(add([], range(10, 20)), range(10, 20))

        raise unittest.SkipTest
        self.assertEqual(add(range(10), range(10, 20)), range(20))


class TestListMethods(unittest.TestCase):

    def test_append1(self):
        @jit
        def append(lst):
            l1 = len(lst)

            lst.append(0)
            l2 = len(lst)

            lst.append(1)
            l3 = len(lst)

            lst.append(2)
            l4 = len(lst)

            return l1, l2, l3, l4

        self.assertEqual(append([1]), (1, 2, 3, 4))

    def test_append2(self):
        @jit
        def append(lst):
            lst.append(1)
            lst.append(2)
            lst.append(3)
            return lst

        self.assertEqual(append([0]), [0, 1, 2, 3])

    def test_pop(self):
        @jit
        def pop(lst):
            lst.pop()
            lst.pop()
            return lst

        self.assertEqual(pop(range(5)), [0, 1, 2])

    def test_insert(self):
        @jit
        def insert(lst):
            lst.insert(2, 8)
            return lst

        self.assertEqual(insert(range(5)), [0, 1, 8, 2, 3, 4])

    def test_remove(self):
        @jit
        def remove(lst):
            lst.remove(4)
            return lst

        self.assertEqual(remove([2, 4, 5, 4]), [2, 5, 4])

    def test_reverse(self):
        @jit
        def reverse(lst):
            lst.reverse()
            return lst

        self.assertEqual(reverse(range(5)), range(5)[::-1])

    def test_index(self):
        raise unittest.SkipTest

        @jit
        def index(lst):
            return lst.index(4+2j)

        self.assertEqual(index([2+2j, 7+1j, 4+2j, 3+1j]), 2)

    def test_count(self):
        raise unittest.SkipTest

        @jit
        def count(lst):
            return lst.count(4+2j)

        self.assertEqual(count([2+2j, 7+1j, 4+2j, 4+2j, 3+1j, 4+2j]), 3)

    def test_iter(self):
        raise unittest.SkipTest

        @jit
        def iterate(lst1, lst2):
            for x in lst1:
                lst2.append(x)
            return lst2

        self.assertEqual(iterate(range(5), [-1]), range(-1, 5))


if __name__ == '__main__':
    unittest.main(verbosity=3)