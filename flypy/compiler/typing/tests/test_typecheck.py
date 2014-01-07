# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import ctypes.util
import unittest

from flypy import jit
from flypy.errors import UnificationError, InferError

libc = ctypes.CDLL(ctypes.util.find_library("c"))

libc.printf.argtypes = [ctypes.c_char_p]
libc.restype = ctypes.c_int

#===------------------------------------------------------------------===
# Tests
#===------------------------------------------------------------------===

class TestTypeCheck(unittest.TestCase):

    def test_call_ctypes_mismatch_argno(self):
        @jit
        def simple():
            libc.printf("hello", "world")

        try:
            simple()
        except Exception, e:
            self.assertTrue(isinstance(e, (UnificationError, TypeError)))
        else:
            self.fail("Expecting UnificationError")


    def test_call_ctypes_mismatch_argtypes(self):
        @jit
        def simple():
            libc.printf(10)

        try:
            simple()
        except Exception, e:
            self.assertTrue(isinstance(e, (UnificationError, TypeError)))
        else:
            self.fail("Expecting UnificationError")


if __name__ == '__main__':
    unittest.main()