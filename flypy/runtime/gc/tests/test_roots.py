# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import random
import ctypes
import unittest

from flypy import jit, cast, typeof, sizeof, NULL
from flypy.runtime.obj.core import newbuffer
from flypy.types import Pointer, float64, int32, int64, void, struct_
from flypy.runtime.gc import semispace as gc, roots

class TestBumpAllocator(unittest.TestCase):

    def test_root_finding(self):
        @jit
        def f():
            buf = newbuffer(frame_t, 2)
            frame1 = buf.pointer()
            frame1[0].prev = NULL

            frame2 = buf.pointer() + 1
            frame2[0].prev = frame1

            roots = newbuffer(Pointer[void], 5)
            roots[0] = root1
            roots[1] = NULL
            roots[2] = root2
            roots[3] = root3
            roots[4] = NULL

            frame1[0].trace_funcs = cast(roots.pointer(), Pointer[void])
            frame2[0].trace_funcs = cast(roots.pointer() + 2, Pointer[void])

            tracers = newbuffer(Pointer[void], 5)
            roots[0] = tracer1
            roots[1] = NULL
            roots[2] = tracer2
            roots[3] = tracer3
            roots[4] = NULL

            frame1[0].trace_funcs = cast(tracers.pointer(), Pointer[void])
            frame2[0].trace_funcs = cast(tracers.pointer() + 2, Pointer[void])

            # find_roots()

        frame_t = roots.StackFrame[()]
        root1 = ctypes.cast(14, ctypes.c_void_p)
        root2 = ctypes.cast(15, ctypes.c_void_p)
        root3 = ctypes.cast(16, ctypes.c_void_p)

        tracer1 = ctypes.cast(20, ctypes.c_void_p)
        tracer2 = ctypes.cast(21, ctypes.c_void_p)
        tracer3 = ctypes.cast(22, ctypes.c_void_p)


if __name__ == '__main__':
    unittest.main()