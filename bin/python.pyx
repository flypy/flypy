# -*- coding: utf-8 -*-

"""
Python main interpreter, initialize the Boehm GC right from main().

NOTE: You may have to set PYTHONHOME to have it interpret your path correctly.
"""

from __future__ import print_function, division, absolute_import

import sys
from os.path import dirname, splitext


cdef extern from "gc.h":
    void GC_INIT()

GC_INIT()


def main(argv):
    assert len(argv) == 1
    [filename] = argv
    
    modname, ext = splitext(dirname(filename))
    globals = { '__file__': '__main__', '__name__': modname }
    code = compile(open(filename).read(), filename, 'exec', dont_inherit=True)
    eval(code, globals)

main(sys.argv[1:])