# -*- coding: utf-8 -*-

"""
Garbage collection using the Boehm collector.
"""

from __future__ import print_function, division, absolute_import
import os

from numba2 import jit, typeof
from numba2.runtime import sizeof, cast, Pointer, Type
from . import boehmlib

import cffi

__all__ = ['gc_alloc']

root = os.path.dirname(os.path.abspath(__file__))
lib = os.path.join(root, "boehmlib.so")

#===------------------------------------------------------------------===
# Decls
#===------------------------------------------------------------------===

ffi = cffi.FFI()

ffi.cdef("""
void boehm_collect();
void *boehm_malloc(size_t nbytes);
""")

gc = ffi.dlopen(lib)

# We can't take the address of these :(

#gc = ffi.verify("""
##include <gc.h>
#
#void boehm_init(void) {
#    GC_INIT();
#}
#
#void boehm_collect(void) {
#    GC_gcollect();
#}
#
#void *boehm_malloc(size_t nbytes) {
#    return GC_MALLOC(nbytes);
#}
#""", libraries=["gc"])

#===------------------------------------------------------------------===
# Implementations
#===------------------------------------------------------------------===

@jit('int64 -> Type[a] -> Pointer[a]')
def gc_alloc(items, type):
    p = gc.boehm_malloc(items * sizeof(type))
    return cast(p, Pointer[type])

@jit
def gc_collect():
    gc.boehm_collect()
