# -*- coding: utf-8 -*-

"""
Garbage collection using the Boehm collector.
"""

from __future__ import print_function, division, absolute_import
import os

import numba2
from numba2 import cjit
from numba2.types import Pointer, void
from numba2.runtime.ffi import sizeof, cast
from numba2.runtime.obj.core import Type
from . import boehmlib
from numba2.extern_support import extern_cffi

__all__ = ['gc_alloc']

root = os.path.dirname(os.path.abspath(__file__))
lib = os.path.join(root, "boehmlib.so")

#===------------------------------------------------------------------===
# Decls
#===------------------------------------------------------------------===

gc, gclib_cffi = extern_cffi(".numba.runtime.gc", lib, """
void boehm_collect();
void *boehm_malloc(size_t nbytes);
void boehm_disable();
void boehm_enable();
void boehm_register_finalizer(void *obj, void *dtor);
""")


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

@cjit('int64 -> Type[a] -> Pointer[void]')
def gc_alloc(items, type):
    p = gc.boehm_malloc(items * sizeof(type))
    return p #cast(p, Pointer[type])

@cjit('int64 -> Type[a] -> Pointer[a]')
def gc_delalloc(items, type):
    p = gc.boehm_malloc(items * sizeof(type))

    return cast(p, Pointer[type])

@cjit
def gc_collect():
    gc.boehm_collect()

@cjit
def gc_disable():
    gc.boehm_disable()

@cjit
def gc_enable():
    gc.boehm_enable()

@cjit('Pointer[void] -> Pointer[void] -> void')
def gc_add_finalizer(obj, finalizer):
    gc.boehm_register_finalizer(obj, finalizer)