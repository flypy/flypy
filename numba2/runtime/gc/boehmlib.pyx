# -*- coding: utf-8 -*-

"""
Boehm support utilities.
"""

from __future__ import print_function, division, absolute_import

cdef extern from "gc.h":
    void GC_INIT()
    void GC_gcollect()
    void *GC_MALLOC(size_t nbytes)

GC_INIT()

cdef public void boehm_collect():
    GC_gcollect()

cdef public void *boehm_malloc(size_t nbytes):
    return GC_MALLOC(nbytes)