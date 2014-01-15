# -*- coding: utf-8 -*-

"""
Boehm support utilities.
"""

# future feature absolute_import is not defined ?
from __future__ import print_function, division #, absolute_import

cdef extern from "gc.h":
    void GC_INIT()
    void GC_gcollect()
    void *GC_MALLOC(size_t nbytes)
    void GC_disable()
    void GC_enable()

    ctypedef void (*GC_finalization_proc) (void * obj, void * client_data)

    void GC_register_finalizer(void * obj, GC_finalization_proc fn,
                  void * cd, GC_finalization_proc *ofn,
                  void * *ocd)


GC_INIT()

cdef public void boehm_collect():
    GC_gcollect()

cdef public void *boehm_malloc(size_t nbytes):
    return GC_MALLOC(nbytes)

cdef public void boehm_disable():
    GC_disable()

cdef public void boehm_enable():
    GC_enable()

cdef public void boehm_register_finalizer(void *obj, void *dtor):
    cdef GC_finalization_proc old_finalizer
    cdef void *old_client_data

    GC_register_finalizer(obj, <GC_finalization_proc> dtor, NULL,
                          &old_finalizer, &old_client_data)
