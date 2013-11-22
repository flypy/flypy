# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import warnings
from ctypes import Structure, POINTER, c_longlong, c_uint, c_int, \
    c_ubyte, c_char_p, byref, pointer

from numba2 import jit, typeof
from numba2.runtime.obj.core import (struct_, Pointer, from_cstring, String)
from numba2.support import libfinder

class mpd_context(Structure):
    
    _fields_ = [('prec', c_longlong),
                ('emax', c_longlong),
                ('emin', c_longlong),
                ('trap', c_uint),
                ('status', c_uint),
                ('newtrap', c_uint),
                ('round', c_int),
                ('clamp', c_int),
                ('allcr', c_int)]

class mpd_t(Structure):

    _fields_ = [('flags', c_ubyte),
                ('exp', c_longlong),
                ('digits', c_longlong),
                ('len', c_longlong),
                ('alloc', c_longlong),
                ('data', POINTER(c_uint))]

msg = "mpdec library not found, no accelerated decimals supported"
try:
    dll = libfinder.open_lib_ctypes(libfinder.find_lib("mpdec"))
except OSError, e:
    warnings.warn(msg)
    raise ImportError("%s: %s" % (msg, str(e)))

dll.mpd_new.argtypes = []
dll.mpd_new.restype = POINTER(mpd_t)

dll.mpd_del.argtypes = [POINTER(mpd_t)]
dll.mpd_del.restype = None

dll.mpd_set_string.argtypes = [POINTER(mpd_t), c_char_p, POINTER(mpd_context)]
dll.mpd_set_string.restype = None

dll.mpd_to_sci.argtypes = [POINTER(mpd_t), c_int]
dll.mpd_to_sci.restype = c_char_p

dll.mpd_add.argtypes = [POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_context)]
dll.mpd_add.restype = None

dll.mpd_sub.argtypes = [POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_context)]
dll.mpd_sub.restype = None

dll.mpd_mul.argtypes = [POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_context)]
dll.mpd_mul.restype = None

dll.mpd_div.argtypes = [POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_context)]
dll.mpd_div.restype = None

dll.mpd_compare.argtypes = [POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_t), POINTER(mpd_context)]
dll.mpd_compare.restype = c_int

mpd_new_func = dll.mpd_new
mpd_del_func = dll.mpd_del
mpd_set_string_func = dll.mpd_set_string
mpd_to_sci_func = dll.mpd_to_sci
mpd_add_func = dll.mpd_add
mpd_sub_func = dll.mpd_sub
mpd_mul_func = dll.mpd_mul
mpd_div_func = dll.mpd_div
mpd_cmp_func = dll.mpd_compare


context = mpd_context()
context_ref = pointer(context)
dll.mpd_init(byref(context))


@jit
class _Decimal(object):
    layout = [('mpd', Pointer[typeof(mpd_t)])]
    
    @jit
    def __init__(self, mpd):
        
        #print('__init__')
        self.mpd = mpd

    @jit
    def __del__(self):
        
        #print('__del__')
        mpd_del_func(self.mpd)

    @jit
    def __repr__(self):
       
        return 'Decimal'

    @jit
    def __str__(self):
        
        return from_cstring(mpd_to_sci_func(self.mpd, 0))

    @jit
    def __add__(self, right):
        
        mpd_result = mpd_new_func()
        mpd_add_func(mpd_result, self.mpd, right.mpd, context_ref)
        return _Decimal(mpd_result)

    @jit
    def __sub__(self, right):
        
        mpd_result = mpd_new_func()
        mpd_sub_func(mpd_result, self.mpd, right.mpd, context_ref)
        return _Decimal(mpd_result)

    @jit
    def __mul__(self, right):
        mpd_result = mpd_new_func()
        mpd_mul_func(mpd_result, self.mpd, right.mpd, context_ref)
        return _Decimal(mpd_result)

    @jit
    def __lt__(self, right):
        mpd_temp = mpd_new_func()
        result = mpd_cmp_func(mpd_temp, self.mpd, right.mpd, context_ref)
        mpd_del_func(mpd_temp)
        if result == -1:
            return True
        return False

    @jit
    def __gt__(self, right):
        mpd_temp = mpd_new_func()
        result = mpd_cmp_func(mpd_temp, self.mpd, right.mpd, context_ref)
        mpd_del_func(mpd_temp)
        if result == 1:
            return True
        return False


@jit
def decimal(value):
    new_mpd = mpd_new_func()
    d = _Decimal(new_mpd)
    mpd_set_string_func(d.mpd, value.buf.p, context_ref)
    return d
    
