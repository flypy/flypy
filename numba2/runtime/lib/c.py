# -*- coding: utf-8 -*-

"""
C library bindings.
"""

from __future__ import print_function, division, absolute_import

import cffi

#===------------------------------------------------------------------===
# Decls
#===------------------------------------------------------------------===

ffi = cffi.FFI()
ffi.cdef("""
void *malloc(size_t size);
int memcmp(void *s1, void *s2, size_t n);
int printf(char *s, ...);
int puts(char *s);
size_t strlen(char *s);
""")

libc = ffi.dlopen(None)
