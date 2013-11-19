# -*- coding: utf-8 -*-

"""
C library bindings.
"""

from __future__ import print_function, division, absolute_import

import cffi
from numba2.extern_support import externlib

#===------------------------------------------------------------------===
# Decls
#===------------------------------------------------------------------===

ffi = cffi.FFI()
ffi.cdef("""
void *malloc(size_t size);
void free(void *ptr);
int memcmp(void *s1, void *s2, size_t n);
int printf(char *s, ...);
int snprintf(char *str, size_t size, const char *format, ...);
int puts(char *s);
size_t strlen(char *s);
""")

libclib = ffi.dlopen(None)
libc = externlib(".numba.runtime.c", libclib, '''
malloc
free
memcmp
printf
snprintf
puts
strlen
''')
