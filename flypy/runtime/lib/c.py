# -*- coding: utf-8 -*-

"""
C library bindings.
"""

from __future__ import print_function, division, absolute_import

from flypy.extern_support import extern_cffi

#===------------------------------------------------------------------===
# Decls
#===------------------------------------------------------------------===

libc, libc_cffi = extern_cffi(".flypy.runtime.c", None, """
void *malloc(size_t size);
void *realloc(void *ptr, size_t size);
void free(void *ptr);
int memcmp(void *s1, void *s2, size_t n);
int printf(char *s, ...);
int snprintf(char *str, size_t size, const char *format, ...);
int puts(char *s);
size_t strlen(char *s);
unsigned long clock();
""")
