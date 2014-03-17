# -*- coding: utf-8 -*-

"""
External references.

flypy runtime and library code should avoid injecting runtime addresses into
the generated code. Instead, it should uses undefined external symbols that
are resolved by LLVM JIT by mapping runtime addresses for to external symbols.
"""

from __future__ import print_function, division, absolute_import
from flypy.support import cffi_support
from pykit.utils import dylib_support
from flypy import coretypes, typeof
import cffi

class ExternalSymbol(object):
    """Represent an external symbol
    """

    def __init__(self, name, typ, ptr):
        """
        :param name: symbol name
        :param typ: symbol type; usually function type
        :param ptr: cpu pointer for the default target
        """
        self.name = name
        self.type = typ
        self.pointer = ptr

    def __repr__(self):
        return "ExternalSymbol(%s, %s)" % (self.name, self.type)

    def install(self):
        """Install symbol for the CPU target.
        Safe to call this multiple times for the same symbol.
        """
        if not dylib_support.has(self.name):
            dylib_support.install(self.name, self.pointer)

class ExternalLibrary(object):
    """
    A dummy object to house ExternalSymbol instances
    """
    pass

# --- utils

def is_extern_symbol(pyval):
    return isinstance(pyval, ExternalSymbol)

# --- shorthand

def extern(name, ffiobj):
    """Creates an ExternalSymbol object and bind CFFI value to the default
    target "cpu".
    """
    # Get type
    foreignfunc = typeof(ffiobj)
    functy = coretypes.Function[foreignfunc.parameters]
    functy.varargs = foreignfunc.varargs
    # Get pointer
    ptr = cffi_support.get_pointer(ffiobj)

    return ExternalSymbol(name, functy, ptr)


def externlib(prefix, lib, symbols):
    """
    Wraps external libraries and expose them as external library.
    """
    if isinstance(symbols, str):
        symbols = symbols.split()
    fullnames = ['.'.join((prefix, sym)) for sym in symbols]
    extlib = ExternalLibrary()
    for raw, mangled in zip(symbols, fullnames):
        setattr(extlib, raw, extern(mangled, getattr(lib, raw)))
    return extlib

def extern_cffi(prefix, dll_path, declstr):
    ffi = cffi.FFI()
    ffi.cdef(declstr)
    clib = ffi.dlopen(dll_path)
    symbols = _parse_cdecl(declstr)
    return externlib(prefix, clib, symbols), clib

# ----- cdecl

def _parse_cdecl(declstr):
    parser = cffi.cparser.Parser()
    parser.parse(declstr)
    functions = []
    # Get all the functions
    for k in parser._declarations:
        typ, val = k.split()
        if typ == 'function':
            functions.append(val)
    return functions

# ----- overloads

@typeof.case(ExternalSymbol)
def typeof(pyval):
    return pyval.type
