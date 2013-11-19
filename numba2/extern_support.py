"""
External references.

Numba runtime and library code should avoid injecting runtime addresses into
the generated code. Instead, it should uses undefined external symbols that
are resolved by LLVM JIT by mapping runtime addresses for to external symbols.
"""

from __future__ import print_function, division, absolute_import
from pykit.utils import dylib_support

class ExternalSymbol(object):
    """Represent an external symbol
    """

    def __init__(self, name, ffiobj):
        self.name = name
        self.ffiobj = ffiobj
        self._type = None
        self._ptr = None

    @property
    def type(self):
        if self._type is None:
            # This indirection is necessary to make the import works
            from numba2 import typeof, types
            foreignfunc = typeof(self.ffiobj)
            functy = types.Function[foreignfunc.parameters]
            functy.varargs = foreignfunc.varargs
            self._type = functy
        return self._type

    @property
    def pointer(self):
        if self._ptr is None:
            # This indirection is necessary to make the import works
            from numba2 import cffi_support
            self._ptr = cffi_support.get_pointer(self.ffiobj)
        return self._ptr


    def __repr__(self):
        return "ExternalSymbol(%s, %s)" % (self.name, self.type)

    def install(self):
        if not dylib_support.has(self.name):
            dylib_support.install(self.name, self.pointer)



class ExternalLibrary(object):
    pass

# --- utils

def is_extern_symbol(pyval):
    return isinstance(pyval, ExternalSymbol)

def from_extern_symbol(extsym):
    """Get Numba type from the symbol
    """
    return extsym.type

# --- shorthand

def extern(name, value):
    """Creates an ExternalSymbol object and bind CFFI value to the default
    target "cpu".
    """
    es = ExternalSymbol(name, value)
    return es


def externlib(prefix, lib, symbols):
    if isinstance(symbols, str):
        symbols = symbols.split()
    fullnames = ['.'.join((prefix, sym)) for sym in symbols]
    extlib = ExternalLibrary()
    for raw, mangled in zip(symbols, fullnames):
        setattr(extlib, raw, extern(mangled, getattr(lib, raw)))
    return extlib

