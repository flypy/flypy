"""
External references.

Numba runtime and library code should avoid injecting runtime addresses into
the generated code. Instead, it should uses undefined external symbols that
are resolved by LLVM JIT by mapping runtime addresses for to external symbols.
"""

from __future__ import print_function, division, absolute_import
from numba2 import typeof, cffi_support, types
from pykit.utils import dylib_support

class ExternalSymbol(object):
    """Represent an external symbol
    """

    def __init__(self, name, typ):
        self.name = name
        self.type = typ
        self.mapping = {}

    def add_mapping(self, target, pointer):
        self.mapping[target] = pointer

    def get_mapping(self, target):
        return self.mapping[target]

    def __repr__(self):
        return "ExternalSymbol(%s, %s)" % (self.name, self.type)


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
    # Adapt a ForeignFunction to a Function type
    foreignfunc = typeof(value)
    functy = types.Function[foreignfunc.parameters]
    # Build external symbol
    es = ExternalSymbol(name, functy)
    address = cffi_support.get_pointer(value)
    es.add_mapping("cpu", address)
    dylib_support.install(name, address)
    return es


def externlib(prefix, lib, symbols):
    if isinstance(symbols, str):
        symbols = symbols.split()
    fullnames = ['.'.join((prefix, sym)) for sym in symbols]
    extlib = ExternalLibrary()
    for raw, mangled in zip(symbols, fullnames):
        setattr(extlib, raw, extern(mangled, getattr(lib, raw)))
    return extlib

