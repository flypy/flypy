# -*- coding: utf-8 -*-

"""
Bytecode and serialization for the purpose of defining a key to find cached
IR.
"""

from __future__ import print_function, division, absolute_import
import zlib
import types

#===------------------------------------------------------------------===
# Errors
#===------------------------------------------------------------------===

class IncompatibleConstantError(object):
    pass

#===------------------------------------------------------------------===
# Blobbify
#===------------------------------------------------------------------===

def make_code_blob(py_func, argtypes):
    """
    Create a code "blob" for the given Python function and numba argument
    types.

    Return
    ------
    A json string encoding the function and argument types structurally.
    """
    try:
        code = code_tuple(py_func)
    except IncompatibleConstantError:
        return None

    result = str((code, tuple(map(qualify, argtypes))))
    #result = zlib.compress(result)
    return result

def code_tuple(func):
    """Build a tuple for the code object"""
    attributes = ['argcount', 'code', 'filename', 'firstlineno', 'flags',
                  'freevars', 'lnotab', 'name', 'nlocals', 'stacksize']
    attrs = [getattr(func.func_code, 'co_' + attrib) for attrib in attributes]
    attrs.append([encode_constant(const) for const in func.func_code.co_consts])
    attrs.append([encode_constant(const) for const in find_globals(func)])
    return tuple(attrs)

def find_globals(func):
    """Load any globals references by the function"""
    global_names = func.func_code.co_names
    #return [func.func_globals[name] for name in global_names]
    return global_names

#===------------------------------------------------------------------===
# Constants
#===------------------------------------------------------------------===

def compatible_const(const):
    """See whether we can blobify the constant"""
    if isinstance(const, tuple):
        return all(map(compatible_const, const))
    return isinstance(const, (types.NoneType, bool, int, float, str, complex))

def encode_constant(const):
    """Return a string-encodable representation for `const` that is compatible"""
    # TODO: use 'conversion' and mutability flags to determine whether `const`
    #       can be serialized

    if not compatible_const(const):
        raise IncompatibleConstantError(const)

    return {'type': type(const).__name__, 'value': const}

#===------------------------------------------------------------------===
# Types
#===------------------------------------------------------------------===

def qualify(ty):
    """Qualify the type"""
    name = ".".join([ty.impl.__module__, ty.impl.__name__])
    return str((name, ty))