"""
Implement objects.

Required compiler support:

    1) Special methods (__add__, etc)
    2) ctypes/cffi support
    3) Scoping (__del__)
"""

from core import structclass, Struct, signature, Py_ssize_t, void

import cffi
ffi = cffi.FFI()
ffi.cdef("PyObject *add(PyObject *a, PyObject *b);", override=True)
lib = ffi.dlopen(None)

PyObject = Struct([
    ('ob_refcnt', Py_ssize_t),
    ('ob_type', void.pointer()),
])

@structclass(inline=['*'])
class Object(object):
    """
    Implement objects using the CPython C-API.

    NOTE: a numba <-> Cython bridge would make this even easier
    """

    layout = Struct([('obj', PyObject.pointer())])

    def __init__(self, ptr):
        self.obj = ptr
        self.incref()

    def incref(self):
        pass # TODO: implement

    def decref(self):
        pass # TODO: implement

    def __del__(self):
        self.decref()

    @signature('Object -> Object -> Object')
    def __add__(self, other):
        return lib.add(self, other) # PyNumber_Add()

    # ...
