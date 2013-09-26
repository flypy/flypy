# -*- coding: utf-8 -*-

"""
Support for 'opaque' functions, which are not implemented directly.
"""

from __future__ import print_function, division, absolute_import
from ..functionwrapper import FunctionWrapper

def implement_opaque(func, impl):
    assert isinstance(func, FunctionWrapper)
    assert func.implementor is None
    func.implementor = impl

def implement(func, py_func, argtypes, env):
    assert isinstance(func, FunctionWrapper)
    assert func.implementor is not None, func
    cache = env["numba.typing.cache"]
    impl = func.implementor(py_func, argtypes)
    cache.insert((func, argtypes), (impl, None))
    return impl
