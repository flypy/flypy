# -*- coding: utf-8 -*-

"""
Exception model.
"""

from numba2.runtime.obj.exceptions import Exception
from numba2.runtime.obj import Type

from pykit.ir import interp, Const

class ExcModel(interp.ExceptionModel):

    def __init__(self, env):
        self.env = env

    def exc_op_match(self, exc_type_const, exc):
        assert isinstance(exc_type_const, Const)

        context = self.env['numba.typing.context']
        exc_type = context[exc_type_const]
        type = context[exc]

        return self.exc_match(exc_type, type)

    def exc_match(self, exc_type, exception):
        return exc_match(exc_type, exception)


def exc_match(exc_type, exception):
    """
    See whether `exception` matches `exc_type`
    """
    if exc_type.impl == Type: # Type[Exception]
        exc_type = exc_type.parameters[0]
    if exception.impl == Type: # Type[Exception]
        exception = exception.parameters[0]

    assert issubclass(exception.impl, Exception), exception.impl
    assert issubclass(exc_type.impl, Exception), exc_type.impl

    return issubclass(exception.impl, exc_type.impl)
