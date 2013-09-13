# -*- coding: utf-8 -*-

"""
Contains all numba errors.
"""

from __future__ import print_function, division, absolute_import
from contextlib import contextmanager
import sys


class error(Exception):
    """Superclass of all numba exceptions"""

class CompileError(error):
    """General purpose compilation error"""

class EmptyStackError(CompileError):
    """
    Raised when the bytecode emulation stack is empty and we're trying to
    access an operand.
    """

class InferError(CompileError):
    """Raised for type inference errors"""

class SpecializeError(CompileError):
    """
    Raised when we fail to specialize on something requested for some reason.
    """

@contextmanager
def error_context(lineno=-1, during=None):
    # Adapted from numbapro/npm/errors.py
    try:
        yield
    except Exception, e:
        msg = []
        if lineno >= 0:
            msg.append('At line %d:' % lineno)
        if during:
            msg.append('During: %s' % during)

        if isinstance(e, AssertionError):
            msg.append('Internal error: %s' % e)
        elif isinstance(e, error):
            msg.append(str(e))
        else:
            msg.append('%s: %s' % (type(e).__name__, e))

        exc = error('\n'.join(msg))
        raise exc, None, sys.exc_info()[2]