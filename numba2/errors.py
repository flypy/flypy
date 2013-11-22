# -*- coding: utf-8 -*-

"""
Contains all numba errors.
"""

from __future__ import print_function, division, absolute_import
from contextlib import contextmanager
import sys

from blaze.error import UnificationError

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
def error_context(lineno=-1, during=None, pyfunc=None):
    # Adapted from numbapro/npm/errors.py
    try:
        yield
    except Exception, e:
        msg = []
        if pyfunc is not None:
            during = during or ''
            during += ' - %s' % _tell_func(pyfunc)

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

        exctype = type(e)
        exc = exctype('\n'.join(msg))
        raise exc, None, sys.exc_info()[2]


def error_context_phase(env, phase):
    return error_context(during=phase, pyfunc=env['numba.state.py_func'])


def _tell_func(pyfunc):
    code = pyfunc.func_code
    filename = code.co_filename
    firstline = code.co_firstlineno
    return "%s (%s:%d)" % (pyfunc.__name__, filename, firstline)