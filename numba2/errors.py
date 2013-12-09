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

class _ErrorMsg(object):
    def __init__(self, exc, lineno=-1, during=None, pyfunc=None):
        self.exc = exc
        self.lineno = lineno
        self.during = during
        self.pyfunc = pyfunc

    def __str__(self):
        try:
            return self.format()
        except Exception, e:
            return str(e)

    __repr__ = __str__

    def format(self, level=0):
        if self.pyfunc is not None:
            func = 'in function %s ' % _tell_func(self.pyfunc)
        else:
            func = ''

        if self.lineno >= 0:
            lineno = 'at line %d ' % self.lineno
        else:
            lineno = ''

        during = '%s ' % self.during if self.during else ''

        try:
            innerexc = self.exc.args[0]
        except IndexError:
            innerexc = self.exc
        if isinstance(innerexc, _ErrorMsg):
            inner = innerexc.format(level=level + 1)
        else:
            inner = 'Reason: %s' % innerexc

        if level == 0:
            leadline = '\n'
        else:
            leadline = '-' * level

        return "%s%s%s%s\n%s" % (leadline, during, func, lineno, inner)


@contextmanager
def error_context(lineno=-1, during=None, pyfunc=None):
    # Adapted from numbapro/npm/errors.py
    try:
        yield
    except Exception, e:
        em = _ErrorMsg(exc=e, pyfunc=pyfunc, during=during, lineno=lineno)
        exc = type(e)(em)
        raise exc, None, sys.exc_info()[2]


def error_context_phase(env, phase):
    return error_context(during=phase, pyfunc=env['numba.state.py_func'])

def error(env, phase, *msg):
    with error_context_phase(env, phase):
        raise CompileError(*msg)

def _tell_func(pyfunc):
    code = pyfunc.func_code
    filename = code.co_filename
    firstline = code.co_firstlineno
    return "%s (%s:%d)" % (pyfunc.__name__, filename, firstline)