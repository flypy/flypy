# -*- coding: utf-8 -*-

"""
Python's standard exception class hierarchy.

Exceptions found here are defined both in the exceptions module and the
built-in namespace.  It is recommended that user-defined exceptions
inherit from Exception.  See the documentation for the exception
inheritance hierarchy.
"""

from __future__ import print_function, division, absolute_import
import exceptions

from numba2 import overlay, typeof, jit

#===------------------------------------------------------------------===
# Decorator
#===------------------------------------------------------------------===

def ejit(exc_cls):
    py_cls = getattr(exceptions, exc_cls.__name__)
    exc_cls.layout = []
    exc_cls = jit(exc_cls)
    overlay(py_cls, exc_cls)

    @typeof.case(py_cls)
    def exc_typeof(pyval):
        return exc_cls[()]

    return exc_cls

#===------------------------------------------------------------------===
# Exceptions
#===------------------------------------------------------------------===

@ejit
class BaseException(object):
    """ Common base class for all exceptions """


@ejit
class Exception(BaseException):
    """ Common base class for all non-exit exceptions. """


@ejit
class StandardError(Exception):
    """
    Base class for all standard Python exceptions that do not represent
    interpreter exiting.
    """


@ejit
class ArithmeticError(StandardError):
    """ Base class for arithmetic errors. """


@ejit
class AssertionError(StandardError):
    """ Assertion failed. """


@ejit
class AttributeError(StandardError):
    """ Attribute not found. """


@ejit
class BufferError(StandardError):
    """ Buffer error. """


@ejit
class Warning(Exception):
    """ Base class for warning categories. """


@ejit
class BytesWarning(Warning):
    """
    Base class for warnings about bytes and buffer related problems, mostly
    related to conversion from str or comparing to str.
    """


@ejit
class DeprecationWarning(Warning):
    """ Base class for warnings about deprecated features. """


@ejit
class EnvironmentError(StandardError):
    """ Base class for I/O related errors. """


@ejit
class EOFError(StandardError):
    """ Read beyond end of file. """


@ejit
class FloatingPointError(ArithmeticError):
    """ Floating point operation failed. """


@ejit
class FutureWarning(Warning):
    """
    Base class for warnings about constructs that will change semantically
    in the future.
    """


@ejit
class GeneratorExit(BaseException):
    """ Request that a generator exit. """


@ejit
class ImportError(StandardError):
    """ Import can't find module, or can't find name in module. """


@ejit
class ImportWarning(Warning):
    """ Base class for warnings about probable mistakes in module imports """


@ejit
class SyntaxError(StandardError):
    """ Invalid syntax. """


@ejit
class IndentationError(SyntaxError):
    """ Improper indentation. """


@ejit
class LookupError(StandardError):
    """ Base class for lookup errors. """


@ejit
class IndexError(LookupError):
    """ Sequence index out of range. """


@ejit
class IOError(EnvironmentError):
    """ I/O operation failed. """


@ejit
class KeyboardInterrupt(BaseException):
    """ Program interrupted by user. """


@ejit
class KeyError(LookupError):
    """ Mapping key not found. """


@ejit
class MemoryError(StandardError):
    """ Out of memory. """


@ejit
class NameError(StandardError):
    """ Name not found globally. """


@ejit
class RuntimeError(StandardError):
    """ Unspecified run-time error. """


@ejit
class NotImplementedError(RuntimeError):
    """ Method or function hasn't been implemented yet. """


@ejit
class OSError(EnvironmentError):
    """ OS system call failed. """


@ejit
class OverflowError(ArithmeticError):
    """ Result too large to be represented. """


@ejit
class PendingDeprecationWarning(Warning):
    """
    Base class for warnings about features which will be deprecated
    in the future.
    """


@ejit
class ReferenceError(StandardError):
    """ Weak ref proxy used after referent went away. """


@ejit
class RuntimeWarning(Warning):
    """ Base class for warnings about dubious runtime behavior. """


@ejit
class StopIteration(Exception):
    """ Signal the end from iterator.next(). """


@ejit
class SyntaxWarning(Warning):
    """ Base class for warnings about dubious syntax. """


@ejit
class SystemError(StandardError):
    """
    Internal error in the Python interpreter.

    Please report this to the Python maintainer, along with the traceback,
    the Python version, and the hardware/OS platform and version.
    """


@ejit
class SystemExit(BaseException):
    """ Request to exit from the interpreter. """


@ejit
class TabError(IndentationError):
    """ Improper mixture of spaces and tabs. """


@ejit
class TypeError(StandardError):
    """ Inappropriate argument type. """


@ejit
class UnboundLocalError(NameError):
    """ Local name referenced but not bound to a value. """


@ejit
class ValueError(StandardError):
    """ Inappropriate argument value (of correct type). """


@ejit
class UnicodeError(ValueError):
    """ Unicode related error. """


@ejit
class UnicodeDecodeError(UnicodeError):
    """ Unicode decoding error. """


@ejit
class UnicodeEncodeError(UnicodeError):
    """ Unicode encoding error. """


@ejit
class UnicodeTranslateError(UnicodeError):
    """ Unicode translation error. """


@ejit
class UnicodeWarning(Warning):
    """
    Base class for warnings about Unicode related problems, mostly
    related to conversion problems.
    """


@ejit
class UserWarning(Warning):
    """ Base class for warnings generated by user code. """


@ejit
class ZeroDivisionError(ArithmeticError):
    """ Second argument to a division or modulo operation was zero. """
