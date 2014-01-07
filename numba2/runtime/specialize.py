# -*- coding: utf-8 -*-

"""
Specialization. Annotates functions for the compiler to process.
"""

from __future__ import print_function, division, absolute_import
import inspect
from .. import jit, annotate #, SpecializeError

def specialize_value(*args):
    """
    Specialize on values, which must be constant:

        @specialize_value('a', 'b')
    """
    def decorator(f):
        argspec = inspect.getargspec(f)
        for arg in args:
            if arg not in argspec.args and arg != args.varargs:
                raise SpecializeError(
                    "Arg %s listed for specialization not in argspec" % (arg,))
        annotate(f, specialize_value=args)
        return f
    return decorator

@jit('Iterable[a] -> Iterable[a]', specialize_value='iterable')
def unroll(iterable):
    """
    Fully unroll a constant-sized iterable. unroll is detected by the compiler.
    """
    return iterable
