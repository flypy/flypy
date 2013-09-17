# -*- coding: utf-8 -*-

"""
Utilities!
"""

from __future__ import print_function, division, absolute_import

import types
import functools

def applyable_decorator(decorator):
    """
    Construct an applyable decorator, that always calls the decorator
    with the function and any optional args and keyword args:

        @applyable_decorator
        def decorator(f, *args, **kwds):
            ...

    Then use `decorator` as follows:

        @decorator
        def foo():
            ...

    or

        @decorator(value='blah')
        def foo():
            ...

    This does not work if the arguments to `decorator` are themselves
    functions!
    """
    @functools.wraps(decorator)
    def decorator_wrapper(*args, **kwargs):
        if (len(args) == 1 and not kwargs and
                isinstance(args[0], types.FunctionType)):
            return decorator(args[0])
        else:
            return lambda f: decorator(f, *args, **kwargs)

    return decorator_wrapper