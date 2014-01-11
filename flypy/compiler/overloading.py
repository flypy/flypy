# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import inspect

from flypy.typing import resolve, to_blaze

from datashape import overloading
from datashape.overloading import lookup_previous
from datashape.overloading import overload, Dispatcher, flatargs as simple_flatargs

def overloadable(f):
    """
    Make a function overloadable, useful if there's no useful defaults to
    overload on
    """
    return Dispatcher()

# TODO: Cache results
def resolve_overloads(o, scope, bound):
    """
    Resolve the signatures of overloaded methods in the given scope. Further
    resolve any type variables listed in `bound` by their replacement.
    """
    result = Dispatcher()
    for (f, signature, kwds) in o.overloads:
        new_sig = resolve(signature, scope, bound)
        new_sig = to_blaze(new_sig) # Use blaze's coercion rules for now
        overload(new_sig, result, **kwds)(f)
    return result

def best_match(func_wrapper, argtypes):
    """
    Find the right overload for a flypy function.

    Arguments
    ---------
    func_wrapper: FunctionWrapper
        The function

    argtypes: [Type]
        Types to call the overloaded function with

    Returns
    -------
    (py_func, result_signature)
    """
    o = func_wrapper.dispatcher
    scope = determine_scope(func_wrapper.py_func)
    bound = {} # TODO:
    overloaded = resolve_overloads(o, scope, bound)
    argtypes = [to_blaze(t) for t in argtypes]

    overload = overloading.best_match(overloaded, argtypes)
    signature = resolve(overload.resolved_sig, scope, bound)
    return (overload.func, signature, overload.kwds)


def determine_scope(py_func):
    return py_func.__globals__

def flatargs(f, args, kwargs, argspec=None):
    """
    Return a single args tuple matching the actual function signature, with
    extraneous args appended to a new tuple 'args' and extraneous keyword
    arguments inserted in a new dict 'kwargs'.

        >>> def f(a, b=2, c=None): pass
        >>> flatargs(f, (1,), {'c':3})
        (1, 2, 3)
        >>> flatargs(f, (), {'a': 1})
        (1, 2, None)
        >>> flatargs(f, (1, 2, 3), {})
        (1, 2, 3)
        >>> flatargs(f, (2,), {'a': 1})
        Traceback (most recent call last):
            ...
        TypeError: f() got multiple values for keyword argument 'a'
    """
    argspec = inspect.getargspec(f) if argspec is None else argspec
    defaults = argspec.defaults or ()
    kwargs = dict(kwargs)

    def unreachable():
        f(*args, **kwargs)
        assert False, "unreachable"

    # -------------------------------------------------
    # Insert defaults

    tail = min(len(defaults), len(argspec.args) - len(args))
    if tail:
        for argname, default in zip(argspec.args[-tail:], defaults[-tail:]):
            kwargs.setdefault(argname, default)

    # -------------------------------------------------
    # Parse Keywords f(arg=val)

    extra_args = []
    for argpos in range(len(args), len(argspec.args)):
        argname = argspec.args[argpos]
        if argname not in kwargs:
            unreachable()

        extra_args.append(kwargs[argname])
        kwargs.pop(argname)

    # -------------------------------------------------
    # varargs (*args)

    args = args + tuple(extra_args)
    if len(args) > len(argspec.args):
        if not argspec.varargs:
            unreachable()
        args, trailing = args[:len(argspec.args)], args[len(argspec.args):]
        args = args + (trailing,)
    elif argspec.varargs:
        args += ((),)

    # -------------------------------------------------
    # keywords (**kwargs)

    if kwargs and not argspec.keywords:
        unreachable()
    elif kwargs:
        args += (kwargs,)
    elif argspec.keywords:
        args += ({},)

    return args


if __name__ == '__main__':
    import doctest
    doctest.testmod()