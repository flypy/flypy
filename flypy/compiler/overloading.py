# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import inspect

from flypy.typing import resolve, to_blaze

from datashape import overloading
from datashape import coretypes as T
from datashape.overloading import lookup_previous
from datashape.overloading import overload, Dispatcher, flatargs as simple_flatargs
from datashape.util import gensym

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

class FlatArgs(tuple):
    """
    Represent flattened arguments according to numba's calling convention:

        - varargs are passed in tuples:
            def f(*args): ...
        - keyword arguments are passed in dicts:
            def f(**kwargs): ...

    Other arguments and defaults are passed directly.
    """

    def __new__(cls, py_func, values, **kwargs):
        return tuple.__new__(cls, values)

    def __init__(self, py_func, values):
        self.py_func = py_func
        self.argspec = inspect.getargspec(py_func)
        assert len(self) >= self.have_varargs + self.have_keywords

    @property
    def have_varargs(self):
        return bool(self.argspec.varargs)

    @property
    def have_keywords(self):
        return bool(self.argspec.keywords)

    @property
    def positional(self):
        if self.have_varargs or self.have_keywords:
            return self[:-self.have_varargs - self.have_keywords]
        return self

    @property
    def flat(self):
        assert not self.have_keywords
        return self.positional + (self.varargs or ())

    @property
    def varargs(self):
        if self.have_varargs and self.have_keywords:
            return self[-2]
        elif self.have_varargs:
            return self[-1]

    @property
    def keywords(self):
        if self.have_keywords:
            return self[-1]


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
    args = tuple(args)
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

    return FlatArgs(f, args)


freshvar = lambda: T.TypeVar(gensym())

def dummy_signature(f):
    """Create a dummy signature for `f`"""
    argspec = inspect.getargspec(f)
    n = len(argspec.args)

    argtypes = [freshvar() for i in range(n)]
    restype = freshvar()

    if argspec.varargs:
        argtypes.append(freshvar())
    if argspec.keywords:
        argtypes.append(freshvar())

    return T.Function(*argtypes + [restype])


def get_remaining_args(func, args):
    newargs = flatargs(func, args, {})
    return newargs[len(args):]

def fill_missing_argtypes(func, argtypes):
    """
    Fill missing argument types from default values.
    """
    from flypy import typeof

    argtypes = tuple(argtypes)
    remaining = get_remaining_args(func, argtypes)
    return argtypes + tuple(typeof(arg) for arg in remaining)

def get_varargs(flattened_args):
    flattened_args = list(flattened_args)

    if flattened_args and isinstance(flattened_args[-1], dict):
        flattened_args.pop()
    if flattened_args and isinstance(argtypes[-1], tuple):
        varargs = [make_tuple_type(argtypes.pop())]


if __name__ == '__main__':
    import doctest
    doctest.testmod()