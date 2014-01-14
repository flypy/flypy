# -*- coding: utf-8 -*-

"""
Blaze term tree evaluation.

This module defines an intermediate representation that encodes expressions
with are calls (`Apply` terms). Any application can have any number of
arguments, which are sub-expressions:

    - an other `Apply` term
    - a dynamic argument

Dynamic arguments are supplied at runtime, and are indices into a tuple
supplied at runtime. The idea is to compile a query, given is supplied as
a term, which is to be applied potentially many times.

We write an evaluator `eval` that takes a `term` and a tuple of `inputs`.
The term can be created offline, and the `inputs` can be created later.
The function can then be compiled up-front, by compiling the `evaluator` with
the type of `term` and `inputs`.

The resulting function can now be applied efficiently.
"""

from __future__ import print_function, division, absolute_import

from flypy import jit, sjit, typeof
from flypy.conversion import fromobject
from flypy.lib.nplib import NDArray
from flypy.runtime.obj.tupleobject import head, tail, StaticTuple, EmptyTuple

import numpy as np

# TODO: Loop unrolling

#===------------------------------------------------------------------===
# Blaze Terms (Intermediate Representation)
#===------------------------------------------------------------------===

_cache = {} # (f, subterms) -> Appl

def make_applier(f, *subterms):
    """Create a blaze function application term"""

    @jit('Apply[subterms]')
    class Apply(object):
        """Function application term"""

        layout = [('subterms', 'subterms')] # Tuple of sub-expressions

        @jit
        def apply(self, args):
            args = eval_subterms(self.subterms, args)
            return f(*args)

        def __repr__(self):
            return "Apply(%s, %s)" % (f.py_func.__name__, self.subterms)

    # NOTE: flypy doesn't reconstruct flypy objects recursively, it only
    #       acts on roots!
    subterms = fromobject(subterms, typeof(subterms))
    return Apply(subterms)

def constant(n):
    """A Term representing a constant"""
    @jit
    class Constant(object):
        layout = []

        @jit
        def apply(self, args):
            return n

        def __repr__(self):
            return "Const(%s)" % (n,)

    return Constant

def argmarker():
    """Create an stream of argument indices into the args tuple"""
    item = 0
    while True:
        yield Arg(item)
        item = Succ(item)

# -- term nodes (internal) -- #

@jit('Arg[argno]')
class Arg(object):
    """Argument to a term"""

    layout = [('argno', 'argno')]

    @jit
    def apply(self, args):
        return lookup(args, self.argno)

    def __repr__(self):
        return "Arg(%s)" % (self.argno,)


@jit('Succ[pred]')
class Succ(object):
    """Term to represent an static index into the arguments tuple"""
    layout = [('pred', 'pred')]

    def __repr__(self):
        return "Succ(%s)" % (self.pred,)


#===------------------------------------------------------------------===
# Evaluator
#===------------------------------------------------------------------===

@jit('StaticTuple[a, b] -> args -> r')
def eval_subterms(subterms, inputs):
    """Evaluate the given `subterms` (code) with the given `inputs` (data)"""
    hd = eval(head(subterms), inputs)
    tl = eval_subterms(tail(subterms), inputs)
    return StaticTuple(hd, tl)

@jit('EmptyTuple[] -> args -> r')
def eval_subterms(subterms, inputs):
    return subterms

@jit('StaticTuple[a, b] -> Succ[pred] -> r')
def lookup(args, n):
    """Look up the nth argument"""
    return lookup(tail(args), n.pred)

@jit('StaticTuple[a, b] -> zero -> r')
def lookup(args, n):
    return head(args)

@jit
def eval(term, inputs):
    """Evaluate a given term with the argument tuple"""
    return term.apply(inputs)

#===------------------------------------------------------------------===
# Scalar Test
#===------------------------------------------------------------------===

@jit('a : integral -> a')
def usub(x):
    return -x

@jit('a -> a -> a')
def add(a, b):
    return a + b

@jit('a -> a -> a')
def mul(a, b):
    return a * b

makearg = argmarker()

a = next(makearg)
b = next(makearg)
x = make_applier(add, a, b) # a + b
y = make_applier(usub, x)
term = make_applier(mul, x, y)

print(term)
print(eval(term, (2, 3)))
print(eval(term, (7, 3)))

#===------------------------------------------------------------------===
# Example flypy loop nest wrapper
#===------------------------------------------------------------------===

@jit('Loop[subterm]')
class Loop(object):
    layout = [('subterm', 'subterm')]

    @jit
    def apply(self, inputs):
        # This assumes `inputs` are already broadcast
        array = head(inputs)
        extent = array.dims.extent

        for i in range(extent):
            eval(self.subterm, index(inputs, i))

        return head(inputs)

    def __repr__(self):
        return "Loop(%s)" % (self.subterms,)


@jit('InnerLoop[subterm]')
class InnerLoop(object):
    layout = [('subterm', 'subterm')]

    @jit
    def apply(self, inputs):
        # This assumes `inputs` are already broadcast
        out = head(inputs)
        extent = out.dims.extent
        inputs = tail(inputs)

        for i in range(extent):
            out[i] = eval(self.subterm, index(inputs, i))

        return out

    def __repr__(self):
        return "InnerLoop(%s)" % (self.subterms,)


@jit('StaticTuple[NDArray[dtype, dims], tl] -> int64 -> r')
def index(inputs, i):
    hd = head(inputs)
    tl = tail(inputs)
    return StaticTuple(hd[i], index(tl, i))

#@jit('t -> int64 -> r')
#def index(inputs, i):
#    hd = head(inputs)
#    tl = tail(inputs)
#    return StaticTuple(hd, index(tl, i))

@jit('EmptyTuple[] -> int64 -> r')
def index(inputs, i):
    return inputs


# 1-dimensional example

makearg = argmarker()

a = next(makearg)
b = next(makearg)
x = make_applier(add, a, b)
inner = InnerLoop(x)
outer = Loop(inner)

def test_1d():
    A = np.arange(25) #.reshape(5, 5)
    out = np.empty(25) #(5, 5))
    args = (out, A, A)
    print(eval(inner, args))

# 2-dimensional example

def test_2d():
    A = np.arange(25).reshape(5, 5)
    out = np.empty(25).reshape((5, 5))
    args = (out, A, A)
    print(eval(outer, args))

test_1d()
test_2d()

#===------------------------------------------------------------------===
# Example with CKernels
#===------------------------------------------------------------------===

@sjit('CkernelData[term, inputs]')
class CkernelData(object):
    layout = [('term', 'term'), ('inputs', 'inputs')]

@jit('Pointer[CkernelData] -> r')
def ckernel_element_wrapper(p):
    data = p[0]
    out = head(data.inputs)
    inputs = tail(data.inputs)
    out[0] = eval(data.term, inputs)