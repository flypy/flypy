# -*- coding: utf-8 -*-

"""
Cartesian Product Algorithm. We consider types as sets of classes (Sum) and
infer using the cartesian product of the argument sets/types.

Note that promotion is handled through overloading, e.g.:

    @overload('α -> β -> γ')
    def __add__(self, other):
        return other.__radd__(self)

    @overload('α -> α -> β')
    def __add__(self, other):
        result = ... # Perform addition
        return result

    @overload('α -> β -> γ')
    def __radd__(self, other):
        type = promote(typeof(self), typeof(other))
        return convert(other, type) + convert(self, type)

These are implemented in a trait which can be implemented by a user-defined
type (like Int). If there is no more specific overload, __radd__ will do the
promotion, triggering either an error or a call to a real implementation.
"""

from __future__ import print_function, division, absolute_import
import collections
from functools import partial
from itertools import product

from .typing import Type, Function, Opaque, Void, Bool, Pointer, Method
from ..errors import InferError

import pykit.types
from pykit import ir
from pykit.utils import flatten

import networkx

# ______________________________________________________________________
# Utils

def make_type(type):
    if isinstance(type, Type):
        return set([Type])
    elif isinstance(type, pykit.types.Type):
        return set([typemap(type)])
    else:
        assert isinstance(type, set)
        return type

def copy_context(context):
    return dict((binding, set(type)) for binding, type in context.iteritems())

def typemap(ty):
    if isinstance(ty, pykit.types.Type):
        return Type(type(ty).__name__, *map(typemap, ty))
    return ty

# ______________________________________________________________________

class InferenceCache(object):
    def __init__(self):
        self.typings = {} # (func, argtypes) -> { (signature, context, constraints) }
        self.templates = {}  # func -> { template }

    def lookup(self, func, argtypes):
        return self.typings.get((func, tuple(argtypes)))


def infer(cache, func, argtypes):
    """Infer types for the given function"""
    # Check cache
    signature, context = cache.lookup(func, argtypes)
    if signature:
        return signature

    # Build template
    if func in cache.templates:
        cache.templates[func] = build_graph(func)

    G, context, constraints = cache.templates[func]

    # Infer typing context
    context = copy_context(context)
    infer_graph(cache, G, context, constraints)

    # Cache result
    signature = Function(context['return'], *argtypes)
    cache.typings[func, argtypes] = (signature, context, constraints)
    return signature, context, constraints

def run(func, env):
    cache = env['numba.typing.cache']
    argtypes = env['numba.typing.argtypes']
    signature, context, constraints = infer(cache, func, argtypes)

    env['numba.typing.signature'] = signature
    env['numba.typing.context'] = context
    env['numba.typing.constraints'] = constraints

# ______________________________________________________________________

def build_graph(func):
    G = networkx.DiGraph()
    context = {}

    context['return'] = set()
    for arg in func.args:
        G.add_node(arg)

    for op in func.ops:
        G.add_node(op)
        if op.type != Opaque():
            context[op] = make_type(op.type)
        for arg in op.args:
            if isinstance(arg, (ir.Const, ir.GlobalValue)):
                G.add_node(arg)
                context[arg] = make_type(arg.type)

    constraints = generate_constraints(G)
    return G, context, constraints

# ______________________________________________________________________

def generate_constraints(func, G):
    gen = ConstraintGenerator(func, G)
    ir.visit(gen, func, errmissing=True)
    return gen.constraints


class ConstraintGenerator(object):
    """
    Generate constraints for untyped pykit IR.
    """

    def __init__(self, func, G):
        self.func = func
        self.G = G
        self.constraints = {} # Op -> constraint
        self._allocas = {}     # Op -> node
        self.return_node = 'return'

    def op_alloca(self, op):
        """
        Γ ⊢ a : α *
        ------------------
        Γ ⊢ alloca a : α *

        Γ ⊢ a : Opaque
        ----------------
        Γ ⊢ alloca a : ⊥
        """
        if op not in self._allocas:
            node = 'var%d' % len(self._allocas)
            self.G.add_node(node)
            self._allocas[op] = node

        self.G.add_edge(self._allocas[op], op, )
        self.constraints[op] = 'pointer'

    def op_load(self, op):
        """
        Γ ⊢ x : α *
        --------------
        Γ ⊢ load x : α
        """
        self.G.add_edge(self._allocas[op.args[0]], op)

    def op_store(self, op):
        """
        Γ ⊢ var : α *      Γ ⊢ x : α
        -----------------------------
        Γ ⊢ store x var : Void
        """
        value, var = op.args
        self.G.add_edge(value, self._allocas[var])

    def op_phi(self, op):
        """
        Γ ⊢ l : α       Γ ⊢ r : β
        -------------------------
           Γ ⊢ φ(l, r) : α + β
        """
        for incoming in op.args[1]:
            self.G.add_edge(incoming, op)

    def op_getfield(self, op):
        """
        Γ ⊢ x : { a : α }
        -----------------
           Γ ⊢ x.a : α
        """
        self.G.add_edge(op.args[0], op)
        self.constraints[op] = 'attr'

    def op_call(self, op):
        """
        Γ ⊢ f : (α -> β)        Γ ⊢ x : α
        ----------------------------------
                   Γ ⊢ f(a) : β
        """
        for arg in flatten(op.args):
            self.G.add_edge(arg, op)
        self.constraints[op] = 'call'

    def op_convert(self, op):
        """
        Γ ⊢ x : α     β ≠ Opaque
        ------------------------
          Γ ⊢ convert(x, β) : β

        Γ ⊢ x : α   convert(x, Opaque) : β
        ----------------------------------
                    Γ ⊢ α = β
        """
        if op.type != Opaque():
            self.G.add_edge(op.type, op)
        else:
            self.G.add_edge(op.args[0], op)

    def exc_setup(self, op):
        pass

    def exc_throw(self, op):
        pass

    def exc_catch(self, op):
        pass

    def op_jump(self, op):
        pass

    def op_cbranch(self, op):
        """
        Γ ⊢ cbranch (x : Bool)
        """
        self.G.add_edge(Bool(), op.args[0])

    def op_ret(self, op):
        """
        Γ ⊢ f : (α -> β)
        ----------------
        Γ ⊢ return x : β
        """
        self.G.add_node(op.args[0] or Void(), self.return_node)

# ______________________________________________________________________

def infer_graph(cache, G, context, constraints):
    """
    infer_graph(G, context, constraints)

    Type inference on a constraint graph.

    Parameters
    ----------
    G : graph
        networkx directed graph of type flow
    context : dict
        Γ mapping Ops to type sets
    constraints: dict
        maps nodes (Ops) from the graph to the constraints the node represents

    Constaints include:

        'pointer': the Pointer type constructor
        'flow'   : represents type flow-in
        'attr'   : attribute access
        'call'   : call of a dynamic or static function
    """
    W = collections.deque(G) # worklist
    while W:
        node = W.popleft()
        changed = infer_node(cache, G, context, constraints, node)
        if changed:
            for neighbor in G.neighbors(node):
                W.appendleft(neighbor)

def infer_node(cache, G, context, constraints, node):
    """Infer types for a single node"""
    changed = False
    C = constraints[node]
    typeset = context[node]
    nb = G.neighbors(node)

    processed = set()

    if C == 'pointer':
        for neighbor in nb:
            for type in context[neighbor]:
                result = Pointer(type)
                changed |= result not in typeset
                typeset.add(result)

    elif C == 'flow':
        for neighbor in nb:
            for type in context[neighbor]:
                changed |= type not in typeset
                typeset.add(type)

    elif C == 'attr':
        [neighbor] = nb
        attr = node['attr']
        for type in context[neighbor]:
            if attr not in type.fields:
                raise InferError("Type %s has no attribute %s" % (type, attr))
            value, result = type.fields[attr]
            if result.name == 'Function':
                func, self = value, type
                result = Method(func, self)
            changed |= result not in typeset
            typeset.add(result)

    else:
        assert C == 'call'
        func = node['func']
        func_types = context[func]
        arg_typess = [context[arg] for arg in node['args']]

        # Iterate over cartesian product, processing only unpreviously
        # processed combinations
        for func_type in func_types:
            for arg_types in product(*arg_typess):
                key = (node, func_type, tuple(arg_types))
                if key not in processed:
                    processed.add(key)
                    result = infer_call(cache, func, func_type, arg_types)
                    changed |= result not in typeset
                    typeset.add(result)

    return changed


def infer_call(cache, func, func_type, arg_types):
    """
    Infer a single call. We have three cases:

        1) Static receiver function
        2) Higher-order function
            This is already typed
        3) Method. We need to insert 'self' in the cartesian product
    """
    if func_type.name == 'Method':
        func = func_type[0]
        self = func_type[1]
        arg_types = [[self_type] + arg_types for self_type in self]

    elif not isinstance(func, ir.Function):
        # Higher-order function
        restype = func_type.params[0]
        assert restype
        return  restype

    if isinstance(func_type, frozenset):
        # Overloaded function or method
        raise NotImplemented("Overloaded functions")
        # func = find_overload(func_type, arg_types)

    signature = infer(cache, func, arg_types)
    return signature.params[0] # Return return type