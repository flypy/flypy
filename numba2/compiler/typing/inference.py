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
from pprint import pprint
import collections
from itertools import product

from numba2.typing import promote, typeof, parse
from numba2.errors import InferError
from numba2.types import Type, Function, Pointer, bool_, void
from numba2.functionwrapper import FunctionWrapper
from .. import opaque

import pykit.types
from pykit import ir
from pykit.utils import flatten

import networkx

#===------------------------------------------------------------------===
# Utils
#===------------------------------------------------------------------===

def copy_context(context):
    return dict((binding, set(type)) for binding, type in context.iteritems())

def view(G):
    import matplotlib.pyplot as plt
    networkx.draw(G)
    plt.show()

Method = type(parse("Method[func, self]"))

#===------------------------------------------------------------------===
# Inference structures
#===------------------------------------------------------------------===

class Context(object):
    """
    Type inference context.

    Attributes
    ==========

        context: { Operation : set([Type]) }
            bindings, mapping Operations (variables) to types (sets)

        graph: networkx.DiGraph
            constraint graph

        constraints: { Node : str }
            constraints for the nodes in the constraint graph

        metadata: { Node : dict }
            extra metadata for graph nodes in the constraint graph

    Only 'context' is mutable after construction!
    """

    def __init__(self, func, context, constraints, graph, metadata):
        self.func = func
        self.context = context
        self.constraints = constraints
        self.graph = graph
        self.metadata = metadata

    def copy(self):
        return Context(self.func, copy_context(self.context),
                       self.constraints, self.graph, self.metadata)

#===------------------------------------------------------------------===
# Inference
#===------------------------------------------------------------------===

def run(func, env):
    cache = env['numba.typing.cache']
    argtypes = env['numba.typing.argtypes']
    ctx, signature = infer(cache, func, argtypes)

    env['numba.typing.signature'] = signature
    env['numba.typing.context'] = ctx.context
    env['numba.typing.constraints'] = ctx.constraints

def infer(cache, func, argtypes):
    """Infer types for the given function"""
    argtypes = tuple(argtypes)

    # -------------------------------------------------
    # Check cache

    cached = cache.lookup(func, argtypes)
    if cached:
        return cached

    ctx = infer_function(cache, func, argtypes)

    # -------------------------------------------------
    # Cache result

    typeset = ctx.context['return']
    restype = reduce(promote, typeset)

    signature = Function[argtypes + (restype,)]
    cache.typings[func, argtypes] = ctx, signature
    return ctx, signature


def infer_function(cache, func, argtypes):
    if isinstance(func, FunctionWrapper) and func.opaque:
        restype = func.signature.restype # TODO: unify
        func = opaque.implement(func, argtypes)
        ctx = Context(func, {'return': set([restype])}, {}, None, {})
        return ctx

    # -------------------------------------------------
    # Build template

    ctx = cache.lookup_ctx(func)
    if ctx is None:
        ctx = build_graph(func)
        cache.ctxs[func] = ctx

    ctx = ctx.copy()

    # -------------------------------------------------
    # Infer typing context

    seed_context(ctx, argtypes)
    infer_graph(cache, ctx)
    return ctx

# ______________________________________________________________________

def build_graph(func):
    """
    Build a constraint network and initial context. This is a generic
    templates share-able between input types.
    """
    G = networkx.DiGraph()
    context = initial_context(func)

    for op in func.ops:
        G.add_node(op)
        for arg in flatten(op.args):
            if isinstance(arg, (ir.Const, ir.GlobalValue, ir.FuncArg)):
                G.add_node(arg)

    constraints, metadata = generate_constraints(func, G)
    return Context(func, context, constraints, G, metadata)

def initial_context(func):
    """Initialize context with argtypes"""
    context = { 'return': set() }
    context['return'] = set()
    count = 0

    for op in func.ops:
        context[op] = set()
        if op.opcode == 'alloca':
            context['alloca%d' % count] = set()
            count += 1
        for arg in flatten(op.args):
            if isinstance(arg, ir.Const):
                context[arg] = typeof(arg.const)
            elif isinstance(arg, ir.GlobalValue):
                raise NotImplementedError("Globals")

    return context

def seed_context(ctx, argtypes):
    for arg, argtype in zip(ctx.func.args, argtypes):
        ctx.context[arg] = set([argtype])

# ______________________________________________________________________

def generate_constraints(func, G):
    gen = ConstraintGenerator(func, G)
    ir.visit(gen, func, errmissing=True)
    return gen.constraints, gen.metadata


class ConstraintGenerator(object):
    """
    Generate constraints for untyped pykit IR.
    """

    def __init__(self, func, G):
        self.func = func
        self.G = G
        self.constraints = {}  # Op -> constraint
        self.metadata = {}     # Node -> object
        self.allocas = {}     # Op -> node
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
        if op not in self.allocas:
            node = 'alloca%d' % len(self.allocas)
            self.G.add_node(node)
            self.allocas[op] = node

        self.G.add_edge(self.allocas[op], op)
        self.constraints[op] = 'pointer'

    def op_load(self, op):
        """
        Γ ⊢ x : α *
        --------------
        Γ ⊢ load x : α
        """
        self.G.add_edge(self.allocas[op.args[0]], op)

    def op_store(self, op):
        """
        Γ ⊢ var : α *      Γ ⊢ x : α
        -----------------------------
        Γ ⊢ store x var : void
        """
        value, var = op.args
        self.G.add_edge(value, self.allocas[var])

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
        arg, attr = op.args

        self.G.add_edge(arg, op)
        self.metadata[op] = { 'attr': attr }
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

        func, args = op.args
        self.metadata[op] = { 'func': func, 'args': args}

    def op_convert(self, op):
        """
        Γ ⊢ x : α     β ≠ Opaque
        ------------------------
          Γ ⊢ convert(x, β) : β

        Γ ⊢ x : α   convert(x, Opaque) : β
        ----------------------------------
                    Γ ⊢ α = β
        """
        if op.type != pykit.types.Opaque:
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
        Γ ⊢ cbranch (x : bool)
        """
        self.G.add_edge(bool_, op.args[0])

    def op_ret(self, op):
        """
        Γ ⊢ f : (α -> β)
        ----------------
        Γ ⊢ return x : β
        """
        self.G.add_edge(op.args[0] or void, self.return_node)

# ______________________________________________________________________

def infer_graph(cache, ctx):
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
    W = collections.deque(ctx.graph) # worklist
    # pprint(ctx.graph.edge)
    # view(ctx.graph)

    while W:
        node = W.popleft()
        changed = infer_node(cache, ctx, node)
        if changed:
            for neighbor in ctx.graph.neighbors(node):
                W.appendleft(neighbor)

def infer_node(cache, ctx, node):
    """Infer types for a single node"""
    changed = False
    C = ctx.constraints.get(node, 'flow')
    if isinstance(node, Type):
        typeset = set([node])
    else:
        typeset = ctx.context[node]

    incoming = ctx.graph.predecessors(node)
    outgoing = ctx.graph.neighbors(node)

    processed = set()

    if C == 'pointer':
        for neighbor in incoming:
            for type in ctx.context[neighbor]:
                result = Pointer[type]
                changed |= result not in typeset
                typeset.add(result)

    elif C == 'flow':
        for neighbor in incoming:
            for type in ctx.context[neighbor]:
                changed |= type not in typeset
                typeset.add(type)

    elif C == 'attr':
        [neighbor] = incoming
        attr = ctx.metadata[node]['attr']
        for type in ctx.context[neighbor]:
            if attr not in type.fields:
                raise InferError("Type %s has no attribute %s" % (type, attr))
            value = type.fields[attr]
            func, self = value, type
            result = Method(func, self)
            changed |= result not in typeset
            typeset.add(result)

    else:
        assert C == 'call'
        func = ctx.metadata[node]['func']
        func_types = ctx.context[func]
        arg_typess = [ctx.context[arg] for arg in ctx.metadata[node]['args']]

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
    if type(func_type) == Method:
        func = func_type.parameters[0]
        self = func_type.parameters[1]
        #[[self_type] + arg_types for self_type in self]
        # arg_types = (self,) + arg_types

    elif not isinstance(func, ir.Function):
        # Higher-order function
        restype = func_type.restype
        assert restype
        return  restype

    if isinstance(func_type, frozenset):
        # Overloaded function or method
        raise NotImplemented("Overloaded functions")
        # func = find_overload(func_type, arg_types)

    ctx, signature = infer(cache, func, arg_types)
    return signature.restype