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

from flypy import promote, typeof, parse, typejoin
from flypy.errors import InferError, errctx
from flypy.types import Mono, Function, Pointer, bool_, void
from flypy.typing import resolve_simple, TypeVar, TypeConstructor
from flypy.functionwrapper import FunctionWrapper
from flypy.viz.prettyprint import debug_print
from flypy.errors import error_context
from .resolution import infer_call, Method, make_method, infer_getattr
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

def print_context(func, env, context):
    def _sort_key(op):
        if isinstance(op, ir.Op):
            return opindex[op]
        return op

    ops = list(func.ops)
    opindex = dict(zip(ops, range(len(ops))))

    print(("Type context %s:" % (env['flypy.state.func_name'],)).center(90))
    try:
        for op in sorted(context, key=_sort_key):
            ty = context[op]
            print("    %s: %s" % (op, ty))
    except ValueError:
        print("unable to print context :((")


def run(func, env):
    with errctx(env):
        cache = env['flypy.inference.cache']
        argtypes = env['flypy.typing.argtypes']
        ctx, signature = infer(cache, func, env, argtypes)

        env['flypy.typing.signature'] = signature
        env['flypy.typing.context'] = ctx.context
        env['flypy.typing.constraints'] = ctx.constraints

        if debug_print(func, env) and not env['flypy.state.opaque']:
            print_context(func, env, ctx.context)

        return ctx.func, env

def infer(cache, func, env, argtypes):
    """Infer types for the given function"""
    argtypes = tuple(argtypes)

    # -------------------------------------------------
    # Check cache

    cached = cache.lookup(func, argtypes)
    if cached:
        return cached

    # -------------------------------------------------
    # Infer

    if env["flypy.state.opaque"]:
        ctx = infer_opaque(func, env, argtypes)
    else:
        ctx = infer_function(cache, func, argtypes, env)

    # -------------------------------------------------
    # Cache result

    typeset = ctx.context['return']

    if env['flypy.state.generator']:
        from flypy.runtime.obj.generatorobject import Generator

        element_type = reduce(typejoin, ctx.context['generator'])
        restype = Generator[element_type, void]
        typeset.clear()
        typeset.add(restype)
    elif typeset:
        restype = reduce(typejoin, typeset)
    else:
        restype = void

    signature = Function[argtypes + (restype,)]
    cache.typings[func, argtypes] = ctx, signature
    return ctx, signature

def infer_opaque(func, env, argtypes):
    func = env["flypy.state.function_wrapper"]
    py_func = env["flypy.state.py_func"]
    restype = env["flypy.typing.restype"]
    func = opaque.implement(func, py_func, argtypes, env)
    ctx = Context(func, {'return': set([restype])}, {}, None, {})
    envs = env['flypy.state.envs']
    envs[func] = env
    return ctx

def infer_function(cache, func, argtypes, env):
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
    infer_graph(cache, ctx, env)
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
    context = { 'return': set(), 'generator': set(), void: void, bool_: bool_ }
    context['return'] = set()
    count = 0

    for op in func.ops:
        context[op] = set()
        if op.opcode == 'alloca':
            context['alloca%d' % count] = set()
            count += 1
        for arg in flatten(op.args):
            if (isinstance(arg, ir.Const) and
                    isinstance(arg.const, FunctionWrapper)):
                context[arg] = set([None])
            elif isinstance(arg, ir.Const):
                context[arg] = set([typeof(arg.const)])
            elif isinstance(arg, ir.Undef):
                context[arg] = set()
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
        self.generator   = 'generator'

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

    def op_unpack(self, op):
        print(op.args[0])
        self.G.add_edge(op.args[0], op)

    def op_setfield(self, op):
        pass # Handle this in the type checker

    def op_exc_setup(self, op):
        pass

    def op_exc_throw(self, op):
        pass

    def op_exc_catch(self, op):
        pass

    def op_jump(self, op):
        pass

    def op_cbranch(self, op):
        """
        Γ ⊢ cbranch (x : bool)
        """
        self.G.add_edge(bool_, op.args[0])

    def op_yield(self, op):
        """
        Γ ⊢      x : α
        --------------------
        Γ ⊢ (yield x) : void
        """
        self.G.add_edge(void, op)
        self.G.add_edge(op.args[0] or void, self.generator)

    def op_ret(self, op):
        """
        Γ ⊢ f : (α -> β)
        ----------------
        Γ ⊢ return x : β
        """
        self.G.add_edge(op.args[0] or void, self.return_node)


#===------------------------------------------------------------------===
# Inference on Graph
#===------------------------------------------------------------------===

def infer_graph(cache, ctx, env):
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
    env : dict
        compiler environment

    Constaints include:

        'pointer': the Pointer type constructor
        'flow'   : represents type flow-in
        'attr'   : attribute access
        'call'   : call of a dynamic or static function
    """
    W = collections.deque(ctx.graph) # worklist
    # pprint(ctx.graph.edge)
    # view(ctx.graph)

    processed = set()

    while W:
        node = W.popleft()
        changed = infer_node(cache, ctx, node, env, processed)
        if changed:
            for neighbor in ctx.graph.neighbors(node):
                W.appendleft(neighbor)

def infer_node(cache, ctx, node, env, processed):
    """Infer types for a single node"""
    changed = False
    C = ctx.constraints.get(node, 'flow')
    if isinstance(node, Mono):
        typeset = set([node])
    else:
        typeset = ctx.context[node]

    incoming = ctx.graph.predecessors(node)
    outgoing = ctx.graph.neighbors(node)

    # Get line number of original function
    if isinstance(node, ir.Op):
        lineno = node.metadata.get('lineno', -1)
    else:
        lineno = -1

    with error_context(lineno=lineno, during="Infer call"):
        inferer = inference_table[C]
        return inferer(ctx, env, node, processed, incoming, typeset, changed)

### *** type inference rules *** ###

def infer_pointer(ctx, env, node, processed, incoming, typeset, changed):
    """
    Infer pointer creation:

        alloca var
    """
    for neighbor in incoming:
        for type in ctx.context[neighbor]:
            #result = Pointer[type]
            result = type
            changed |= result not in typeset
            typeset.add(result)

    return changed

def infer_flow(ctx, env, node, processed, incoming, typeset, changed):
    """
    Infer data flow:

        a = b
    """
    for neighbor in incoming:
        for type in ctx.context[neighbor]:
            changed |= type not in typeset
            typeset.add(type)

    return changed

def infer_attr(ctx, env, node, processed, incoming, typeset, changed):
    """
    Infer attributes:

        obj.a
    """
    [neighbor] = incoming
    attr = ctx.metadata[node]['attr']
    for type in ctx.context[neighbor]:
        if attr in type.fields:
            result = make_method(type, attr)
        elif attr in type.layout:
            result = type.resolved_layout[attr]
        elif '__getattr__' in type.fields:
            _, _, result = infer_getattr(type, env)
        elif '__getattribute__' in type.fields:
            raise InferError("__getattribute__ note supported")
        else:
            raise InferError("Type %s has no attribute %s" % (type, attr))

        changed |= result not in typeset
        typeset.add(result)

    return changed

def infer_appl(ctx, env, node, processed, incoming, typeset, changed):
    """
    Infer function application:

        f(a)
    """
    func = ctx.metadata[node]['func']
    func_types = ctx.context[func]
    arg_typess = [ctx.context[arg] for arg in ctx.metadata[node]['args']]

    # Iterate over cartesian product, processing only unpreviously
    # processed combinations
    for func_type in set(func_types):
        for arg_types in product(*arg_typess):
            key = (node, func_type, tuple(arg_types))
            if key not in processed:
                processed.add(key)
                _, signature, result = infer_call(func, func_type,
                                                  arg_types, env)
                if isinstance(result, TypeVar):
                    raise TypeError("Expected a concrete type result, "
                                    "not a type variable! (%s)" % (func,))
                changed |= result not in typeset
                typeset.add(result)
                if None in func_types:
                    func_types.remove(None)
                    func_types.add(signature)

    return changed


inference_table = {
    'pointer': infer_pointer,
    'flow': infer_flow,
    'attr': infer_attr,
    'call': infer_appl,
}