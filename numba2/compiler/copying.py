# -*- coding: utf-8 -*-

"""
Copy functions to a function with a new name.
"""

from __future__ import print_function, division, absolute_import

from pykit.ir import copy_function, vmap, Function
from pykit.analysis import callgraph
from pykit.utils import make_temper

temper = make_temper()

def copy_graph(func, env, funcs=None, graph=None):
    """
    Copy a function graph.
    """
    if funcs is None:
        funcs = {}

    graph = graph or callgraph.callgraph(func)
    envs = env['numba.state.envs']

    copy_functions(graph, funcs, envs)
    update_callgraph(graph, funcs)

    return funcs

def copy_functions(graph, funcs, envs):
    """
    Copy a graph of functions.

    Arguments
    =========
    graph: networkx.DiGraph
        callgraph

    funcs: {Function: (Function, env) }
        Old functions mapping to new, copied, functions and environments.

    envs: { Function: env }
        All available environments
    """
    for func in graph.node:
        if func not in funcs:
            funcs[func] = copy(func, envs[func])


def update_callgraph(graph, funcs):
    """
    Update all function in the callgraph with the newly copied functions.
    """
    for old_src in graph.node:
        src, src_env = funcs[old_src]

        # Update all callsites
        for op in src.ops:
            if op.opcode == 'call':
                old_dst, args = op.args
                if isinstance(old_dst, Function):
                    new_dst, _ = funcs[old_dst]
                    op.set_args([new_dst, args])


def copy(func, env):
    new_func = copy_func(func, env)
    new_env = copy_env(func, new_func, env)
    env['numba.state.envs'][new_func] = new_env
    return new_func, new_env


def copy_func(func, env):
    new_func = copy_function(func)
    new_func.name = temper(new_func.name)
    return new_func


def copy_env(old_func, new_func, env):
    new_env = dict(env)
    new_env['numba.typing.context'] = copy_ir_valuemap(
        old_func, new_func, new_env['numba.typing.context'] or {})
    return new_env


def copy_ir_valuemap(old_func, new_func, valuemap):
    """
    Copy "value maps" mapping IR Operations to values. This should be used
    when a function is copied.

    Arguments
    =========
    oldops: [Op]
    newops: [Op]
    valuemap: {Op : object}
    """
    old = vmap(lambda x: x, old_func)
    new = vmap(lambda x: x, new_func)
    replacements = dict(zip(old, new))
    return dict((replacements.get(oldop, oldop), value)
                    for oldop, value in valuemap.iteritems())
