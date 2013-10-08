# -*- coding: utf-8 -*-

"""
Handle phase transitions by copying the graph so functions can be reused later.
"""

from __future__ import print_function, division, absolute_import
from collections import namedtuple

from . import copying

from pykit import ir
from pykit.analysis import callgraph

Transition = namedtuple('Transition', ['old_func', 'new_func', 'new_env'])

def single_copy(func, env):
    if not isinstance(func, ir.Function):
        return func, env
    return copying.copy(func, env)


def transition_copy_graph(func, env):
    """
    Transition to a new phase:

        - create a copy of all functions in the call graph
        - update and cache copied function results in the 'numba.state.copies'
          dict
    """
    if not isinstance(func, ir.Function):
        return func, env

    graph = callgraph.callgraph(func)
    envs = env['numba.state.envs']
    phase = env['numba.state.phase']

    func_copies = env['numba.state.copies']
    if phase in func_copies:
        _, new_func, new_env = func_copies[phase]
        return new_func, new_env

    all_copies = find_copies(graph, envs, phase)
    copying.copy_graph(func, env, all_copies, graph)
    update_copies(all_copies, phase)

    _, new_func, new_env = func_copies[phase]
    return new_func, new_env


def find_copies(graph, envs, phase):
    new = {} # { old Function : (new Function (copy), new env) }

    for f in graph.node:
        e = envs[f]
        f_copies = e['numba.state.copies']
        if phase in f_copies:
            transition = f_copies[phase]
            new[f] = (transition.new_func, transition.new_env)
        else:
            # Assert that `f` is in `phase`
            pass

    return new


def update_copies(all_copies, phase):
    for old_func, (new_func, new_env) in all_copies.iteritems():
        f_copies = new_env['numba.state.copies']
        f_copies[phase] = Transition(old_func, new_func, new_env)
