# -*- coding: utf-8 -*-

"""
Handle a restricted form of generators and inlining.
"""

from __future__ import print_function, division, absolute_import

from collections import namedtuple
from functools import partial

from numba2.errors import error
from numba2.runtime import builtins
from numba2.compiler.optimizations import inlining, reg2mem

from pykit.ir import Builder, copying
from pykit.analysis import loop_detection, callgraph
from pykit.utils import listify

#===------------------------------------------------------------------===
# Driver
#===------------------------------------------------------------------===

def generator_fusion(func, env):
    changed = True
    envs = env['numba.state.envs']
    dependences = callgraph.callgraph(func).node
    while changed:
        changed = False
        for f in dependences:
            e = envs[f]
            consumers = identify_consumers(f, e)
            #print("consumers", f.name, consumers)
            fuse_generators(f, e, consumers)
            changed |= bool(consumers)

#===------------------------------------------------------------------===
# Consumer Identification
#===------------------------------------------------------------------===

Consumer = namedtuple('Consumer', ['generator', 'iter', 'next', 'loop'])

@listify
def identify_consumers(func, env):
    """
    Identify consumers of generators, that is find the loops that iterate
    over a generator.
    """
    generator_objects = find_generators(func, env)
    #print("generators", generator_objects)
    if not generator_objects:
        # We can stop now
        return

    loop_forest = loop_detection.find_natural_loops(func)
    loops = loop_detection.flatloops(loop_forest)
    heads = dict((loop.head, loop) for loop in loops)

    expect_call = partial(expect_single_call, func, env)
    #print("loops", loops, "heads", heads)

    for generator_obj in generator_objects:
        # Check for a nesting of next(iter(my_generator()))
        iter = expect_call(generator_obj, builtins.iter)
        next = expect_call(iter, builtins.next)
        if iter and next and next.block in heads:
            loop = heads[next.block]
            yield Consumer(generator_obj, iter, next, loop)

@listify
def find_generators(func, env):
    """
    Find all calls to generators.
    """
    envs = env['numba.state.envs']

    for op in func.ops:
        if op.opcode == 'call':
            f, args = op.args
            if f in envs and envs[f]['numba.state.generator'] == 1:
                yield op

# -- helpers -- #

def expect_single_call(func, env, defining_op, numba_func):
    if not defining_op or defining_op.opcode != 'call':
        return

    uses = func.uses[defining_op]
    if len(uses) == 1:
        [op] = uses
        if op.opcode == 'call':
            f, args = op.args
            envs = env['numba.state.envs']
            e = envs[f]
            if e['numba.state.function_wrapper'] == numba_func:
                return op

#===------------------------------------------------------------------===
# Generator Fusion
#===------------------------------------------------------------------===

def fuse_generators(func, env, consumers):
    """
    Rewrite straightforward uses of generators, i.e. where a generator is
    allocated and consumed by a single consumer loop.
    """
    envs = env['numba.state.envs']

    for consumer in consumers:
        generator_func = consumer.generator.args[0]

        empty_body = detach_loop(func, consumer)
        move_generator(func, consumer, empty_body)
        clean_loop_body(func, consumer)

        valuemap = inlining.inline_callee(func, consumer.generator,
                                          env, envs[generator_func])
        consume_yields(func, consumer, generator_func, valuemap)


def rewrite_general_generators(func, env):
    """
    Rewrite general use of generators.
    """
    generator_objects = find_generators(func, env)
    for gen in generator_objects:
        error(env, 'lower', "Generator %s could not be fused" % (gen,))


#=== rewrites ===#

def detach_loop(func, consumer):
    loop, iter = consumer.loop, consumer.iter

    for block in loop.blocks:
        func.del_block(block)

    func.reset_uses()

    b = Builder(func)
    jump = iter.block.terminator
    assert jump.opcode == 'jump' and jump.args[0] == loop.head
    jump.delete()

    b.position_at_end(iter.block)
    _, newblock = b.splitblock(terminate=True)
    return newblock

def move_generator(func, consumer, empty_body):
    gen = consumer.generator
    gen.unlink()

    b = Builder(func)
    b.position_at_end(empty_body)
    b.emit(gen)

    with b.at_end(empty_body):
        loop_exit = determine_loop_exit(consumer.loop)
        b.jump(loop_exit)

def determine_loop_exit(loop):
    [exc_setup] = [op for op in loop.head if op.opcode == 'exc_setup']
    [loop_exit] = exc_setup.args[0]
    return loop_exit

def clean_loop_body(func, consumer):
    loop = consumer.loop

    # Patch back-edge to jump to loop exit instead
    assert loop.tail.terminator.opcode == 'jump', "expected a back-edge"
    assert loop.tail.terminator.args[0] == loop.head, "expected a back-edge"

    consumer.iter.delete()
    loop.tail.terminator.delete()

def consume_yields(func, consumer, generator_func, valuemap):
    b = Builder(func)
    copier = lambda x : x

    loop = consumer.loop
    inlined_values = set(valuemap.values())

    for block in func.blocks:
        if block in inlined_values:
            for op in block.ops:
                if op.opcode == 'yield':
                    # -- Replace 'yield' by the loop body -- #
                    b.position_after(op)
                    _, resume = b.splitblock()

                    # Copy blocks
                    blocks = [copier(block) for block in loop.blocks]

                    # Insert blocks
                    prev = op.block
                    for block in blocks:
                        func.add_block(block, after=prev)
                        prev = block

                    # Fix wiring
                    b.jump(blocks[0])
                    b.position_at_end(blocks[-1])
                    b.jump(resume)

                    # We just introduced a bunch of copied blocks
                    func.reset_uses()

                    # Update phis with new predecessor
                    b.replace_predecessor(loop.tail, op.block, loop.head)
                    b.replace_predecessor(loop.tail, op.block, loop.head)

                    # Replace next() by value produced by yield
                    value = op.args[0]
                    consumer.next.replace_uses(value)
                    op.delete()

    # We don't need these anymore
    consumer.next.delete()