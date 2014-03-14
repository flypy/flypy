# -*- coding: utf-8 -*-

"""
Handle a restricted form of generators and inlining.
"""

from __future__ import print_function, division, absolute_import

from collections import namedtuple
from functools import partial

from flypy.types import struct_, void, int64
from flypy.errors import error
from flypy.runtime import builtins
from flypy.compiler.optimizations import inlining, reg2mem

from pykit.ir import Op, OpBuilder, Builder, FuncArg, copying
from pykit.analysis import loop_detection, callgraph
from pykit.utils import listify, flatten

#===------------------------------------------------------------------===
# Driver
#===------------------------------------------------------------------===

def lower_generators(func, env):
    """
    Rewrite generators.
    """
    generator_objects = find_generators(func, env)
    for gen in generator_objects:
        error(env, 'lower', "Generator %s could not be fused" % (gen,))


def lower_producer(func, env):
    """
    Rewrite generator function (producer):

        def f(x):
            x = x + 1

            yield x * 2     # stat1
            yield x + 1     # stat2
            yield x / 2     # stat3

    -->

        def f(gen, x):
            if gen.state == 0:
                jump(label0)
            elif gen.state == 1:
                jump(stat1)
            else:
                jump(stat2)

            # label0

            ...
    """
    restype = env['flypy.typing.restype']
    compute_generator_state(func, env)
    vars = collect_generator_vars(func, env)
    state_type = compute_generator_state(func, env, vars)
    rewrite_state_variables(func, env, vars)
    yieldpoints = rewrite_yieldpoints(func, env)
    insert_dispatcher(func, env, yieldpoints)


#===------------------------------------------------------------------===
# Generator state
#===------------------------------------------------------------------===

def collect_generator_vars(func, env):
    """
    Generator variables we need to save:

        def seq(start):
            i = start                   # i0
            while True:
                # i = phi(i0, i2)       # <- save this variable
                yield i
                i += 1                  # i2
    """
    # Collect all allocas
    vars = []
    for op in func.blocks.head.ops:
        if op.opcode == 'alloca':
            vars.append(op)

    # Collect all register variables that are used across yield points
    #
    #   %0 = ...
    #   yield
    #   use %0      # <- save %0 in generator state

    registers = []
    seen = set()
    before = set()
    for op in func.ops:
        if op.opcode == 'yield':
            # Process yield point
            seen.update(before)
            before.clear()
            continue

        for arg in flatten(op.args):
            if arg in seen and arg.opcode != 'load':
                # We have seen `arg` before some yield point, and we're
                # using it after a yield point. We need to save this
                # computation!
                registers.append(arg)

    return func.args + vars + registers


def compute_generator_state(func, env, vars):
    """
    Build a struct of generator variables we need to save:

        { i: int64, %5: float32 }
    """
    context = env['flypy.typing.context']
    names = [var.result for var in vars]
    types = [context[var] for var in vars]
    return struct_(zip(names, types))


def rewrite_state_variables(func, env, vars):
    """Rewrite generator state variable accesses"""
    for var in vars:
        if var.opcode == 'alloca':
            _rewrite_alloca(func, env, var)
        else:
            _rewrite_registers(func, env, var)


def _rewrite_alloca(func, env, var):
    """
    Rewrite stack variables:

        %x = alloca int64
        %0 = load %0
        %1 = mul %0 2
        store %1 %x

    to:

        %0 = getfield(gen, 'x')
        %1 = mul %0 2
        setfield(gen, 'x', %1)
    """
    g_state = func.args[0]

    for use in func.uses[var]:
        attr = use.result
        if use.opcode == 'load':
            newop = Op('getfield', use.type, [g_state, attr], use.result)
        else:
            assert use.opcode == 'store', "TODO: use of allocas address"
            val = use.args[0]
            newop = Op('setfield', use.type, [g_state, attr, val], use.result)

        use.replace(newop)

    var.delete()


def _rewrite_registers(func, env, op):
    """
    Rewrite register uses that persist across yield points:

        %1 = mul %0 2
        yield %y
        %2 = add %1 %s
        ...

    to:

        %1 = mul %0 2
        setfield(gen, 'x', %1)
        yield %y
        %t1 = getfield(gen, 'x')
        %2 = add %t1 %s
        ...
    """
    context = env['flypy.typing.context']

    g_state = func.args[0]
    builder = Builder(func)
    builder.position_after(op)

    attr = op.result
    if not isinstance(op, FuncArg):
        setfield = builder.setfield(g_state, attr, op)
        context[setfield] = void

    for use in func.uses[op]:
        builder.position_before(use)
        val = builder.getfield(op.type, g_state, attr)
        use.replace_args({op: val})
        context[val] = context[op]

#===------------------------------------------------------------------===
# Yield
#===------------------------------------------------------------------===

def rewrite_yieldpoints(func, env):
    """
    Rewrite generator yield points to return from the function.

        def f(x):
            for i in range(10):
                yield i
                print "blah"

        def f(gen):
            if gen.label == 1:
                jump(bb2)

            for i in range(10):

              bb1:
                gen.label = 1
                ret i           # formerly, 'yield i'

              bb2:
                print "blah"
                jump(bb_for_loop)


    Returns
    =======
    List of basic blocks where to resume after yield points: [bb2]
    """
    context = env['flypy.typing.context']
    builder = Builder(func)
    opbuidler = OpBuilder()

    g_obj = func.args[0]
    yieldpoints = []

    for op in func.ops:
        if op.opcode == 'yield':
            if func.uses[op]:
                error(env, op, 'Cannot yet handle sending values into generator')

            # Rewrite 'yield i' to 'ret i'
            newop = opbuilder.ret(op.args[0], result=op.result)
            op.replace(newop)

            # Introduce new basic block for resume point (bb2)
            builder.position_after(op)
            _, newblock = builder.splitblock(terminate=True)

            # Update generator label: gen.label = 1
            builder.position_before(op)
            label = OConst(len(yieldpoints) + 1)
            setfield = builder.setfield(g_obj, 'label', label)

            context[setfield] = void
            context[label] = int64

            yieldpoints.append(newblock)

    return yieldpoints



@listify
def find_generators(func, env):
    """
    Find all calls to generators.
    """
    envs = env['flypy.state.envs']

    for op in func.ops:
        if op.opcode == 'call':
            f, args = op.args
            if f in envs and envs[f]['flypy.state.generator'] == 1:
                yield op

#===------------------------------------------------------------------===
# Generator Fusion
#===------------------------------------------------------------------===

def fuse_generators(func, env, consumers):
    """
    Rewrite straightforward uses of generators, i.e. where a generator is
    allocated and consumed by a single consumer loop.
    """
    envs = env['flypy.state.envs']

    for consumer in consumers:
        generator_func = consumer.generator.args[0]

        empty_body = detach_loop(func, consumer)
        move_generator(func, consumer, empty_body)
        clean_loop_body(func, consumer)

        valuemap = inlining.inline_callee(func, consumer.generator,
                                          env, envs[generator_func])
        consume_yields(func, consumer, generator_func, valuemap)


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