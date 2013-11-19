# -*- coding: utf-8 -*-

"""
First stage in numba translation. Translates Python bytecode to untyped
pykit IR.

Initially adapted from numbapro/npm/symbolic.py by Siu Kwan Lam.
"""

from __future__ import print_function, division, absolute_import

import __builtin__
import inspect
import dis
import operator
import collections

from numba2.errors import error_context, CompileError, EmptyStackError
from numba2.runtime.obj import tupleobject
from .bytecode import ByteCode

from pykit.ir import Function, Builder, Op, Const, Value, ops
from pykit import types

#===------------------------------------------------------------------===
# Translation
#===------------------------------------------------------------------===

COMPARE_OP_FUNC = {
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne,
    'in': operator.contains,
    'is': operator.is_,
    'exception match': isinstance,
}

const = lambda val: Const(val, types.Opaque)

class Translate(object):
    """
    Translate bytecode to untypes pykit IR.
    """

    def __init__(self, func):
        self.func = func
        self.bytecode = ByteCode(func)

        # -------------------------------------------------
        # Find predecessors

        self.blocks = {}            # offset -> Block
        self.block2offset = {}      # Block -> offset
        self.allocas = {}           # varname -> alloca
        self.stacks = {}            # Block -> value stack
        self.exc_handlers = set()   # { Block }

        # -------------------------------------------------
        # Block stacks

        self.block_stack   = []
        self.loop_stack    = []
        self.except_stack  = []
        self.finally_stack = []

        # -------------------------------------------------
        # CFG

        self.predecessors = collections.defaultdict(set)
        self.phis = collections.defaultdict(list)

        # -------------------------------------------------
        # Variables and scoping

        self.code = self.bytecode.code
        self.varnames = self.bytecode.code.co_varnames
        self.consts = self.bytecode.code.co_consts
        self.names = self.bytecode.code.co_names
        self.argnames = self.varnames[:self.bytecode.code.co_argcount]

        self.globals = dict(vars(__builtin__))
        self.builtins = set(self.globals.values())
        self.globals.update(self.func.func_globals)

        # -------------------------------------------------
        # Error checks

        argspec = inspect.getargspec(self.func)
        assert not argspec.varargs, "does not support varargs"
        assert not argspec.keywords, "does not support keywords"

    def initialize(self):
        """Initialize pykit untypes structures"""

        # Setup Function
        sig = types.Function(types.Opaque, [types.Opaque] * len(self.argnames))
        self.dst = Function(func_name(self.func), self.argnames, sig)

        # Setup Builder
        self.builder = Builder(self.dst)

        # Setup Blocks
        for offset in self.bytecode.labels:
            block = self.dst.new_block("Block%d" % offset)
            self.blocks[offset] = block
            self.stacks[block] = []

        # Setup Variables
        self.builder.position_at_beginning(self.dst.startblock)
        for varname in self.varnames:
            stackvar = self.builder.alloca(types.Pointer(types.Opaque))
            self.allocas[varname] = stackvar

            # Initialize function arguments
            if varname in self.argnames:
                self.builder.store(self.dst.get_arg(varname), stackvar)

    def interpret(self):
        self.curblock = self.dst.startblock

        for inst in self.bytecode:
            if inst.offset in self.blocks:
                # Block switch
                newblock = self.blocks[inst.offset]
                if self.curblock != newblock:
                    self.switchblock(newblock)
            elif self.curblock.is_terminated():
                continue

            self.op(inst)

        # -------------------------------------------------
        # Finalize

        self.update_phis()

    def op(self, inst):
        with error_context(lineno=inst.lineno):
            self.lineno = inst.lineno
            attr = 'op_%s' % inst.opname.replace('+', '_')
            fn = getattr(self, attr, self.generic_op)
            fn(inst)

    def generic_op(self, inst):
        raise NotImplementedError(inst)

    def switchblock(self, newblock):
        """
        Switch to a new block and merge incoming values from the stacks.
        """
        if not self.curblock.is_terminated():
            self.jump(newblock)

        self.builder.position_at_end(newblock)
        self.curblock = newblock

        # -------------------------------------------------
        # Find predecessors

        if newblock in self.exc_handlers:
            self.push_insert('exc_fetch')
            self.push_insert('exc_fetch_value')
            self.push_insert('exc_fetch_tb')

        # -------------------------------------------------
        # Find predecessors

        incoming = self.predecessors.get(newblock)
        if not incoming:
            return

        # -------------------------------------------------
        # Merge stack values

        stack = max([self.stacks[block] for block in incoming], key=len)
        for value in stack:
            phi = self.push_insert('phi', [], [])
            self.phis[newblock].append(phi)

    def update_phis(self):
        for block in self.dst.blocks:
            phis = self.phis[block]
            preds  = list(self.predecessors[block])
            stacks = [self.stacks[pred] for pred in preds]
            stacklen = len(phis)

            # -------------------------------------------------
            # Sanity check

            assert all(len(stack) == stacklen for stack in stacks)

            if not preds or not stacklen:
                continue

            # -------------------------------------------------
            # Update φs with stack values from predecessors

            for pos, phi in enumerate(phis):
                values = []
                for pred in preds:
                    value_stack = self.stacks[pred]
                    value = value_stack[-1 - pos]
                    values.append(value)

                phi.set_args([preds, values])

    @property
    def stack(self):
        return self.stacks[self.curblock]

    def insert(self, opcode, *args):
        type = types.Void if ops.is_void(opcode) else types.Opaque
        op = Op(opcode, type, list(args))
        op.add_metadata({'lineno': self.lineno})
        self.builder.emit(op)
        return op

    def push_insert(self, opcode, *args):
        inst = self.insert(opcode, *args)
        self.push(inst)
        return inst

    def push(self, val):
        self.stack.append(val)

    def peek(self):
        """
        Take a peek at the top of stack.
        """
        if not self.stack:
            # Assuming the bytecode is valid, our predecessors must have left
            # some values on the stack.
            # return self._insert_phi()
            raise EmptyStackError
        else:
            return self.stack[-1]

    def pop(self):
        if not self.stack:
            # return self._insert_phi()
            raise EmptyStackError
        else:
            return self.stack.pop()

    def _insert_phi(self):
        with self.builder.at_front(self.curblock):
            phi = self.insert('phi', [], [])

        self.phis[self.curblock].append(phi)
        return phi

    def call(self, func, args=()):
        if not isinstance(func, Value):
            func = const(func)
        return self.push_insert('call', func, list(args))

    def call_pop(self, func, args=()):
        self.call(func, args)
        return self.pop()

    def binary_op(self, op):
        rhs = self.pop()
        lhs = self.pop()
        self.call(op, args=(lhs, rhs))

    def unary_op(self, op):
        tos = self.pop()
        self.call(op, args=(tos,))

    def jump(self, target):
        self.predecessors[target].add(self.curblock)
        self.insert('jump', target)

    def jump_if(self, cond, truebr, falsebr):
        self.predecessors[truebr].add(self.curblock)
        self.predecessors[falsebr].add(self.curblock)
        self.insert('cbranch', cond, truebr, falsebr)

    # ------- stack ------- #

    def op_POP_BLOCK(self, inst):
        block = self.block_stack.pop()
        if isinstance(block, LoopBlock):
            self.loop_stack.pop()
        elif isinstance(block, ExceptionBlock):
            self.except_stack.pop()
        elif isinstance(block, FinallyBlock):
            self.finally_stack.pop()

    def op_POP_TOP(self, inst):
        self.pop()

    def op_DUP_TOP(self, inst):
        value = self.pop()
        self.push(value)
        self.push(value)

    def op_DUP_TOPX(self, inst):
        count = inst.arg
        self.stack.extend(self.stack[-count:])

    def op_ROT_TWO(self, inst):
        one = self.pop()
        two = self.pop()
        self.push(one)
        self.push(two)

    def op_ROT_THREE(self, inst):
        one = self.pop()
        two = self.pop()
        three = self.pop()
        self.push(one)
        self.push(three)
        self.push(two)

    def op_ROT_FOUR(self, inst):
        one = self.pop()
        two = self.pop()
        three = self.pop()
        four = self.pop()
        self.push(one)
        self.push(four)
        self.push(three)
        self.push(two)

    # ------- control flow ------- #

    def op_POP_JUMP_IF_TRUE(self, inst):
        falsebr = self.blocks[inst.next]
        truebr = self.blocks[inst.arg]
        self.jump_if(self.pop(), truebr, falsebr)

    def op_POP_JUMP_IF_FALSE(self, inst):
        truebr = self.blocks[inst.next]
        falsebr = self.blocks[inst.arg]
        self.jump_if(self.pop(), truebr, falsebr)

    def op_JUMP_IF_TRUE(self, inst):
        falsebr = self.blocks[inst.next]
        truebr = self.blocks[inst.next + inst.arg]
        self.jump_if(self.peek(), truebr, falsebr)

    def op_JUMP_IF_FALSE(self, inst):
        truebr = self.blocks[inst.next]
        falsebr = self.blocks[inst.next + inst.arg]
        self.jump_if(self.peek(), truebr, falsebr)

    def op_JUMP_IF_TRUE_OR_POP(self, inst):
        falsebr = self.blocks[inst.next]
        truebr = self.blocks[inst.arg]
        self.jump_if(self.peek(), truebr, falsebr)

    def op_JUMP_IF_FALSE_OR_POP(self, inst):
        truebr = self.blocks[inst.next]
        falsebr = self.blocks[inst.arg]
        self.jump_if(self.peek(), truebr, falsebr)

    def op_JUMP_ABSOLUTE(self, inst):
        target = self.blocks[inst.arg]
        self.jump(target)

    def op_JUMP_FORWARD(self, inst):
        target = self.blocks[inst.next + inst.arg]
        self.jump(target)

    def op_RETURN_VALUE(self, inst):
        val = self.pop()
        if isinstance(val, Const) and val.const is None:
            val = None # Generate a bare 'ret' instruction
        self.insert('ret', val)

    def op_CALL_FUNCTION(self, inst):
        argc = inst.arg & 0xff
        kwsc = (inst.arg >> 8) & 0xff

        def pop_kws():
            val = self.pop()
            key = self.pop()
            if key.opcode != 'const':
                raise ValueError('keyword must be a constant')
            return key.value, val

        kws = list(reversed([pop_kws() for i in range(kwsc)]))
        args = list(reversed([self.pop() for i in range(argc)]))
        assert not kws, "Keyword arguments not yet supported"

        func = self.pop()
        self.call(func, args)

    def op_GET_ITER(self, inst):
        self.call(iter, [self.pop()])

    def op_FOR_ITER(self, inst):
        """
        Translate a for loop to:

            it = getiter(iterable)
            try:
                while 1:
                    i = next(t)
                    ...
            except StopIteration:
                pass
        """
        iterobj = self.peek()
        delta = inst.arg
        loopexit = self.blocks[inst.next + delta]

        # -------------------------------------------------
        # Try

        self.insert('exc_setup', [loopexit])
        self.call(next, [iterobj])

        # We assume a 1-to-1 block mapping, resolve a block split in a
        # later pass
        self.insert('exc_end')

        # -------------------------------------------------
        # Catch

        with self.builder.at_front(loopexit):
            self.insert('exc_catch', [Const(StopIteration, type=types.Exception)])

    def op_BREAK_LOOP(self, inst):
        scope = self.loops[-1]
        self.jump(target=self.blocks[scope[1]])

    def op_BUILD_TUPLE(self, inst):
        count = inst.arg
        items = [self.pop() for _ in range(count)]
        ordered = list(reversed(items))
        if all(isinstance(item, Const) for item in ordered):
            # create constant tuple
             self.push(const(tuple(item.const for item in ordered)))
        elif len(ordered) < tupleobject.STATIC_THRESHOLD:
            # Build static tuple
            result = self.call_pop(tupleobject.EmptyTuple)
            for item in items:
                result = self.call_pop(tupleobject.StaticTuple,
                                       args=(item, result))
            self.push(result)
        else:
            raise NotImplementedError("Generic tuples")

    def op_LOAD_ATTR(self, inst):
        attr = self.names[inst.arg]
        obj = self.pop()
        if isinstance(obj, Const) and hasattr(obj.const, attr):
            val = getattr(obj.const, attr)
            self.push(const(val))
        else:
            self.push_insert('getfield', obj, attr)

    def op_LOAD_GLOBAL(self, inst):
        name = self.names[inst.arg]
        if name not in self.globals:
            raise NameError("Could not resolve %r at compile time" % name)
        value = self.globals[name]
        self.push(const(value))

    def op_LOAD_DEREF(self, inst):
        i = inst.arg
        cell = self.func.__closure__[i]
        value = cell.cell_contents
        self.push(const(value))

    def op_LOAD_FAST(self, inst):
        name = self.varnames[inst.arg]
        self.push_insert('load', self.allocas[name])

    def op_LOAD_CONST(self, inst):
        val = self.consts[inst.arg]
        self.push(const(val))

    def op_STORE_FAST(self, inst):
        value = self.pop()
        name = self.varnames[inst.arg]
        self.insert('store', value, self.allocas[name])

    def op_STORE_ATTR(self, inst):
        attr = self.names[inst.arg]
        obj = self.pop()
        value = self.pop()
        self.insert('setfield', obj, attr, value)

    def op_STORE_SUBSCR(self, inst):
        tos0 = self.pop()
        tos1 = self.pop()
        tos2 = self.pop()
        self.call(operator.setitem, (tos1, tos0, tos2))
        self.pop()

    def op_UNPACK_SEQUENCE(self, inst):
        value = self.pop()
        itemct = inst.arg
        for i in reversed(range(itemct)):
            self.push_insert('unpack', value, i, itemct)

    def op_COMPARE_OP(self, inst):
        opname = dis.cmp_op[inst.arg]

        if opname == 'not in':
            self.binary_op('in')
            self.unary_op('not')
        elif opname == 'is not':
            self.binary_op('is')
            self.unary_op('not')
        else:
            opfunc = COMPARE_OP_FUNC[opname]
            self.binary_op(opfunc)

    def op_UNARY_POSITIVE(self, inst):
        self.unary_op(operator.pos)

    def op_UNARY_NEGATIVE(self, inst):
        self.unary_op(operator.neg)

    def op_UNARY_INVERT(self, inst):
        self.unary_op(operator.invert)

    def op_UNARY_NOT(self, inst):
        self.unary_op(operator.not_)

    def op_BINARY_SUBSCR(self, inst):
        self.binary_op(operator.getitem)

    def op_BINARY_ADD(self, inst):
        self.binary_op(operator.add)

    def op_BINARY_SUBTRACT(self, inst):
        self.binary_op(operator.sub)

    def op_BINARY_MULTIPLY(self, inst):
        self.binary_op(operator.mul)

    def op_BINARY_DIVIDE(self, inst):
        self.binary_op(operator.floordiv)

    def op_BINARY_FLOOR_DIVIDE(self, inst):
        self.binary_op(operator.floordiv)

    def op_BINARY_TRUE_DIVIDE(self, inst):
        self.binary_op(operator.truediv)

    def op_BINARY_MODULO(self, inst):
        self.binary_op(operator.mod)

    def op_BINARY_POWER(self, inst):
        self.binary_op(operator.pow)

    def op_BINARY_RSHIFT(self, inst):
        self.binary_op(operator.rshift)

    def op_BINARY_LSHIFT(self, inst):
        self.binary_op(operator.lshift)

    def op_BINARY_AND(self, inst):
        self.binary_op(operator.and_)

    def op_BINARY_OR(self, inst):
        self.binary_op(operator.or_)

    def op_BINARY_XOR(self, inst):
        self.binary_op(operator.xor)

    def op_INPLACE_ADD(self, inst):
        self.binary_op(operator.add)

    def op_INPLACE_SUBTRACT(self, inst):
        self.binary_op(operator.sub)

    def op_INPLACE_MULTIPLY(self, inst):
        self.binary_op(operator.mul)

    def op_INPLACE_DIVIDE(self, inst):
        self.binary_op(operator.floordiv)

    def op_INPLACE_FLOOR_DIVIDE(self, inst):
        self.binary_op(operator.floordiv)

    def op_INPLACE_TRUE_DIVIDE(self, inst):
        self.binary_op(operator.truediv)

    def op_INPLACE_MODULO(self, inst):
        self.binary_op(operator.mod)

    def op_INPLACE_POWER(self, inst):
        self.binary_op(operator.pow)

    def op_INPLACE_RSHIFT(self, inst):
        self.binary_op(operator.rshift)

    def op_INPLACE_LSHIFT(self, inst):
        self.binary_op(operator.lshift)

    def op_INPLACE_AND(self, inst):
        self.binary_op(operator.and_)

    def op_INPLACE_OR(self, inst):
        self.binary_op(operator.or_)

    def op_INPLACE_XOR(self, inst):
        self.binary_op(operator.xor)

    def op_SLICE_0(self, inst):
        tos = self.pop()
        sl = self.insert('slice', *map(slicearg, [None, None, None]))
        self.call(operator.getitem, args=(tos, sl))

    def op_SLICE_1(self, inst):
        start = self.pop()
        tos = self.pop()
        sl = self.insert('slice', *map(slicearg, [start, None, None]))
        self.call(operator.getitem, args=(tos, sl))

    def op_SLICE_2(self, inst):
        stop = self.pop()
        tos = self.pop()
        sl = self.insert('slice', *map(slicearg, [None, stop, None]))
        self.call(operator.getitem, args=(tos, sl))

    def op_SLICE_3(self, inst):
        stop = self.pop()
        start = self.pop()
        tos = self.pop()
        sl = self.insert('slice', *map(slicearg, [start, stop, None]))
        self.call(operator.getitem, args=(tos, sl))

    def op_BUILD_SLICE(self, inst):
        argc = inst.arg
        tos = [self.pop() for _ in range(argc)]

        if argc == 2:
            self.push_insert('slice', *map(slicearg, [tos[1], tos[0], None]))
        elif argc == 3:
            self.push_insert('slice', *map(slicearg, [tos[2], tos[1], tos[0]]))
        else:
            raise Exception('unreachable')

    # ------- Exceptions ------- #

    def op_RAISE_VARARGS(self, inst):
        nargs = inst.arg
        if nargs == 3:
            raise CompileError("Traceback argument to raise not supported")

        args = list(reversed([self.pop() for _ in range(nargs)]))

        if self.except_stack:
            except_block = self.except_stack[-1]
            self.predecessors[except_block].add(self.curblock)

        self.insert('exc_throw', *args)

    # ------- Blocks ------- #

    def op_SETUP_LOOP(self, inst):
        loop_block = self.blocks[inst.next + inst.arg]

        block = LoopBlock(loop_block)
        self.block_stack.append(block)
        self.loop_stack.append(block)

    def op_SETUP_EXCEPT(self, inst):
        except_block = self.blocks[inst.next + inst.arg]
        self.predecessors[except_block].add(self.curblock)
        self.exc_handlers.add(except_block)

        with self.builder.at_front(self.curblock):
            self.builder.exc_setup([except_block])

        block = ExceptionBlock(except_block)
        self.block_stack.append(block)
        self.except_stack.append(block)

    def op_SETUP_FINALLY(self, inst):
        finally_block = self.blocks[inst.next + inst.arg]
        self.predecessors[finally_block].add(self.curblock)

        block = FinallyBlock(finally_block)
        self.block_stack.append(block)
        self.finally_stack.append(block)

    def op_END_FINALLY(self, inst):
        self.pop()
        self.pop()
        self.pop()
        # self.insert('end_finally')

    # ------- print ------- #

    def op_PRINT_ITEM(self, inst):
        self.call(print, [self.pop()])

    def op_PRINT_NEWLINE(self, inst):
        self.call(print, [const('\n')])

    # ------- Misc ------- #

    def op_STOP_CODE(self, inst):
        pass

#===------------------------------------------------------------------===
# Internals
#===------------------------------------------------------------------===

def func_name(func):
    if func.__module__:
        return ".".join([func.__module__, func.__name__])
    return func.__name__

def slicearg(v):
    """Construct an argument to a slice instruction"""
    return Const(v, types.Int64)

#===------------------------------------------------------------------===
# Exceptions
#===------------------------------------------------------------------===

Exc = collections.namedtuple('Exc', ['arg'])
Val = collections.namedtuple('Val', ['arg'])
Tb  = collections.namedtuple('Tb', ['arg'])

#===------------------------------------------------------------------===
# Blocks
#===------------------------------------------------------------------===

class LoopBlock(object):
    def __init__(self, end):
        self.end = end

class ExceptionBlock(object):
    def __init__(self, first_except_block):
        self.first_except_block = first_except_block

class FinallyBlock(object):
    def __init__(self, finally_block):
        self.finally_block = finally_block

#===------------------------------------------------------------------===
# Globals
#===------------------------------------------------------------------===

def lookup_global(func, name, env):
    func_globals = env['numba.state.func_globals']

    if (func is not None and name in func.__code__.co_freevars and
            func.__closure__):
        cell_idx = func.__code__.co_freevars.index(name)
        cell = func.__closure__[cell_idx]
        value = cell.cell_contents
    elif name in func_globals:
        value = func_globals[name]
    else:
        raise CompileError("No global named '%s'" % (name,))

    return value
