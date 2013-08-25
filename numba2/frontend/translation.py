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

from numba2.errors import error_context, CompileError, EmptyStackError
from .bytecode import ByteCode

from pykit.ir import Function, Builder, Op, Const, ops
from pykit import types

#===------------------------------------------------------------------===
# Entrypoint
#===------------------------------------------------------------------===

def translate(func):
    """
    Entry point.

    Parameters
    ----------

    func : Python function
        Python function to translate

    Returns : pykit.ir.Function
        Untyped pykit function. All types are Opaque unless they are constant.
    """
    t = Translate(func)
    t.initialize()
    t.interpret()
    return t.dst

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
    }

class Translate(object):
    """
    Translate bytecode to untypes pykit IR.
    """

    def __init__(self, func):
        self.func = func
        self.bytecode = ByteCode(func)

        self.blocks = {}      # offset -> Block
        self.allocas = {}     # varname -> alloca
        # self.blockstacks = {} # Block -> stack
        self._stack = []

        self.scopes = []

        self.varnames = self.bytecode.code.co_varnames
        self.consts = self.bytecode.code.co_consts
        self.names = self.bytecode.code.co_names
        self.argnames = self.varnames[:self.bytecode.code.co_argcount]

        self.globals = dict(vars(__builtin__))
        self.builtins = set(self.globals.values())
        self.globals.update(self.func.func_globals)

        argspec = inspect.getargspec(self.func)
        assert not argspec.defaults, "does not support defaults"
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
            # self.blockstacks[block] = []

        # Setup Variables
        self.builder.position_at_beginning(self.dst.startblock)
        for varname in self.varnames:
            stackvar = self.builder.alloca(types.Opaque, [])
            self.allocas[varname] = stackvar

            # Initialize function arguments
            if varname in self.argnames:
                self.builder.store(self.dst.get_arg(varname), stackvar)

    # def interpret(self):
    #     # TODO: Why is this not a simple linear pass?
    #
    #     pending_run = set([0])
    #     processed = set()
    #
    #     # interpretation loop
    #     while pending_run:
    #         offset = min(pending_run)
    #         pending_run.discard(offset)
    #         if offset not in processed:    # don't repeat the work
    #             processed.add(offset)
    #             self.builder.position_at_end(self.blocks[offset])
    #
    #             while offset in self.bytecode:
    #                 inst = self.bytecode[offset]
    #                 self.op(inst)
    #                 offset = inst.next
    #                 if self.curblock.is_terminated():
    #                     break
    #
    #                 if offset in self.blocks:
    #                     pending_run.add(offset)
    #                     if not self.curblock.is_terminated():
    #                         self.jump(target=self.blocks[offset])
    #                     break

    def interpret(self):
        prevblock = self.dst.startblock
        for inst in self.bytecode:
            if inst.offset in self.blocks:
                # Block switch
                curblock = self.blocks[inst.offset]
                if prevblock != curblock:
                    if not prevblock.is_terminated():
                        self.jump(curblock)
                    self.builder.position_at_end(curblock)
                    prevblock = curblock
            elif self.curblock.is_terminated():
                # Dead code
                # TODO: Is this really necessary here?
                continue

            self.op(inst)

    def op(self, inst):
        with error_context(lineno=inst.lineno):
            self.lineno = inst.lineno
            attr = 'op_%s' % inst.opname.replace('+', '_')
            fn = getattr(self, attr, self.generic_op)
            fn(inst)

    def generic_op(self, inst):
        raise NotImplementedError(inst)

    @property
    def stack(self):
        return self._stack
        # return self.blockstacks[self.builder.basic_block]

    @property
    def curblock(self):
        return self.builder.basic_block

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
        if not self.stack:
            raise EmptyStackError
        else:
            return self.stack[-1]

    def pop(self):
        if not self.stack:
            raise EmptyStackError
        else:
            return self.stack.pop()

    def call(self, func, args=()):
        self.push_insert('pycall', func, *args)

    def binary_op(self, op):
        rhs = self.pop()
        lhs = self.pop()
        self.call(op, args=(lhs, rhs))

    def unary_op(self, op):
        tos = self.pop()
        self.call(op, args=(tos,))

    def jump(self, target):
        self.insert('jump', target)

    def jump_if(self, cond, truebr, falsebr):
        self.insert('cbranch', cond, truebr, falsebr)

    # ------- op_* ------- #

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
        self.insert('ret', val)

    def op_SETUP_LOOP(self, inst):
        self.scopes.append((inst.next, inst.next + inst.arg))

    def op_POP_BLOCK(self, inst):
        self.scopes.pop()

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
        self.push_insert('getiter', self.pop())

    def op_POP_TOP(self, inst):
        self.pop()

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
        iterobj = self.pop()
        delta = inst.arg
        loopexit = self.blocks[inst.next + delta]

        self.insert('exc_setup', [loopexit])
        self.push_insert('next', iterobj)

        with self.builder.at_front(loopexit):
            self.insert('exc_catch', StopIteration)

    def op_BREAK_LOOP(self, inst):
        scope = self.scopes[-1]
        self.jump(target=self.blocks[scope[1]])

    def op_BUILD_TUPLE(self, inst):
        count = inst.arg
        items = [self.pop() for _ in range(count)]
        ordered = [i for i in reversed(items)]
        # if all(it.opcode == 'const' for it in ordered):   # create const tuple
        #     self.push_insert('const', tuple(i.value for i in ordered))
        # else:
        self.push_insert('new_tuple', ordered)

    def op_LOAD_ATTR(self, inst):
        attr = self.names[inst.arg]
        obj = self.pop()
        self.insert('getfield', obj, attr)

    def op_LOAD_GLOBAL(self, inst):
        name = self.names[inst.arg]
        if name not in self.globals:
            raise NameError("Could not resolve %r at compile time" % name)
        value = self.globals[name]
        self.push(Const(value, types.Opaque))

    def op_LOAD_FAST(self, inst):
        name = self.varnames[inst.arg]
        self.push_insert('load', self.allocas[name])

    def op_LOAD_CONST(self, inst):
        const = self.consts[inst.arg]
        self.push(Const(const))

    def op_STORE_FAST(self, inst):
        value = self.pop()
        name = self.varnames[inst.arg]
        self.insert('store', value, self.allocas[name])

    def op_STORE_SUBSCR(self, inst):
        tos0 = self.pop()
        tos1 = self.pop()
        tos2 = self.pop()
        self.insert('setitem', tos1, tos0, tos2)
        self.pop()

    def op_UNPACK_SEQUENCE(self, inst):
        value = self.pop()
        itemct = inst.arg
        for i in reversed(range(itemct)):
            self.push_insert('unpack', value, i, itemct)

    def op_COMPARE_OP(self, inst):
        opfunc = COMPARE_OP_FUNC[dis.cmp_op[inst.arg]]
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

    def op_RAISE_VARARGS(self, inst):
        nargs = inst.arg
        if nargs == 3:
            raise CompileError("Traceback argument to raise not supported")

        args = list(reversed([self.pop() for _ in range(nargs)]))
        exc_type = args[0]
        if exc_type.type != types.Exception:
            raise CompileError(
                "Expected a statically known exception type, "
                "got %s" % (exc_type,))

        self.insert('exc_throw', make_exc(*args))


#---------------------------------------------------------------------------
# Internals

def func_name(func):
    return ".".join([func.__module__, func.__name__])

def slicearg(v):
    """Construct an argument to a slice instruction"""
    return Const(v, types.Int64)

def make_exc(exc_type, exc_value=None):
    """Construct an exception IR value from the exception type and value"""
    # TODO: implement
    return exc_type