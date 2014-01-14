# -*- coding: utf-8 -*-

"""
flypy function wrapper.
"""

from __future__ import print_function, division, absolute_import
import types
import ctypes
from functools import partial
from itertools import starmap
import copy

from flypy.rules import typeof
from flypy.compiler.overloading import lookup_previous, overload, Dispatcher
from flypy.compiler.signature import dummy_signature, flatargs
from flypy.linker import llvmlinker

# TODO: Reuse flypy.flypywrapper.pyx for autojit Python entry points

class FunctionWrapper(object):
    """
    Result of @jit for functions.
    """

    def __init__(self, parent, py_func, abstract=False,
                 opaque=False, target="cpu"):
        self.parent = parent            # parent function, which we are a
                                        # copy of

        self._dispatcher = Dispatcher() # dispatcher for overloads
        self._pending_overloads  = []   # pending overloads
        self.closed = False             # whether this function is closed
                                        # for extension

        self.py_func = py_func
        # self.signature = signature
        self.abstract = abstract

        self.llvm_funcs = {}
        self.ctypes_funcs = {}
        self.link_state = {}
        self.envs = {}

        self.opaque = opaque
        self.implementor = None
        self.target = target

    def __call__(self, *args, **kwargs):
        from flypy.representation import byref, stack_allocate
        from flypy.conversion import (
            toctypes, fromctypes, toobject, fromobject, ctype)
        #from flypy.support.ctypes_support import CTypesStruct
        #from flypy.types import Function

        # Keep this alive for the duration of the call
        keepalive = list(args) + list(kwargs.values())

        # Order arguments
        args = flatargs(self.py_func, args, kwargs)

        # Translate
        cfunc, restype = self.translate([typeof(x) for x in args.flat])

        # Construct flypy values
        argtypes = [typeof(x) for x in args]
        arg_objs = list(starmap(fromobject, zip(args, argtypes)))

        # Map flypy values to a ctypes representation
        args = []
        for arg, argtype in zip(arg_objs, argtypes):
            c_arg = toctypes(arg, argtype, keepalive)
            if byref(argtype) and stack_allocate(argtype):
                c_arg = ctypes.pointer(c_arg)
            args.append(c_arg)

        # We need this cast since the ctypes function constructed from LLVM
        # IR has different structs (which are structurally equivalent)
        c_restype = ctype(restype)
        if byref(restype):
            c_result = c_restype() # dummy result value
            args.append(ctypes.pointer(c_result))
            c_restype = None # void

        c_signature = ctypes.PYFUNCTYPE(c_restype, *[type(arg) for arg in args])
        cfunc = ctypes.cast(cfunc, c_signature)

        # Handle calling convention
        if byref(restype):
            cfunc(*args)
        else:
            c_result = cfunc(*args)

        # Map ctypes result back to a python value
        result = fromctypes(c_result, restype)
        result_obj = toobject(result, restype)

        return result_obj

    def translate(self, argtypes, target=None):
        target = target or self.target

        key = tuple(argtypes), target
        if key in self.ctypes_funcs:
            env = self.envs[key]
            return self.ctypes_funcs[key], env["flypy.typing.restype"]

        # Translate
        llvm_func, env = self._do_lower(target, argtypes)
        cfunc = env["codegen.llvm.ctypes"]

        # Cache
        self.llvm_funcs[key] = llvm_func
        if cfunc is not None:
            self.ctypes_funcs[key] = cfunc
        self.envs[key] = env

        return cfunc, env["flypy.typing.restype"]

    def _do_lower(self, target, argtypes):
        from .pipeline import phase, environment
        env = environment.fresh_env(self, argtypes, target)
        llvm_func, env = phase.codegen(self, env)
        return llvm_func, env

    def link(self, argtypes, target):
        key = tuple(argtypes), target
        env = self.envs[key]
        module = self._do_linkage(env)
        return module

    def _do_linkage(self, env):
        target = env["flypy.target"]
        thismod = env["codegen.llvm.module"]
        thisfunc = env["flypy.state.llvm_func"]
        cachekey = target, thisfunc
        if self.link_state.get(cachekey):
            return thismod

        envs = env["flypy.state.envs"]
        depfuncs= env["flypy.state.dependences"]
        depenvs = [envs[f] for f in depfuncs]

        if __debug__:
            for dep in depenvs:
                if dep["flypy.target"] != target:
                    raise AssertionError("Mismatching target")

        depmods = []
        for dep in depenvs:
            fw = dep['flypy.state.function_wrapper']
            if fw is not self:
                mod = fw._do_linkage(dep)
                depmods.append(mod)

        linker = llvmlinker.Linker(thismod)
        for m in depmods:
            linker.add_module(m)
        linker.link()
        thismod.verify()

        self.link_state[cachekey] = True
        return thismod

    def static_compile(self, argtypes, target):
        self.translate(argtypes, target='dpp')
        module = self.link(argtypes, 'dpp')
        return module

    def resolve_dispatcher(self):
        """
        Resolve the dispatcher, that is parse all signatures and build
        a dispatcher object. This happens when the function is needed
        to generate code somewhere. After that it is closed for extension.
        """
        if not self.closed:
            pending = self._pending_overloads
            if self.parent:
                self.parent.resolve_dispatcher()
                pending = self.parent._pending_overloads + pending

            populate_dispatcher(self._dispatcher, pending)
            self.closed = True

        return self._dispatcher

    def overload(self, py_func, signature, **kwds):
        if self.closed:
            raise TypeError(
                "This function has been closed for extension, "
                "it has already been used!")
        self._pending_overloads.append((py_func, signature, kwds))

    def get_llvm_func(self, argtypes, target):
        """Get the LLVM function object for the argtypes.
        """
        key = tuple(argtypes), target
        return self.llvm_funcs[key]

    @property
    def signatures(self):
        return [signature for func, signature, _ in self.overloads]

    @property
    def overloads(self):
        return self.resolve_dispatcher().overloads

    def __str__(self):
        return "<flypy function (%s)>" % str(self.py_func)

    __repr__ = __str__

    def __get__(self, instance, owner=None):
        if instance is not None:
            # TODO: return partial(self, instance)
            return partial(self.py_func, instance)
        return self

    def copy(self):
        fw = FunctionWrapper(self, self.py_func, abstract=self.abstract,
                             opaque=self.opaque)
        fw.implementor = self.implementor
        return fw


def wrap(py_func, signature, scope, inline=False, opaque=False, abstract=False,
         target="cpu", **kwds):
    """
    Wrap a function in a FunctionWrapper. Take care of overloading.
    """
    func = lookup_previous(py_func, [scope])

    if isinstance(func, FunctionWrapper):
        pass
    elif isinstance(func, types.FunctionType) and func != py_func:
        raise TypeError(
            "Function %s in current scope is not overloadable" % (func,))
    else:
        func = FunctionWrapper(None, py_func, abstract=abstract,
                               opaque=opaque, target=target)

    func.overload(py_func, signature, inline=inline, opaque=opaque,
                  abstract=abstract, target=target, **kwds)
    return func

def populate_dispatcher(dispatcher, overloads):
    """
    Populate dispatcher with the given overloads.
    """
    for py_func, signature, kwds in overloads:
        if not signature:
            signature = dummy_signature(py_func)
        overload(signature, dispatcher=dispatcher, **kwds)(py_func)