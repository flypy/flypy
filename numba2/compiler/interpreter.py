# -*- coding: utf-8 -*-

"""
IR interpreter. Run a series of translation phases on a numba function, and
interpreter the IR with the arguments.
"""

from __future__ import print_function, division, absolute_import

from numba2 import typeof, jit
from numba2.compiler.frontend import translate, interpret
from numba2.pipeline import environment, phase

from pykit.ir import interp, tracing

#===------------------------------------------------------------------===
# Helpers
#===------------------------------------------------------------------===

def expect(nb_func, phase, args, expected, handlers=None, debug=False):
    result = interpret(nb_func, phase, args, handlers, debug)
    assert result == expected, "Got %s, expected %s" % (result, expected)

def interpret(nb_func, phase, args, handlers=None, debug=False):
    # Translate numba function
    argtypes = [typeof(arg) for arg in args]
    env = environment.fresh_env(nb_func, argtypes)
    f, env = phase(nb_func, env)

    if debug:
        print("---------------- Interpreting function %s ----------------" % (
                                                                     f.name,))
        print(f)
        print("----------------------- End of %s ------------------------" % (
                                                                     f.name,))
        tracer = tracing.Tracer()
    else:
        tracer = tracing.DummyTracer()

    # Interpreter function
    env.setdefault('interp.handlers', {}).update(handlers or {})
    return interp.run(f, env, args=args, tracer=tracer)