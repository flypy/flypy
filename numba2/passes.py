# -*- coding: utf-8 -*-

"""
Numba passes that perform translation, type inference, code generation, etc.
"""

from __future__ import print_function, division, absolute_import

from .frontend import translate
from .compiler import inference
from .environment import root_env
from .pipeline import run_pipeline

#===------------------------------------------------------------------===
# Utils
#===------------------------------------------------------------------===

def dump(func, env):
    print(func)

#===------------------------------------------------------------------===
# Passes
#===------------------------------------------------------------------===

passes = [
    translate,
    inference,
    dump,
]

#===------------------------------------------------------------------===
# Translation
#===------------------------------------------------------------------===

def translate(py_func, argtypes, env=None, passes=passes):
    if env is None:
        env = dict(root_env)

    if 'numba.typing.argtypes' not in env:
        env['numba.typing.argtypes'] = argtypes

    run_pipeline(py_func, env, passes)