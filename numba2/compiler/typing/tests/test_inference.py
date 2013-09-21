# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest

from pykit.ir import verify, findop
from numba2 import passes, environment
from numba2.caching import InferenceCache
from numba2.types import Function, Bool, Int, Float
from numba2.compiler.frontend import translate

cache = InferenceCache()

__gt__ = translate(lambda x, y: x > y)

int32 = Int[32]
float32 = Float[32]
int32.fields['__gt__'] = (__gt__, Function[int32, int32, Bool])
cache.typings[__gt__, (int32, int32)] = (None, Function[int32, int32, Bool])

pipeline = passes.frontend + passes.typing

#===------------------------------------------------------------------===
# Helpers
#===------------------------------------------------------------------===

def get(f, argtypes):
    env = dict(environment.fresh_env(), **{'numba.typing.cache': cache})
    func, env = passes.translate(f, argtypes, env=env, passes=pipeline)
    context = env['numba.typing.context']
    signature = env['numba.typing.signature']
    return func, context, signature

#===------------------------------------------------------------------===
# Tests
#===------------------------------------------------------------------===

class TestInfer(unittest.TestCase):

    def test_simple(self):
        def simple(a, b):
            return 1

        f, context, signature = get(simple, [int32, int32])
        self.assertEqual(signature, Function[int32, int32, int32])

    def test_branch(self):
        def branch(a, b):
            if a > b:
                result = a
            else:
                result = b
            return result

        f, context, signature = get(branch, [int32, int32])
        self.assertEqual(signature, Function[int32, int32, int32])
        type = context[findop(f, 'call')]
        self.assertEqual(type, set([Bool]))

    def test_loop(self):
        def loop(a, b):
            result = a
            while a > b:
                result = b
            return result

        f, context, signature = get(loop, [int32, int32])
        self.assertEqual(signature, Function[int32, int32, int32])
        type = context[findop(f, 'call')]
        self.assertEqual(type, set([Bool]))


if __name__ == '__main__':
    # TestInfer('test_simple').debug()
    # TestInfer('test_branch').debug()
    # TestInfer('test_loop').debug()
    unittest.main()