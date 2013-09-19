# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from pykit.parsing import cirparser
from pykit.ir import verify, interp, findop
from pykit.analysis import cfa

from numba2.frontend import translate
from numba2.caching import InferenceCache
from numba2.compiler.simplify import simplify
from numba2.types import Type, Function, Bool, Int, Float
from numba2.compiler.inference import infer

source = """
#include <pykit_ir.h>

Opaque simple(Opaque a, Opaque b) {
    return 1;
}

Opaque branch(Opaque a, Opaque b) {
    Opaque result;

    if (a > b)
        result = a;
    else
        result = b;

    return result;
}

Opaque loop(Opaque a, Opaque b) {
    Opaque result = a;
    while (a > b) {
        result = b;
    }
    return result;
}
"""

mod = cirparser.from_c(source)
verify(mod)
cache = InferenceCache()

__gt__ = translate(lambda x, y: x > y)

int32 = Int[32]
float32 = Float[32]
int32.fields['__gt__'] = (__gt__, Function(Bool, int32, int32))
cache.typings[__gt__, (int32, int32)] = (None, Function(Bool, int32, int32))

def get(name):
    f = mod.get_function(name)
    cfa.run(f)
    simplify(f)
    return f

class TestInfer(unittest.TestCase):

    def test_simple(self):
        f = get('simple')
        ctx, signature = infer(cache, f, [int32, int32])
        self.assertEqual(signature, Function(int32, int32, int32))

    def test_branch(self):
        f = get('branch')
        ctx, signature = infer(cache, f, [int32, int32])
        self.assertEqual(signature, Function(int32, int32, int32))
        type = ctx.context[findop(f, 'call')]
        self.assertEqual(type, set([bool]))

    def test_loop(self):
        f = get('loop')
        ctx, signature = infer(cache, f, [int32, int32])
        self.assertEqual(signature, Function(int32, int32, int32))
        type = ctx.context[findop(f, 'call')]
        self.assertEqual(type, set([bool]))


if __name__ == '__main__':
    TestInfer('test_simple').debug()
    # TestInfer('test_branch').debug()
    # TestInfer('test_loop').debug()
    unittest.main()