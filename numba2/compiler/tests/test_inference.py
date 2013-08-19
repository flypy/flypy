# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from pykit.parsing import cirparser
from pykit.ir import verify, interp, findop
from pykit.analysis import cfa

from numba2.compiler.simplify import simplify
from numba2.compiler.typing import Type, Function
from numba2.compiler.inference import infer, InferenceCache

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

bool = Type('Bool')
int32 = Type('Int', 32, False)
float32 = Type('Float', 32)
int32.fields['__gt__'] = Function(bool, int32, int32)

def get(name):
    f = mod.get_function(name)
    cfa.run(f)
    simplify(f)
    return f

class TestInfer(unittest.TestCase):

    def test_simple(self):
        f = get('simple')
        context, signature = infer(cache, f, [int32, int32])
        self.assertEqual(signature, Function(int32, int32, int32))

    def test_branch(self):
        f = get('branch')
        context, signature = infer(cache, f, [int32, int32])
        self.assertEqual(signature, Function(int32, int32, int32))
        type = context[findop(f, 'call')]
        self.assertEqual(type, bool)

    def test_loop(self):
        f = get('loop')
        context, signature = infer(cache, f, [int32, int32])
        self.assertEqual(signature, Function(int32, int32, int32))
        type = context[findop(f, 'call')]
        self.assertEqual(type, bool)


TestInfer('test_simple').debug()
TestInfer('test_branch').debug()
# TestInfer('test_loop').debug()