# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import datetime
import unittest

from flypy import jit
from flypy.types import int32, float64
from flypy.pipeline import phase, environment
from flypy.cache import keys, codecache

now = datetime.datetime.now()

class TestCache(unittest.TestCase):

    def setUp(self):
        self.db = codecache.example_db()

    def test_cache_code(self):
        def py_func(x, y):
            return x + y

        nb_func = jit(py_func)
        argtypes = (float64, float64)
        e = environment.fresh_env(nb_func, argtypes)
        lfunc, env = phase.llvm(nb_func, e)
        self.db.insert(py_func, argtypes, 'llvm', lfunc, env, now)
        cached_module, cached_lfunc, cached_env = self.db.lookup(
            py_func, argtypes, 'llvm')

        self.assertEqual(str(lfunc), str(cached_lfunc))


if __name__ == '__main__':
    unittest.main()