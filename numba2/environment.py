# -*- coding: utf-8 -*-

"""
Numba compilation environment.
"""

from __future__ import print_function, division, absolute_import

from .utils import FrozenDict
from .caching import (FrontendCache, InferenceCache, OptimizationsCache,
                      CodegenCache)

root_env = FrozenDict({
    'numba.frontend.cache': FrontendCache(),
    'numba.typing.cache':   InferenceCache(),
    'numba.opt.cache':      OptimizationsCache(),
    'numba.codegen.cache':  CodegenCache(),
})