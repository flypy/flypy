# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from os.path import dirname, abspath
import sys
import unittest

from pykit.utils.pattern import match as pyoverload

from .entrypoints import jit, ijit, sjit, abstract, unijit
from .compiler import (annotate, overload, overloadable)
from .typing import (overlay, parse, unify, free, UnificationError)
from .rules import typeof, convert, promote, typejoin, is_numba_type
from .types import *
from .conversion import toobject, fromobject, toctypes, fromctypes, ctype
from .runtime import cast
from .runtime.interfaces.interface import implements
from .runtime.ffi import sizeof, malloc, libc
from .runtime import builtins as bltins
from .runtime.lib.librt import debug
from .runtime.special import addressof
from .runtime import ffi
from .runtime.obj.core import NULL

# Trigger extraneous definitions
from .runtime import formatting
from .runtime import mathlib

from .pipeline.passes import translate
from .errors import error, InferError, SpecializeError

__version__ = '0.1'

# ______________________________________________________________________
# numba.test()

root = dirname(dirname(abspath(__file__)))
pattern = "test_*.py"

def test(root=root, pattern=pattern):
    """Run tests and return exit status"""
    tests =  unittest.TestLoader().discover(root, pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=1 + ('-v' in sys.argv))
    result = runner.run(tests)
    return not result.wasSuccessful()

def run_tests(dirs, pattern=pattern, failfast=True):
    """Run tests in specified order, quitting on the first failure if failfast"""
    status = 0
    for dir in dirs:
        status |= test(dir, pattern)
        if failfast and status != 0:
            break

    return status
