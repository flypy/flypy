# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from os.path import dirname, abspath
import unittest

from pykit.utils.pattern import match as pyoverload

from .entrypoints import jit, ijit, abstract
from .compiler import (annotate, overload, overloadable)
from .typing import convert, promote, typeof, typedef, parse
from .types import Type
from .runtime.interfaces.interface import implements

from .passes import translate
from .errors import error, InferError, SpecializeError


__version__ = '0.1'

# ______________________________________________________________________
# numba.test()

root = dirname(dirname(abspath(__file__)))
pattern = "test_*.py"

def test(root=root, pattern=pattern):
    """Run tests and return exit status"""
    tests =  unittest.TestLoader().discover(root, pattern=pattern)
    runner = unittest.TextTestRunner()
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