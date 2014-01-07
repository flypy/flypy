# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from numba2 import jit

@jit
class Foo(object):
    layout = []
