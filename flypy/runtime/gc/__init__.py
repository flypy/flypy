# -*- coding: utf-8 -*-

"""
Garbage collection package.
"""

from __future__ import print_function, division, absolute_import

from . import boehm

impls = {
    "boehm": boehm,
}

def gc_impl(name):
    return impls[name]