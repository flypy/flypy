# -*- coding: utf-8 -*-

"""
Rich compare mixin.

Copied from http://www.voidspace.org.uk/python/articles/comparison.shtml
"""

from __future__ import print_function, division, absolute_import

from flypy import jit

@jit
class RichComparisonMixin(object):

    layout = []

    @jit
    def __eq__(self, other):
        raise NotImplementedError("Equality not implemented")

    @jit
    def __lt__(self, other):
        raise NotImplementedError("Less than not implemented")

    @jit
    def __ne__(self, other):
        return not self.__eq__(other)

    @jit
    def __gt__(self, other):
        return not (self.__lt__(other) or self.__eq__(other))

    @jit
    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    @jit
    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)