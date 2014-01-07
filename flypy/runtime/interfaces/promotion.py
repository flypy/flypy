# -*- coding: utf-8 -*-

"""
Promotion trait for special methods (binary operators).
"""

from __future__ import print_function, division, absolute_import

from ... import promote, typeof, overload, convert
from .. import abstract


@abstract
class Promotion(object):
    """Trait implementing promotion for binary operations"""

    @overload('α -> β -> γ', inline=True)
    def __add__(self, other):
        return other.__radd__(self)

    @overload('α -> α -> β')
    def __add__(self, other):
        # Guard against infinite recursion
        raise NotImplementedError("__add__")

    @overload('α -> β -> γ', inline=True)
    def __radd__(self, other):
        T = promote(typeof(self), typeof(other))
        return convert(other, T) + convert(self, T)
