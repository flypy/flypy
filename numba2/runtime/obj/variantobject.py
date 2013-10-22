# -*- coding: utf-8 -*-

"""
Variant implementation, a | b.
"""

from __future__ import print_function, division, absolute_import
from ... import jit, ijit, typeof

# TODO: *args, **kwargs

def make_variant(left, right):
    class Variant(object):
        # TODO: Implement unions
        layout = [('tag', 'int32'), ('left', 'a'), ('right', 'b')]

    methodnames = set(left.fields) & set(right.fields)

    for name in methodnames:
        setattr(Variant, name, make_variant_method(name, left, right))

    return jit('Variant[a, b]')(Variant)


def make_variant_method(methname, left, right):
    left = left.fields[methname]
    right = right.fields[methname]

    @ijit
    def method(self, *args, **kwargs):
        if self.tag == 0:
            return left(self.left, *args, **kwargs)
        else:
            return right(self.right, *args, **kwargs)

    return method