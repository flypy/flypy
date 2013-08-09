# -*- coding: utf-8 -*-

"""
Annotations set during import type and processed by the compiler.
"""

from __future__ import print_function, division, absolute_import

annotations = {
    'specialize_value': tuple,  # ('arg1', 'arg2', ...)
    'type_constraints': tuple,  # (Constraint, ...)
    'type_signature': object,   # Type of the object
    'type_trait': bool,         # Whether the type is a Trait
}

def annotate(obj, **kwds):
    """Set an annotation and verify validity"""
    for name, value in kwds.iteritems():
        if name not in annotations:
            raise ValueError("Not a known annotation: %s" % (name,))
        elif not isinstance(value, annotations[name]):
            raise TypeError(
                "Expected an instance of %s for annotation %s, got %s" % (
                                            annotations[name], name, name))

        if not hasattr(obj, '__numba_annotations__'):
            obj.__numba_annotations__ = {}

        obj.__numba_annotations__[name] = value

def get(obj, name):
    """Get an annotation from obj"""
    return obj.__numba_annotations__.get(name)