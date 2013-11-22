# -*- coding: utf-8 -*-

"""
Struct object implementation.
"""

from __future__ import print_function, division, absolute_import
import string

from ... import sjit

def struct_(fields, name=None, packed=False):
    if (not isinstance(fields, list) or not fields or not
            isinstance(fields[0], tuple) or not len(fields[0]) == 2):
        raise TypeError("Excepted a list of two-tuples, got %s" % (fields,))

    assert not packed

    fieldnames, fieldtypes = zip(*fields)
    typevars = string.ascii_letters[:len(fieldnames)]

    @sjit('Struct[%s]' % ", ".join(typevars))
    class Struct(object):
        layout = list(zip(fieldnames, typevars))

    if name:
        Struct.__name__ = name

    return Struct[tuple(fieldtypes)]