# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

class MetaType(type):
    """Type of a type"""

    def __getitem__(cls, *args):
        assert len(args) == len(cls.parameters)


# @jit('Type[t]')
class Type(object):
    """
    Type object interface. Type objects are runtime values parameterized
    by themselves, allowing overloading.

    Instances can be indexed from python to obtain concrete types from
    parameterized types.
    """

    __metaclass__ = MetaType
    layout = {'name': 'String', 'parameters': '(Type, ...)'}

    def __init__(self, name, *parameters):
        self.name = name
        self.parameters = parameters
