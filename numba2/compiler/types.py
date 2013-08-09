# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from functools import partial

class Type(object):
    """
    Simple parameterizable type.
    """

    def __init__(self, name, *parameters):
        self.name = name
        self.parameters = parameters

Function = partial(Type, 'Function')