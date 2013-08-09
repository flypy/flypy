# -*- coding: utf-8 -*-

"""
Contains all numba errors.
"""

from __future__ import print_function, division, absolute_import

class error(Exception):
    """Superclass of all numba exceptions"""

class InferError(error):
    """Raised for type inference errors"""

class SpecializeError(error):
    """
    Raised when we fail to specialize on something requested for some reason.
    """
