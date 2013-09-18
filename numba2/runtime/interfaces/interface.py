# -*- coding: utf-8 -*-

"""
Entry points for runtime code.
"""

from __future__ import print_function, division, absolute_import
import inspect
from ...function import Function

def verify_interface(cls, interface):
    """Verify that all abstract methods are implemented"""
    for attr, method in inspect.getmembers(interface):
        if isinstance(method, Function):
            if method.abstract:
                # Check for method implementation
                if not hasattr(cls, attr):
                    raise TypeError(
                        "Class %s does not implement method %s required "
                        "by %s" % (cls, attr, method))

def interface_compatibility(interfaces):
    """Verify that all the given interfaces are compatible with each other"""
    attributes = {}
    for interface in interfaces:
        for attr, method in inspect.getmembers(interface):
            if isinstance(method, Function):
                # TODO: Improve this check
                if attr in attributes and method != attributes[attr]:
                    raise TypeError(
                        "Cannot use incompatible interfaces for method "
                        "%s" % method)
                else:
                    attributes[attr] = method

def copy_methods(cls, interface):
    """Copy implemented methods from the interface"""
    for attr, method in inspect.getmembers(interface):
        if isinstance(method, Function):
            if not method.abstract and not hasattr(cls, attr):
                # Copy method if not present
                setattr(cls, attr, method)