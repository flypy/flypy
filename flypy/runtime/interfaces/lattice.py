# -*- coding: utf-8 -*-

"""
Core flypy interfaces.
"""

from __future__ import print_function, division, absolute_import
from .. import abstract

@abstract
class Top(object):
    """Join"""

@abstract
class Bottom(object):
    """Meet"""