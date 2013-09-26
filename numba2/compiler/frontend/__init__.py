# -*- coding: utf-8 -*-

"""
Frontend package providing translation from bytecode -> untyped pykit IR.
"""

from .frontend import translate, simplify_exceptions
from .interp import run as interpret