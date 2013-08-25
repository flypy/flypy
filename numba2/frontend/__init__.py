# -*- coding: utf-8 -*-

"""
Frontend package providing translation from bytecode -> untyped pykit IR.
"""

from .translation import translate
from .interp import run as interpret