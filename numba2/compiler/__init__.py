# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from .layout import representation_type
lltype = representation_type
from .annotations import annotate
from .overloading import overload, overloadable
from .callingconv import is_numba_cc