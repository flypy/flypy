#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from os.path import abspath, dirname, join
import sys

import flypy

kwds = {}
if len(sys.argv) > 1:
    kwds["pattern"] = '*' + sys.argv[1] + '*'

if 0:
    root = dirname(abspath(flypy.__file__))
    order = ['frontend', 'compiler', 'runtime']
    dirs = [join(root, pkg, 'tests') for pkg in order]
    sys.exit(flypy.run_tests(dirs, **kwds))
else:
    sys.exit(flypy.test())