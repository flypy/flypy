# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from numba2.compiler import excmodel
from numba2.runtime.obj.core import Type
from numba2.runtime.obj.exceptions import Exception, StopIteration

#===------------------------------------------------------------------===
# Tests
#===------------------------------------------------------------------===

class TestExcModel(unittest.TestCase):

    def test_excmodel(self):
        assert excmodel.exc_match(Exception[()], StopIteration[()])
        assert not excmodel.exc_match(StopIteration[()], Exception[()])

        assert excmodel.exc_match(Type[Exception[()]], Type[StopIteration[()]])
        assert not excmodel.exc_match(Type[StopIteration[()]], Type[Exception[()]])


if __name__ == '__main__':
    unittest.main()