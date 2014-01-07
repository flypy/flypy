# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from flypy import jit, typeof
from flypy.runtime.obj import exceptions

class TestExceptionObjs(unittest.TestCase):

    def test_typeof(self):
        self.assertEqual(typeof(StopIteration()), exceptions.StopIteration.type)

if __name__ == '__main__':
    #TestExceptionObjs('test_typeof').debug()
    unittest.main()