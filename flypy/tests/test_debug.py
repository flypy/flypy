# -*- coding: utf-8 -*-

"""
Test that the debug flag is disabled, especially for releases.
"""

from __future__ import print_function, division, absolute_import
import unittest
from flypy.config import config

class TestDebugFlag(unittest.TestCase):

    def test_debug_flag(self):
        self.assertFalse(config.debug)

