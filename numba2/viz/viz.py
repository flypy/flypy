# -*- coding: utf-8 -*-

"""
Some graphviz support.
"""

from __future__ import print_function, division, absolute_import

import os
import logging
import subprocess

import networkx as nx

logger = logging.getLogger(__name__)

#===------------------------------------------------------------------===
# Create image from dot
#===------------------------------------------------------------------===

def dump(G, dotfile):
    path = os.path.expanduser(dotfile)
    nx.write_dot(G, path)
    write_image(path)

def write_image(dot_output):
    prefix, ext = os.path.splitext(dot_output)
    png_output = prefix + '.png'

    fp = open(png_output, 'wb')
    try:
        p = subprocess.Popen(['dot', '-Tpng', dot_output],
                             stdout=fp.fileno(),
                             stderr=subprocess.PIPE)
        p.wait()
    except EnvironmentError as e:
        logger.warn("Unable to write png: %s (did you install the "
                    "'dot' program?). Wrote %s" % (e, dot_output))
    else:
        logger.info("Wrote %s" % png_output)
    finally:
        fp.close()
