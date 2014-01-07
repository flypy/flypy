# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import os
import sys
import shutil
import subprocess
from fnmatch import fnmatchcase
from distutils.util import convert_path

# Do not EVER use setuptools, it makes cythonization fail
# Distribute fixes that
from distutils.core import setup, Extension

import numpy

from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension as CythonExtension

if sys.version_info[:2] < (2, 6):
    raise Exception('numba requires Python 2.6 or greater.')

import versioneer

#------------------------------------------------------------------------
# Setup constants and arguments
#------------------------------------------------------------------------

versioneer.versionfile_source = 'flypy/_version.py'
versioneer.versionfile_build = 'flypy/_version.py'
versioneer.tag_prefix = ''
versioneer.parentdir_prefix = 'flypy-'

cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext

setup_args = {
    'long_description': open('README.md').read(),
}

numba_root = os.path.dirname(os.path.abspath(__file__))

#------------------------------------------------------------------------
# Package finding
#------------------------------------------------------------------------

def find_packages(where='.', exclude=()):
    out = []
    stack=[(convert_path(where), '')]
    while stack:
        where, prefix = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where,name)
            if ('.' not in name and os.path.isdir(fn) and
                os.path.isfile(os.path.join(fn, '__init__.py'))
            ):
                out.append(prefix+name)
                stack.append((fn, prefix+name+'.'))

    if sys.version_info[0] == 3:
        exclude = exclude + ('*py2only*', )

    for pat in list(exclude) + ['ez_setup', 'distribute_setup']:
        out = [item for item in out if not fnmatchcase(item, pat)]

    return out

#------------------------------------------------------------------------
# 2to3
#------------------------------------------------------------------------

def run_2to3():
    import lib2to3.refactor
    from distutils.command.build_py import build_py_2to3 as build_py
    print("Installing 2to3 fixers")
    # need to convert sources to Py3 on installation
    fixes = 'dict imports imports2 unicode metaclass basestring reduce ' \
            'xrange itertools itertools_imports long types exec execfile'.split()
    fixes = ['lib2to3.fixes.fix_' + fix 
             for fix in fixes]
    build_py.fixer_names = fixes
    cmdclass["build_py"] = build_py
    # cmdclass["build"] = build_py

    # Distribute options
    # setup_args["use_2to3"] = True

#------------------------------------------------------------------------
# setup
#------------------------------------------------------------------------

exclude_packages = (

)

setup(
    name="numba",
    version=versioneer.get_version(),
    author="Continuum Analytics, Inc.",
    author_email="numba-users@continuum.io",
    url="http://numba.github.com",
    license="BSD",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        # "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        # "Programming Language :: Python :: 3.2",
        "Topic :: Utilities",
    ],
    description="Compiling Python code using LLVM",
    packages=find_packages(exclude=exclude_packages),
    #entry_points = {
    #    'console_scripts': [
    #        'pycc = numba.pycc:main',
    #        ]
    #},
    scripts=["bin/numba"],
    package_data={
        '': ['*.md'],
        'flypy.runtime.obj': ['*.c', '*.h', '*.pyx', '*.pxd'],
    },
    ext_modules=[
        Extension(
            name="flypy.runtime.lib.libcpy",
            sources=["flypy/runtime/lib/libcpy.pyx"],
            include_dirs=[numpy.get_include()],
            depends=[]),
        Extension(
            name="flypy.runtime.gc.boehmlib",
            sources=["flypy/runtime/gc/boehmlib.pyx"],
            libraries=["gc"]),
    ],
    cmdclass=cmdclass,
    **setup_args
)
