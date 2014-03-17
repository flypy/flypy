"""
Adapted from flypyPro
"""
import sys
import os
import ctypes

def get_conda_lib_dir():
    dirname = 'DLLs' if sys.platform == 'win32' else 'lib'
    libdir = os.path.join(sys.prefix, dirname)
    return libdir


DLLNAMEMAP = {
    'linux2': ('lib', '.so'),
    'linux':  ('lib', '.so'),
    'darwin': ('lib', '.dylib'),
    'win32' : ('',    '.dll')
}

def find_lib(libname, env=None):
    """Find library by name.
    Default search path includes pwd, system lib paths and conda lib path.

    :param libname: library name (don't include prefix "lib" or file
                                  extension)
    :param env: overriding environment variable.
    """
    prefix, ext = DLLNAMEMAP[sys.platform]
    fullname = '%s%s%s' % (prefix, libname, ext)
    candidates = [
        os.path.join(get_conda_lib_dir(), fullname),
        fullname,
        ]

    # Check overriding environment variable
    if env:
        envpath = os.environ.get(env, '')
    else:
        envpath = None

    if envpath:
        if os.path.isdir(envpath):
            envpath = os.path.join(envpath, fullname)
        return [envpath]
    else:
        return candidates

def open_lib_ctypes(paths, constructor=ctypes.CDLL):
    for i, path in enumerate(paths):
        try:
            return constructor(path)
        except OSError:
            if i == len(paths) - 1:
                raise
