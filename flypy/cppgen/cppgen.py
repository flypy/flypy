# -*- coding: utf-8 -*-

"""
Generate C++ classes from flypy classes that can produce flypy-compatible
objects.
"""

from __future__ import print_function, division, absolute_import

import inspect
import textwrap

from pykit.utils import make_temper

from flypy import jit

reserved = [
    "this", "template", "struct", "class", "union", # TODO: others
]

def format_type(type):
    return str(type)

def generate(cls, emit=print):
    if not hasattr(cls, 'layout'):
        raise TypeError("Class has to attribute 'layout', is this a flypy type?")

    ctor = cls.__init__.py_func
    argspec = inspect.getargspec(ctor)

    if argspec.varargs or argspec.keywords or argspec.defaults:
        raise TypeError(
            "Varargs, keyword arguments or default values not supported")

    temp = make_temper()
    for keyword in reserved:
        temp(keyword)

    classname = cls.__name__
    attrs = []
    params = []
    typevars = [tvar.symbol for tvar in cls.type.parameters]
    templates = ["typename %s" % (tvar,) for tvar in typevars]
    argnames = [temp(argname) for argname in argspec.args[1:]]
    initializers = []

    for attr, type in cls.type.resolved_layout.items():
        params.append("%s %s" % (format_type(type), attr))
        attrs.append("%s %s;" % (format_type(type), attr))
    for arg in argnames:
        initializers.append("this._data.%s = %s;" % (arg, arg))

    emit(textwrap.dedent("""
    template<%(typevars)s>
    class %(classname)s {
        struct {
    %(attrs)s
        } _data;

      public:
        %(classname)s(%(params)s) {
    %(initializers)s
        }
    };
    """) % {
        'typevars':     ", ".join(templates),
        'classname':    classname,
        'attrs':        "\n".join("        " + a for a in attrs),
        'params':       ", ".join(params),
        'initializers': "\n".join("        " + i for i in initializers),
    })
