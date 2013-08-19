# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from functools import partial
from .overload import overload, overloadable
from pykit.utils import make_temper

# ______________________________________________________________________

_temp = make_temper()
typevar_names = u'αβγδεζηθικλμνξοπρςστυφχψω'

def make_stream(seq=typevar_names):
    for x in seq:
        yield _temp(x)

gensym = make_stream(typevar_names).next

# ______________________________________________________________________

class Typevar(object):
    """Type variable"""

    def __init__(self, name=""):
        if name:
            name = _temp(name)
        else:
            name = gensym()
        self.name = name

    def __repr__(self):
        return ('T(%s)' % (unicode(self),)).encode('utf-8')

    def __unicode__(self):
        return self.name


class Constraint(object):
    """Typing constraint"""

    def __init__(self, pos, op, args):
        self.pos  = pos
        self.op   = op
        self.args = args

    def __repr__(self):
        if len(self.args) == 2:
            return "%s %s %s" % (self.args[0], self.op, self.args[1])
        return "%s(%s)" % (self.op, ", ".join(map(str, self.args)))


def free(t, freevars=None, seen=None):
    """Find free variables in a Type"""
    if freevars is None:
        freevars = []
        seen = set()

    if t in seen:
        pass
    elif isinstance(t, Typevar):
        freevars.append(t)
    elif isinstance(t, Type):
        seen.add(t)
        for param in t.params:
            free(param, freevars, seen)

    return freevars

# ______________________________________________________________________

def stringify(type, level=0):
    if isinstance(type, Typevar):
        return type.name
    elif not isinstance(type, Type):
        return unicode(type)

    params = [stringify(p, level + 1) for p in type.params]
    if type.name == 'Function':
        result = u" -> ".join(params[1:] + [params[0]])
        result = u"(%s)" % result
    elif type.name == 'Sum':
        result = u"{%s}" % u",".join(params)
    else:
        result = u"%s(%s)" % (type.name, u", ".join(params))

    fv = free(type)
    if level == 0 and fv:
        quantification = u"∀%s." % ",".join(var.name for var in fv)
        return quantification + result
    else:
        return result

class Type(object):
    """
    Simple parameterizable type.
    """

    def __init__(self, name, *params):
        self.name = name
        self.params = params
        self.fields = {}

    def __getitem__(self, i):
        return self.params[i]

    def __len__(self):
        return len(self.params)

    def __eq__(self, other):
        return (isinstance(other, Type) and self.name == other.name and
                self.params == other.params)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.name, self.params))

    __unicode__ = stringify

    def __repr__(self):
        return unicode(self).encode('UTF-8')


Void     = partial(Type, 'Void')
Bool     = partial(Type, 'Bool')
Function = partial(Type, 'Function')
Product  = partial(Type, 'Product')
Opaque   = partial(Type, 'Opaque')
Pointer  = partial(Type, 'Pointer')
Method   = partial(Type, 'Method')

def Sum(args):
    if len(args) == 1:
        return args[0]

    types = []
    for arg in args:
        if isinstance(arg, Type) and arg.name == 'Sum':
            types.extend(arg.params)
        else:
            types.append(arg)

    return Type('Sum', *list(set(types)))

def substitute(s_context, ty):
    if isinstance(ty, Typevar):
        return s_context.get(ty, ty)
    elif not isinstance(ty, Type):
        return ty
    else:
        return Type(ty.name, *tuple(substitute(s_context, p) for p in ty.params))

# ______________________________________________________________________

@overloadable
def typeof(pyval):
    """Python value -> Type"""

@overload('ν -> Type[τ] -> τ')
def convert(value, type):
    """Convert a value of type 'a' to the given type"""
    return value

@overload('Type[α] -> Type[β] -> Type[γ]')
def promote(type1, type2):
    """Promote two types to a common type"""
    return Sum([type1, type2])

class TypedefRegistry(object):
    def __init__(self):
        self.typedefs = {} # builtin -> numba function

    def typedef(self, pyfunc, numbafunc):
        assert pyfunc not in self.typedefs
        self.typedefs[pyfunc] = numbafunc


typedef_registry = TypedefRegistry()
typedef = typedef_registry.typedef

T, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10 = [
        Typevar(typevar_names[i]) for i in range(12)]