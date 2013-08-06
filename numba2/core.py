import inspect

class Type(object):
    def pointer(self):
        return pointer(self)

    def __call__(self, *args):
        return self # ... not complete

char = Type()
pointer = Type()
void = Type()
Py_ssize_t = Type()
Rep = Type() # Meta-type, represents staged code

def signature(sigstr):
    def dec(f):
        return f
    return dec

class Struct(Type):
    def __init__(self, fields):
        self.fields = fields
        self.methods = {}

    def add_method(self, name, method):
        self.methods[name] = method

def structclass(cls):
    cls.__numba_struct__ = Struct(cls.layout)
    for name, method in inspect.getmembers(cls, inspect.ismethod):
        cls.__numba_struct__.add_method(name, method)

    return cls

