# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

class Type(type):
    """
    Base of all types. Types are instances of Type, and values are instances
    of types.
    """

    # __metaclass__ = abc.ABCMeta

    def __getitem__(cls, *args):
        assert len(args) == len(cls.parameters)


def jit_class(cls, signature=None):
    """
    @jit('Array[dtype, ndim : Int]')
    """
    if not hasattr(cls, 'layout'):
        raise ValueError("layout of class %s not set" % (cls,))

    assert not hasattr(cls, 'parameters')
    assert not hasattr(cls, 'type')

    # Type.register(cls)

    if signature is not None:
        type = parse_type(signature)
        cls.parameters = free(type)
        cls.layout = substitute(cls.layout, )
    else:
        cls.parameters = ()
        assert not free(cls.layout)

    return Type(cls.__name__, cls.__bases__, vars(cls))


