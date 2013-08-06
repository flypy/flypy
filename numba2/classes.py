"""
Implement classes.

Required compiler support:

    1) Proper MSP to quickly build wrapper. Alternatively we can use strings...

Inspiration:

    https://github.com/zdevito/terra/blob/master/tests/lib/javalike.t
    https://github.com/zdevito/terra/blob/master/tests/lib/golike.t
"""

from core import structclass, Struct, signature, char, void

@cached
def classjit(cls):
    if hasattr(cls, 'layout'):
        layout = cls.layout
    else:
        symtab = type_infer(cls.__init__)
        order = infer_assignment_order(cls.__init__)
        layout = Struct([(varname, symtab[varname]) for varname in order])

    return newclass(layout)

@cached
def newclass(cls, layout):
    """New @jit class"""
    vtable = build_virtual_method_table(cls)
    layout.insert_field(0, ("vtable", vtable))

    if hasattr(cls, '__numba_type__'):
        # Subclass of numba type
        class_type = cls.__numba_type__
        assert class_type.layout.is_prefix(layout)

    @structclass('ClassType[Object cls]')
    class ClassType(object):
        layout = layout

    # Insert proxy methods that call into vtable
    for name, signature in vtable.fields:
        argnames = signature.argnames
        with quote as body:
            def wrapper(escape[argnames]):
                return escape[argnames[0]].vtable.[name]([argnames])

        wrapper = body.body[0]
        wrapper.signature = signature
        ClassType.add_method(name, wrapper)

    return ClassType[cls]
