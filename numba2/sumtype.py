"""
Implement sum types.

Required compiler support:

    1) Assignment -> __assign__ where implemented
    2) unfify() function to unify types at control flow merges
    3) Exception analysis to see what statements can raise (split CFG blocks)
"""

from core import structclass, Struct, signature, char, void

@cached
def sumtype(type1, type2):
    @structclass('SumType[type T1, type T2]')
    class SumType(object):

        layout = Struct([('tag', bool),
                         ('obj1', 'T1'),
                         ('obj2', 'T2')])

        def __init__(self):
            self.tag = False

        @overload('T1 -> Void')
        def __assign__(self, obj):
            self.tag = False
            self.obj1 = obj

        @overload('T2 -> Void')
        def __assign__(self, obj):
            self.tag = True
            self.obj2 = obj

        if '__add__' in type1.methods and '__add__' in type2.methods:
            @signature # type infer result type
            def __add__(self, other):
                if self.tag:
                    return self.obj1 + other
                else:
                    return self.obj2 + other

    return SumType

def unify(a, b):
    return sumtype(a, b)