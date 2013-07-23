"""
Implement range().
"""

from core import structclass, Struct, signature, char, void, typedef

layout = Struct([('start', 'Int'),
                 ('stop', 'Int'),
                 ('step', 'Int')])


class RangeIter(object):
    layout = Struct([('start', 'Int'),
                     ('step', 'Int'),
                     ('len', 'Int')])

    def __init__(self, start, stop, step):
        self.start = start
        self.step = step
        self.len = (stop - start) / step

    @signature('RangeIter -> RangeIter')
    def __iter__(self):
        return self

    @signature('RangeIter -> Int')
    def __next__(self):
        if self.len == 0:
            raise StopIteration
        self.len -= 1
        self.start += self.step
        return self.start


@structclass
class Range(object):

    layout = Struct([('start', 'Int'),
                     ('stop', 'Int'),
                     ('step', 'Int')])

    @signature('Range -> RangeIter')
    def __iter__(self):
        return RangeIter(self.start)


@signature('Int -> Int -> Int -> Range')
def range_(start, stop=None, step=None):
    if stop is None:
        stop = start
        start = 0
    if step is None:
        step = 1
    return Range(start, stop, step)

typedef(range, range_)