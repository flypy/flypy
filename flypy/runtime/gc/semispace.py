# -*- coding: utf-8 -*-

"""
Semi-space copying collector.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from flypy import jit, sjit, cast, sizeof, NULL
from flypy.runtime.obj.core import newbuffer, Buffer, Pointer, head, tail
from flypy.lib.bitvector import BitVector
from flypy.types import void, int8, int64, uint64
from . import roots

ptr_size  = ctypes.sizeof(ctypes.c_void_p)

# live data / heap size ratio threshold to increase heap size
# TODO: Research good parameters
OCCUPATION_THRESHOLD_LO = 0.25
OCCUPATION_THRESHOLD_HI = 0.7

# how much to grow the heap when OCCUPATION_THRESHOLD is exceeded
GROW_FACTOR = 2
SHRINK_FACTOR = 2

@sjit
class BumpAllocator(object):
    """
    Simple bump-allocator. Bumps a pointer forward until it runs out of memory.

    We regard the heap as growing upwards from the bottom to the top.
    """
    layout = [
        ('heap',   'Buffer[int8]'),
        ('offset', 'Pointer[int8]'),
        ('_top',   'Pointer[int8]'),
    ]

    @jit
    def __init__(self, initial_size=1024**2):
        self.heap = newbuffer(int8, initial_size)
        self.offset = self.bottom()
        self._top = self.bottom() + initial_size

    @jit('allocator -> size -> Pointer[int8]')
    def alloc(self, size):
        obj = self.offset
        new_offset = align_pointer(obj + size, 8)
        if new_offset > self.top():
            return cast(0, Pointer[int8])

        self.offset = new_offset
        return obj

    @jit
    def reset(self):
        self.offset = self.bottom()

    @jit
    def size(self):
        return len(self.heap)

    @jit
    def resize(self, new_size):
        #assert (self.offset - self.bottom()) <= new_size
        self.heap.resize(new_size)
        self.reset()

    @jit
    def bottom(self):
        return self.heap.pointer()

    @jit
    def top(self):
        return self._top

    @jit
    def occupation(self):
        return occupation(self, self.offset)


@jit
def occupation(allocator, offset):
    total = allocator.top() - allocator.bottom()
    occupied = offset - allocator.bottom()
    return total / float(occupied)


@sjit
class GC(object):
    """
    Simple semi-space copying GC

        fromspace:
            heap we are currently allocating objects in
        tospace:
            heap to copy live data to when fromspace is full
        forwarding_table:
            vector of bits, indicating for each pointer-sized word whether
            it contains a forwarding pointer. See the 'copy' method for
            an additional explanation
    """

    layout = [
        ('fromspace', 'BumpAllocator[]'),
        ('tospace',   'BumpAllocator[]'),
        ('forwarding_table', 'BitVector[]'),
    ]

    @jit
    def __init__(self, initial_size=1024**2):
        self.fromspace = BumpAllocator(initial_size)
        self.tospace = BumpAllocator(initial_size)
        self.init_forwarding_table()
        #self.finalizers = {}

    @jit
    def init_forwarding_table(self):
        """Initialize an empty forwarding table for the heap"""
        self.forwarding_table = BitVector(self.fromspace.size())

    @jit('gc -> Pointer[int8] -> int64')
    def offset(self, obj):
        """Offset of object in heap, used for marking forwarded objects"""
        return (obj - self.fromspace.bottom()) >> ptr_size

    @jit('gc -> Pointer[int8] -> uint64')
    def forwarded(self, obj):
        """Whether this object was copied to tospace"""
        return self.offset(obj) in self.forwarding_table

    @jit('allocator -> size -> Pointer[int8]')
    def alloc(self, size, top_of_stack):
        obj = self.fromspace.alloc(size)
        if obj == NULL:
            # fromspace is full, collect garbage
            self.collect(size, top_of_stack)
            obj = self.fromspace.alloc(size)
            # assert obj != NULL # MemoryError should have been raised by
            #                    # collect, unless `obj` really is too large

        return obj

    @jit('gc -> int64 -> Pointer[StackFrame[]] -> void')
    def collect(self, size, top_of_stack):
        """
        Collect garbage by copying all live data to tospace, starting with
        `roots`. If there is no or little garbage and we're running out of heap
        space, grow the heap. `size` indicates the size of the requested memory.

        This method may raise MemoryError.

        For every root, we determine the trace function and call it on the root.
        The trace function traces the object itself (since only it knows its
        size), and traces its children:

            def __flypy_trace__(self, gc):
                obj = gc.trace((int8 *) self, sizeof(self))
                obj.field1 = obj.field1.trace(gc)
                ...
                obj.fieldN = obj.fieldN.trace(gc)
                return obj
        """
        # Copy all live data
        for item in roots.find_roots(top_of_stack):
            obj = head(item)
            trace = head(tail(item))
            trace(obj)

        # Swap spaces and adjust heap if necessary
        self.fromspace, self.tospace = self.tospace, self.fromspace
        if self.resize_heap(size):
            self.init_forwarding_table()
        else:
            self.forwarding_table.clear()

    @jit
    def resize_heap(self, mininum_size):
        """
        Adjust heap size based on occupation ratio.
        Returns True if the heap got resized.

        NOTE: This may raise MemoryError
        """
        ratio = occupation(self.fromspace, self.fromspace.offset + mininum_size)
        size = self.fromspace.size() + mininum_size
        if ratio > OCCUPATION_THRESHOLD_HI:
            self.fromspace.resize(int(size * GROW_FACTOR))
        elif ratio < OCCUPATION_THRESHOLD_LO:
            self.fromspace.resize(int(size / GROW_FACTOR))
        else:
            return False

        return True

    @jit('gc -> Pointer[int8] -> int64 -> Pointer[int8]')
    def trace(self, obj, size, trace_children):
        """
        Mark the reachable object.  We first check if the object is already
        copied, in which case we're done. Otherwise, we copy the object to the
        tospace, and we trace the children. The copy operation will mark
        the object as copied in the forwarding table.

        We call __flypy_trace__ to mark GC-tracked children. This method
        is automatically generated, and allows us to avoid type tagging values
        to determine pointer locations. It also patches pointers to copied
        objects.
        """
        #assert self.fromspace.bottom() <= obj < self.fromspace.top

        if self.forwarded(obj):
            # Object already copied, use forwarding pointer
            forward_ptr_ptr = cast(obj, Pointer[Pointer[int8]])
            dst_obj = forward_ptr_ptr[0]
            return dst_obj

        copied_obj = self.copy(obj, size)
        trace_children(copied_obj)

        return copied_obj

    @jit('gc -> Pointer[int8] -> int64 -> Pointer[int8]')
    def copy(self, obj, size):
        """
        Copy the object (excluding its children!) to the tospace.

        This writes a forwarding pointer in the fromspace after having
        copied the object, and remembers doing so by writing a bit in the
        forwarding table corresponding to the offset of the object.

        We could determine the forwarding address for any object that has
        pointers using the type, for instance by overwriting the first pointer
        in the object with the new pointer, and checking in which address space
        that pointer points.

        However, we support tagless representations, and we may not have any
        reference, like a mutable value with only integers. We cannot
        safely determine whether an overwritten location is a forwarding
        address, or actually a value that happens to fall in the range
        of the to-space.
        """
        dst_obj = self.tospace.alloc(size, type)
        memcpy(obj, dst_obj, size)

        forward_ptr_ptr = cast(obj, Pointer[Pointer[int8]])
        forward_ptr_ptr[0] = dst_obj
        self.forwarding_table.mark(self.offset(obj))

        return dst_obj



@jit('Pointer[int8] -> int64 -> Pointer[int8]')
def align_pointer(p, alignment):
    "Align pointer memory on a given boundary"
    i = p.ptrtoint()
    offset = i % alignment
    if offset > 0:
        i += alignment - offset
    return cast(i, Pointer[int8])


@jit('Pointer[void] -> Pointer[void] -> int64 -> void')
def memcpy(src, dst, size):
    """
    Copy `size` bytes from `src` to `dst`

    Note that the memory of src and dst may not overlap!
    """
    src = cast(src, Pointer[int8])
    dst = cast(dst, Pointer[int8])
    for i in range(size):
        dst[i] = src[i]
