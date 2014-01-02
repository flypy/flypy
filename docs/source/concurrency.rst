Concurrency
===========

Support for a shared-nothing environment with typed channels for communication
between tasks. Immutable data may be copied or may be wrapped in an atomically
refcounted structure are shared between threads. Mutable data may be sent if
wrapped in a thread-safe implementation, based on STM or fine grained locking.

Array data is especially important to avoid copies for. We can implement a
method to obtain slices which own a sub-region of the array, and which can be
safely used. Arrays then have a notion of "locked" and "unlocked" sub-regions.
Owned slices may be sent over a channel to become a slice which can be
used locally in another thread:

.. code-block:: python

    # -- Thread 1 -- #
    owned_slice = Array.owned[:, :N/2]

    # Array now has a locked sub-region and we cannot perform read/write
    # operations on Array any longer

    # We cannot read or write to `owned_slice` either, we can only `send` it:
    chan.send(owned_slice)
    # `owned_slice` cannot be used for further sends over any channel

    # -- Thread 2 -- #
    slice = chan.recv()

    # We can now safely use `slice`
    item = slice[0, 0]

    # when `slice` goes out of scope, we atomically unlock the owned region

Owned Mutable Data
------------------
[ crazy ideas below, not convinced this is a good idea ]

For owned mutable data users can rely on thread-safe implementations, but that
may be wastful in terms of locking or transactional overhead. We can support
a global exchange heap, similar to rust. Objects allocated in this manner are
always reference counted. These objects may be sent over a channel iff their
reference count and the reference count of any composed owned mutable values
are exactly one. In case of an error a garbage collection may need to be
triggered to determine whether there were any lingering dead references.

To avoid reference counting cycles, only tree-structures are allowed, such
that no cycle may exist between objects. This means we do not have to
implement a collector for cyclic references.