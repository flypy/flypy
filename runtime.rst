Numba Runtime
=============
Nearly all built-in data types are implemented in the runtime.

Garbage Collector
=================
To support mutable heap-allocated types, we need a garbage collector.
To get started quickly we can use Boehm or reference counting. We will
want to port one of the available copying collectors and use a shadowstack or
a lazy pointer stack (for bonus points). The GC should then be local to
each thread, since there is no shared state between threads (only owned
and borrowed data is allowed).

Garbage collection is abstracted by pykit.

Exceptions
==========
Exceptions are also handled by pykit. We can implement several models,
depending on the target architecture:

    * costful (error return codes)
        - This will be used on the GPU
    * zero-cost
        - This should be used where supported. We will start with costful
    * setjmp/longjmp
        - This will need to happen for every stack frame in case of a shadow stack

Local exception handling will be translated to jumps. This is not contrived,
since we intend to make heavy use of inlining:

.. code-block:: python

    while 1:
        try:
            i = x.__next__()
        except StopIteration:
            break

``x.__next__()`` may be inlined (and will be in many instances, like range()),
and the ``raise StopIteration`` will be translated to a jump. Control flow
simplification can further optimize the extra jump (jump to break, break to
loop exit).

Threads
=======
As mentioned in the core language overview, memory is not shared unless
borrowed. This process is unsafe and correctness must be ensured by the
user. Immutable data can be copied over channels between threads. Due to
a thread-local GC, all threads can run at the same time and allocate memory
at the same time.

We will remove prange and simply use a parallel map with a closure.

Traits
======
Traits are mostly a compile-time type-checking detail and some simple runtime
decorator support. Traits with dynamic dispatch require vtables, something
we can implement in the runtime as well:

    https://github.com/zdevito/terra/blob/master/tests/lib/golike.t

Extension Types
===============
Extension types are currently built on top of CPython objects. This should
be avoided. We need to decouple numba with anything CPython, for the sake
of portability as well as pycc.

Extension types can also easily be written in the runtime:

    - ``unify()`` needs to return the supertype or raise a type error
    - ``convert(obj, Object)`` needs to do a runtime typecheck
    - ``coerce_distance`` needs to return a distance for how far the supertype
      is up the inheritance tree

The approach is simple: generate a wrapper method for each method in the
extension type that does a vtable lookup.

Closures
========
This time we will start with the most common case: closures consumed as
inner functions. This means we don't need dynamic binding for our cell
variables, and we can do simple lambda lifting instead of complicated
closure conversion. This also trivially works on the GPU, allowing one
to use map, filter etc, with lambdas trivially.

