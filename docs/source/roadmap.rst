Roadmap
=======

Numba roadmap. This partitions implementations into things that need
to be implemented in the compiler, and things that need to be implemented
in the runtime.

Runtime implementations should always be preferred, so if things can be shifted
from implementation in the compiler to the runtime, that should probably be
done.

Compiler
--------

See ``numba/compiler``. Items are listed in order of urgency.

Local Exception Rewriting
~~~~~~~~~~~~~~~~~~~~~~~~~
Rewrite 'try/except' with a 'raise' in the 'try' to a jump.

Constant Specialization
~~~~~~~~~~~~~~~~~~~~~~~
Accepting a constant as function argument and specializing for it.

Constant Folding and Propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This should include ``isinstance``.

Unrolling
~~~~~~~~~
Unroll static iterables. Very important for partial evaluation, and for
optimizations such as efficient array indexing.

Inference <-> Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~
Use type inference to enable optimizations, and optimizations to enable
type inference. For instance, type inference can resolve ``isinstance``,
which can prune branches which would otherwise generate superfluous types
that are merged at subsequent join points, making the types too general.

Type Lattice
~~~~~~~~~~~~
Implement a type join that finds the most general type between two types.

Generator Fusion
~~~~~~~~~~~~~~~~
Fusion of generators into consumers when there is "static control flow" between
the consumer and producer.

Escape Analysis & Stack Allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Not urgent, only an optimization.

Runtime
-------

See ``numba/runtime``. Items are listed in order of urgency.

Pointers
~~~~~~~~
Simple addition and subtraction.

Conversions
~~~~~~~~~~~
Conversion mechanism, that invokes the overloaded ``convert`` function.
Generate a simple Python function:

.. code-block:: python

    def make_convert(f):
        nargs = len(inspect.getarg(f).args)
        py_result = make_function_that_call_f(nargs)
        return jit(Object(*[Object] * nargs)(py_result)

Implement further Object -> Numba conversion and vice versa for Array, Int, etc.

Arrays
~~~~~~
Start with just allocation and indexing. This needs pointer support, and
NumPy -> Numba conversion.

Iterators
~~~~~~~~~
Needed for ``for`` loops, e.g. over ``range`` etc. This is mostly implemented,
perhaps with the biggest piece missing a combination of inlining + local
exception rewriting.

CFFI/Ctypes Support
~~~~~~~~~~~~~~~~~~~
Needed to implement a fast and simple minimal array runtime (allocation, etc).

Variants
~~~~~~~~
This may likely need some special support in the compiler as well. The
actual type implementation can be a binary tree like ``StaticTuple``, that
supports the intersection of operations. Pay special attention to ``None``.

Tuples
~~~~~~
Implement static tuples (static size and types), as cons cells. This needs
variants (e.g. ``T | None``). Switch to generic types with generic types.

Objects
~~~~~~~
Miscellaneous things. Implement this is a class that calls functions in
``libcpy.pyx`` via CFFI. Performance is not an issue.

Garbage Collector
~~~~~~~~~~~~~~~~~
Implement a simple GC. Start with Boehm.

Mutable User-defined Types
~~~~~~~~~~~~~~~~~~~~~~~~~~
Allow mutation of objects. This needs a GC to support shared data.

