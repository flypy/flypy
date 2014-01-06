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

Varargs and Unpacking
~~~~~~~~~~~~~~~~~~~~~
Support for ``def f(*args):`` and support for (static tuple) unpacking in calls.
This is useful to make termeval work for n-ary blaze functions:

    https://github.com/ContinuumIO/blaze/blob/termeval/blaze/bkernel/termeval.py

Higher-order functions
~~~~~~~~~~~~~~~~~~~~~~
Support higher-order functions. Initially using restrictions similar to HM:

.. code-block:: python

    @jit('(a -> b) -> [a] -> [b]')

where we can immediately infer the type of the function.
Implement the optimization ``call(getptr(f, signature) [args]) -> call(f, [args])``.

The following is not allowed initially:

.. code-block:: python

    @jit('(a -> b) -> c')

Later we can implement passing specializations by inferring the uses of the passed
function, and splitting the argument ``f`` into N specialized parameters ``f0, ..., fn-1``.
In generic code this is simply handled by runtime dispatch.

Modular Codegen
~~~~~~~~~~~~~~~
Modular code generation for fast load and compile times.

    - don’t insert runtime pointers
    - reconstruct immutable constants
        - insert external symbols for FFI calls
    - aggregate (specialized) code into compiled modules
    - insert IR globals to pre-specialized code
    - finish one LLVM module per function
    - save metadata regarding specializations (types etc)
    - save metadata regarding module and function dependencies, modification time, etc
    - implement loader
        - load compiled code and attach pointers to IR globals

Constant Folding and Propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This should include ``isinstance``.

Inference <-> Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~
Use type inference to enable optimizations, and optimizations to enable
type inference. For instance, type inference can resolve ``isinstance``,
which can prune branches which would otherwise generate superfluous types
that are merged at subsequent join points, making the types too general.

Subtyping at Type level
~~~~~~~~~~~~~~~~~~~~~~~
Implement a type join that finds the most general type between two types.

Escape Analysis & Stack Allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Not urgent, only an optimization.

Runtime
-------

See ``numba/runtime``. Items are listed in order of urgency.

DyND Support
~~~~~~~~~~~~
Support the DyND array structure natively.

Vectorized Programming
~~~~~~~~~~~~~~~~~~~~~~
Support a Vector and Array data type of a bound-number of stack-allocated values.

C++
~~~
Allow building numba objects in C++ using generated C++ classes. See ``numba2.cppgen``

Datetime
~~~~~~~~
Port datetime support from numba.

Decimals
--------
Finish decimalobject.py

String
------
Add string methods, such as `split` etc.
