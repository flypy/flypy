Next iteration of Numba - The core language ideas
=================================================

In order to get a more sustainable compiler development process,
and a more extensible and usable language, we need a small core langauge
and a runtime on top of that core language. The rest of this document
describes the core language.

I believe we need the following features:

    1) Methods on user-defined types with specified representations
       (structs or otherwise)

        - Careful control over allocation, mutability and ownership

    2) Multiple-dispatch and overloading
    3) User-defined typing rules
    4) Careful control over inlining, unrolling and specialization
    5) Extension of the code generator

Support for multi-stage programming would be nice, but is considered a bonus
and deferred to external tools like macropy or mython for now. The
control over optimizations likely provides enough functionality to generate
good code.

This describes a closed environment with an entirely static (but type-inferred)
language. Polymorphism is provided through specialization to monomorphic code,
multiple dispatch (with compile-time selection), or calls through pointers.
Calls through pointers need to be annotated with exception information,
making exception information static (in a conservative way).

Advantages of statically typed methods:

    - useful error messages
    - automatic type checking for built-in data structures (tuple, list,
      dict, etc)

Our priorities are:

    - a maintainable compiler and extensible language
    - ultimate control over performance
    - simple and defined semantics for GPU compatibility
    - array oriented computing
        - higher-order functions map, reduce, filter, etc
        - small numpy-provided API

1. User-defined Types
=====================

We want to support user-defined types with:

    - control over representation
    - (special) methods
    - control over mutability
    - control over stack- vs gc-allocation

User-defined types do not support inheritance, which is left to a runtime
implementation. This means that the callees are static, and can be
inlined (something we will exploit). Furthermore, even when not
inlined the calls are less expensive than virtual calls.

This means that we can write our runtime in this way. The compiler
needs to support the following types natively:

    - int
    - float
    - pointer
    - struct (with optional methods and properties)
    - array (constant size)

Anything else is written in the runtime:

    - range
    - complex
    - array
    - string/unicode
    - etc

This means we can easily experiment with different data representations and
extend functionality. For instance we can wrap and override the native integer
multiply to check for overflow, and raise an exception or issue a warning, or
convert to a BigInt.

Representation
--------------
Type representation can be specified through a type 'layout':

.. code-block:: python

    @jit
    class Array(object):
        layout = Struct([('data', 'Char *')])

Mutability and Allocation
-------------------------
Each individual field can be specified to be immutable, or all can be specified
immutable through a decorator:

.. code-block:: python

    @jit(immutable=True)
    class Array(object):
        ...

If all fields are immutable, the object can be stack allocated. Unless
manually specified with ``stack=True``, the compiler is free to decide where
to allocate the object.

The ``Array`` above can be stack-allocated since its fields are immutable -
even though the contained data may not be.

If data is mutable, it is allocated on the heap. This means that allocation
of such an object is incompatible with a GPU code generator. Hence, data
structures like Arrays must be passed in from the host, and things like Lists
are not supported. However, one can write a List implementation with static
size that supports appending a bounded number of objects.

We disallow explicit stack allocation for mutable types for the following
reason:

.. code-block:: python

    x = mutable() # stack allocate
    y = x         # copy of x
    y.value = 1   # update y.value, which does not affect x.value

To make this work one would need to track the lifetimes of the object itself
and all the variables the object is written into, at which point we defer you
to the Rust programming language. We leave stack allocation of mutable
objects purely as a compile-time optimization.

Destructors
-----------
Destructors are supported only for heap-allocated types, irrespective of
mutability. If a __del__ method is implemented, the object will be
automatically heap-allocated (unless escape analysis can say otherwise).

Ownership
---------
Ownership is tied to mutability:

    - Data is owned when (recursively) immutable
    - Data is shared when some field is mutable (recursively)

Owned data may be send over a channel to another thread or task. Shared data
cannot be send, unless explicitly marked as a safe operation::

    channel.send(borrow(x))

The user must guarantee that 'x' stays alive while it is consumed. This is
useful for things like parallel computation on arrays.

Type Parameters
---------------
User-defined types are parameterizable:

.. code-block:: python

    @jit('Array[type T, Int ndim]')
    class Array(object):
        ...

Parameters can be types or values of builtin types int or float. This enables
a well-defined form of static typing:

.. code-block:: python

    @jit('Array[type T, Int ndim]')
    class Array(object):

        layout = Struct([('data', 'Char *'), ('strides', 'Tuple[Int, ndim]')])

        @signature('Tuple[Int, ndim] -> T')
        def __getitem__(self, indices):
            ...

This specifies that we take a ``Tuple`` of ``Int``s an size ``ndim`` as
argument, and return an item of type ``T``. The ``T`` and ``ndim`` are
resolved as type parameters, which means they specify concrete types in the
method signature.

The type can now be used as follows:

.. code-block:: python

    myarray = Array[Double, 2]()

This will mostly appear in (numba) library code, and not in user-written code,
which uses higher-level APIs that ultimately construct these types. E.g.:

.. code-block:: python

    @multipledispatch(np.ndarray)
    def typeof(array):
        return Array[typeof(array.dtype), array.ndim]

    @multipledispatch(np.dtype)
    def typeof(array):
        return { np.double: Double, ...}[array.dtype]

2. Multiple-dispatch and Overloading
====================================
These mechanisms provide compile-time selection for our language.
It is required to support the compiled ``convert`` from section 3, and
necessary for many implementations, e.g.:

.. code-block:: python

    @jit('Int -> Int')
    def int(x):
        return x

    @jit('String -> Int')
    def int(x):
        return parse_int(x)

Overloading is also provided for methods:

.. code-block:: python

    @jit
    class SomeNeatClass(object):
        @signature('Int -> Int')
        def __add__(self, other):
            return self.value + other

        @signature('String -> Int')
        def __add__(self, other):
            return str(self.value) + other

We further need a way to "overload" python functions to provide a way to
provide alternative implementations or to type it. We can easily provide
implementations for all builtins:

.. code-block:: python

    pytypedef(builtins.int, int)

3. User-defined Typing Rules
============================
I think Julia really shines here. Analogously we define three functions:

    - typeof(Value) -> Type
    - convert(Type, Value) -> Value
    - unify(Type, Type) -> Type

4. Optimization and Specialization
==================================
We need to allow careful control over optimizations and code specialization.
This allows us to use the abstractions we need, without paying them if we
know we can't afford it. We propose the following intrinsics exposed to
users:

    - ``for x in unroll(iterable): ...``
    - ``@specialize.arg(0)``

Unrolling
---------
The first compiler intrinsic allows unrolling over constant iterables.
For instance, the following would be a valid usage:

.. code-block:: python

    x = (1, 2, 3)
    for i in unroll(x):
        ...

An initial implementation will likely simply recognize special container
types (Tuple, List, etc). Later we may allow arbitrary (user-written!)
iterables, where the result of ``len()`` must be ultimately constant (after
inlining and register promotion).

Specialization
--------------
The ability to specialize on various things, similar to specialization in
rpython (``rpython/rlib/objectmodel.py``).

.. function:: specialize.arg(*args)

    Specialize on the listed arguments, e.g. ``specialize.arg(0, 1)``
    specializes on any combination of values for the first and second
    argument.

    This can further allow ``getattr`` and ``setattr`` when used with
    constant strings, allowing generic code.

.. function:: specialize.eval_if_const()

    Evaluate the function at compile time if all arguments are constant,
    and insert the result in the code stream. The result must have a type
    compatible with the signature.

These decorators should also be supported as extra arguments to ``@signature``
etc.

5. Extension of the Code Generator
==================================
We can support an ``@opaque`` decorator that marks a function or method as
"opaque", which means it must be resolved by the code generator. A decorator
``@codegen(thefunc)`` registers a code generator function for the function or
method being called:

.. code-block:: python

    @jit
    class Int(object):
        @opague('Int -> Int', eval_if_const=True)
        def __add__(self, other):
            return a + b

    @codegen(Int.__add__)
    def emit_add(codegen, self, other):
        return codegen.builder.add(self, other)

This can also be useful to retain high-level information, instead of expanding
it out beforehand. This can enable high-level optimizations, e.g. consider
the following code:

.. code-block:: python

    xs = []
    for i in range(n):
        xs.append(i)

    xs = map(f, xs)

If we expand ``xs = []`` and ``xs.append(i)`` into memory allocations and
resizes before considering the ``map``, we forgo a potential optimization
where the compiler performs loop fusion and eliminates the intermediate list.

So an opague function *may* have an implementation, but it may be resolved at
a later stage during the pipeline if it is still needed:

.. code-block:: python

    @codegen(List.__init__)
    def emit_new_list(codegen, self):
        return codegen.builder.new_list(self.type)

    @llcodegen('new_list')
    def emit_new_list(codegen, self):
        return codegen.gen_call(List.__init__)

This should be done with low-level code that doesn't need further high-level
optimizations. Users must also ensure this process terminates (there must
be no cycles the call graph).