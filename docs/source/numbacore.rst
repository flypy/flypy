Next iteration of Numba - The core language ideas
=================================================

This document describes the core numba language, which is designed to generate
efficient code for general, pythonic code. It will allow us to implement most
of the features that currently reside in the compiler directly in a runtime.
On top of this small core language we can write more advanced features such
as subtype polymorphism through method tables.

I believe we need the following features:

    * Methods on user-defined types with specified representations (structs or otherwise)

        - Careful control over allocation, mutability and ownership

    * Polymorphism: Generic functions, subtyping, overloading

        - Generic functions are specialized for the cartesian product of
          input types
        - Polymorphic code can be generated and implemented in a runtime
          implementation (e.g. virtual method tables like numba's extension
          classes)

    * User-defined typing rules
    * Careful control over inlining, unrolling and specialization
    * Array oriented computing: map/reduce/scan/etc
    * Extension of the code generator

Support for multi-stage programming would be nice, but is considered a bonus
and deferred to external tools like macropy or mython for now. The
control over optimizations likely provides enough functionality to generate
good code.

This describes a closed environment with an optionally static, inferred,
language. Static typing will help provide better error messages, and can
prevent inintended use.

Polymorphism is provided through:

    - generic (monomorphized) functions (like C++ templates)
    - overloading
    - subtyping ("python classes")

This language's goals are ultimate control over performance, and a language
with a well-defined and easily understood subset for the GPU.

This language is inspired by the following languages: Rust, Terra, RPython,
Julia, Parakeet, mypy, copperhead. It focusses on static dispatch flexibility,
allowing specialization for static dispatch, or allowing generating more
generic machine code with runtime dispatch. Like RPython, it will further allow
specialization on constant values, which allows generic code to turn into
essentially static code, enabling partial evaluation opportunities as well as
improved type inference.

What we want in our language is full control over specialization and memory
allocation, and easily-understood semantics for what works on the GPU and what
doesn't. The following sections will detail how the above features will
get us there.

1. User-defined Types
---------------------

We want to support user-defined types with:

    - control over representation
    - (special) methods
    - control over mutability
    - control over stack- vs gc-allocation

User-defined types do not support inheritance, which is left to a runtime
implementation. This means that the callees of call-sites are static, and
can be called directly. This further means they can be inlined (something we
will exploit).

This means that we can even write the most performance-critical parts of
our runtime in this way. The compiler needs to support the following types
natively:

    - int
    - float
    - pointer
    - struct (with optional methods and properties)
    - union
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
~~~~~~~~~~~~~~
Type representation can be specified through a type 'layout':

.. code-block:: python

    @jit
    class Array(object):
        layout = Struct([('data', 'Char *')])

Mutability and Allocation
~~~~~~~~~~~~~~~~~~~~~~~~~
Each individual field can be specified to be immutable, or all can be specified
immutable through a decorator:

.. code-block:: python

    @jit(immutable=True)
    class Array(object):
        ...

If all fields are immutable, the object can be stack allocated. Unless
manually specified with ``stack=True``, the compiler is free to decide where
to allocate the object. This decision may differ depending on the target
(cpu or gpu).

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
    y = x         # copy x into y
    y.value = 1   # update y.value, which does not affect x.value

To make this work one would need to track the lifetimes of the object itself
and all the variables the object is written into, at which point we defer you
to the Rust programming language. We leave stack allocation of mutable
objects purely as a compile-time optimization.

Destructors
~~~~~~~~~~~
Destructors are supported only for heap-allocated types, irrespective of
mutability. If a __del__ method is implemented, the object will be
automatically heap-allocated (unless escape analysis can say otherwise).

Ownership
~~~~~~~~~
Ownership is tied to mutability:

    - Data is owned when (recursively) immutable
    - Data is shared when it, or some field is mutable (recursively)

Owned data may be send over a channel to another thread or task. Shared data
cannot be send, unless explicitly marked as a safe operation::

    channel.send(borrow(x))

The user must guarantee that 'x' stays alive while it is consumed. This is
useful for things like parallel computation on arrays.

Type Parameters
~~~~~~~~~~~~~~~
User-defined types are parameterizable:

.. code-block:: python

    @jit('Array[Type dtype, Int ndim]')
    class Array(object):
        ...

Parameters can be types or values of builtin type int. This allows
specialization for values, such as the dimensionality of an array:

.. code-block:: python

    @jit('Array[Type dtype, Int ndim]')
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

    @overload(np.ndarray)
    def typeof(array):
        return Array[typeof(array.dtype), array.ndim]

    @overload(np.dtype)
    def typeof(array):
        return { np.double: Double, ...}[array.dtype]

2. Polymorphism
---------------
Supported forms of polymorphism are generic functions, overloading and
subtyping.

Generic Functions (@autojit)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generic functions are like ``@autojit``, they provide specialized code for
each unique combination of input types. They may be optionally typed and
constrained (through classes or sets of types).

.. code-block:: python

    @jit('(a -> b) -> [a] -> [b]')
    def map(f, xs):
        ...

This specifies a map implementation that is specialized for each combination
of type instances for type variables `a` and `b`. Type variables may be
further constrained by sets of types or by abstract classes or interfaces,
e.g.:

.. code-block:: python

    @jit('Array[A : Float] -> A')
    def sum(xs):
        ...

Here ``Float`` is the unparameterized version of the the ``Float[nbits]`` class,
which allows ``sum`` to accept any array with floating point numbers of any
size.

An other, perhaps more flexible, way to contrain type variables in generic
functions is to use the subtype relation. By default, typed code will accept
subtypes, e.g. if we have a typed argument ``A``, then we will also accept
a subtype ``B`` for that argument. With parameterized types, we have to be
more careful. By default, we allow only invariant parameters, e.g.
``B <: A`` does not imply ``C[B] <: C[A]``. That is, even though ``B``
may be a subtype of ``A``, a class ``C`` parameterized by ``B`` is not a subtype
of class ``C`` parameterized by ``A``. In generic functions, we may however
indicate variance using ``+`` for `covariance` and ``-`` for `contra-variance`:

.. code-block:: python

    @jit('Array[A : +Number] -> A')
    def sum(array):
        ...

This indicates we will accept an array of ``Number``s, or any subtypes
of ``Number``. This is natural for algorithms that read data, e.g if you can
read objects of type ``A``, you can also read objects of subtype ``B`` of ``A``.

However, if we were writing objects, this would break! Consider the following
code:

.. code-block:: python

    @jit('Array[T : +A] -> Void')
    def write(array):
        array[0] = B()

Here we write an ``B``, which clearly satisfies being an ``A``. However,
if we also have ``C <: B``, and if we provide ``write`` with a ``Array[C]``,
we cannot write a ``B`` into this array!

Instead, this code must have a contra-variant parameter, that is, it may accept
an array of ``B`` and an array of any super-type of ``B``.

Overloading and Multiple-dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
----------------------------
I think Julia does really well here. Analogously we define three functions:

    - typeof(pyobj) -> Type
    - convert(Type, Value) -> Value
    - unify(Type, Type) -> Type

The ``convert`` function may make sense as a method on the objects instead,
which is more pythonic, e.g. ``__convert__``. ``unify`` does not really
make sense as a method since it belongs to neither of the two arguments.

Unify takes two types and returns the result type of the given types. This
result type can be specified by the user. For instance, we may determine
that ``unify(Int, Float)`` is ``Union(Int, Float)``, or that it is ``Float``.
The union will give the same result as Python would, but it is also more
expensive in the terms of the operations used on it (and potentially storage
capacity). Unify is used on types only at control flow merge points.

A final missing piece are a form of ad-hoc polymophism, namely coercions.
This is tricky in the presence of overloading, where multiple coercions
are possible, but only a single coercion is preferable. E.g.:

.. code-block:: python

    @overload('Float32 -> Float32 -> Float32')
    def add(a, b):
        return a + b

    @overload('Complex64 -> Complex64 -> Complex64')
    def add(a, b):
        return a + b

Which implementation is ``add(1, 2)`` supposed to pick, ``Int`` freely coerces
to both ``Float32`` and ``Complex64``? Since we don't want built-in coercion
rules, which are not user-overridable or extensible, we need some sort of
coercion function. We choose a function ``coercion_distance(src_type, dst_type)``
which returns the supposed distance between two types, or raises a TypeError.
Since this is not compiled, we decide to not make it a method of the source
type.

.. code-block:: python

    @overload(Int, Float)
    def coercion_distance(int_type, float_type):
        return ...

These functions are used at compile time to determine which conversions to
insert, or whether to issue typing errors.

4. Optimization and Specialization
----------------------------------
We need to allow careful control over optimizations and code specialization.
This allows us to use the abstractions we need, without paying them if we
know we can't afford it. We propose the following intrinsics exposed to
users:

    - ``for x in unroll(iterable): ...``
    - ``@specialize.arg(0)``

Unrolling
~~~~~~~~~
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
~~~~~~~~~~~~~~
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

.. function:: specialize.generic()

    Generate generic machine code instead of specialized code.

These decorators should also be supported as extra arguments to ``@signature``
etc.

5. Data-parallel Operators
--------------------------
Parakeet and copperhead do this really well. We need map, reduce, zip,
list comprehensions, etc.

6. Extension of the Code Generator
----------------------------------
We can support an ``@opaque`` decorator that marks a function or method as
"opaque", which means it must be resolved by the code generator. A decorator
``@codegen(thefunc)`` registers a code generator function for the function or
method being called:

.. code-block:: python

    @jit('Int[Int size]')
    class Int(object):
        @opague('Int -> Int', eval_if_const=True)
        def __add__(self, other):
            return a + b

    @codegen(Int.__add__)
    def emit_add(func, argtypes):
        # return a new typed function...

Conclusion
----------
The mechanisms above allow us to easily evaluate how code will be compiled,
and asses the performance implications. Furthermore, we can easily see what
is GPU incompatible, i.e. anything that:

    - uses CFFI (this implies use of Object, which is implemented in terms
      of CFFI)
    - uses specialize.generic()
    - allocates anything mutable

Everything else should still work.