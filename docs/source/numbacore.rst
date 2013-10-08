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

    * Polymorphism: Generic functions, traits, overloading

        - subtyping and inheritance is left to a runtime implementation
        - dynamic dispatch for traits is left to a runtime implementation
            - static dispatch only requires some type checking support

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
    - traits (like interfaces)
    - subtyping ("python classes")

This language's goals are ultimate control over performance, and a language
with a well-defined and easily understood subset for the GPU.

This language is inspired by the following languages: Rust, Terra, RPython,
Julia, Parakeet, mypy, copperhead. The traits are very similar to Rust's
traits, and are related to type classes in Haskell and interfaces in Go.

However, Go interfaces do not allow type-based specialization, and hence
need runtime type tagging and method dispatch through vtables. Type
conversion between interfaces needs to be runtime-checked type (and new
vtables build at those points, if not cached).
Compile-time overloading is precluded. In Go, interfaces specify what something
*can do*, as opposed to what something *can be*. This can be a useful in a few
situations, but it means we cannot constrain what things can be (e.g.
any numeric type).

In julia we can constrain the types we operate over, which happens through
subtyping. E.g.:

.. code-block:: julia

    julia> Int <: Integer
    true
    julia> Int <: Real
    true
    julia> Int <: FloatingPoint
    false

So we can define a function which operates only over numbers::

    julia> function f(x :: Number)
             return x * x
           end

Here's a the generated code when ``x`` is an ``Int``:

.. code-block:: llvm

    julia> disassemble(f, (Int,))

    define %jl_value_t* @f618(%jl_value_t*, %jl_value_t**, i32) {
    top:
      %3 = load %jl_value_t** %1, align 8, !dbg !5256
      %4 = getelementptr inbounds %jl_value_t* %3, i64 0, i32 0, !dbg !5256
      %5 = getelementptr %jl_value_t** %4, i64 1, !dbg !5256
      %6 = bitcast %jl_value_t** %5 to i64*, !dbg !5256
      %7 = load i64* %6, align 8, !dbg !5256
      %8 = mul i64 %7, %7, !dbg !5263
      %9 = call %jl_value_t* @jl_box_int64(i64 %8), !dbg !5263
      ret %jl_value_t* %9, !dbg !5263
    }

Disassembling with ``Number`` generates a much larger chunk of code, which
uses boxed code and ultimately (runtime) multiple dispatch of the ``*``
function:

.. code-block:: llvm

    %15 = call %jl_value_t* @jl_apply_generic(%jl_value_t* inttoptr (i64 4316726176 to %jl_value_t*), %jl_value_t** %.sub, i32 2), !dbg !5191

However, since the implementation of a function is specialized for the
supertype, it doesn't know the concrete subtype.
Type inference can help prevent these situations and use subtype-specialized
code. However, it's very easy to make it generate slow code:

.. code-block:: julia

    julia> function g(c)
         if c > 2
           x = 2
         else
           x = 3.0
         end
         return f(x)
       end

    julia> disassemble(g, (Bool,))

This prints a large chunk of LLVM code (using boxed values), since we are
unifying an Int with a Float. Using both ints, or both floats however leads
to very efficient code.

What we want in our language is full control over specialization and memory
allocation, and easily-understood semantics for what works on the GPU and what
doesn't. The following sections will detail how the above features will
get us there.

1. User-defined Types
=====================

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
-----------
Destructors are supported only for heap-allocated types, irrespective of
mutability. If a __del__ method is implemented, the object will be
automatically heap-allocated (unless escape analysis can say otherwise).

Ownership
---------
Ownership is tied to mutability:

    - Data is owned when (recursively) immutable
    - Data is shared when it, or some field is mutable (recursively)

Owned data may be send over a channel to another thread or task. Shared data
cannot be send, unless explicitly marked as a safe operation::

    channel.send(borrow(x))

The user must guarantee that 'x' stays alive while it is consumed. This is
useful for things like parallel computation on arrays.

Type Parameters
---------------
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
===============
Supported forms of polymorphism are generic functions, overloading, traits
and subtyping and inheritance.

Generic Functions (@autojit)
----------------------------
Generic functions are like ``@autojit``, they provide specialized code for
each unique combination of input types. They may be optionally typed and
constrained (through traits).

.. code-block:: python

    @jit('(a -> b) -> [a] -> [b]')
    def map(f, xs):
        ...

This specifies a map implementation that is specialized for each combination
of type instances for type variables `a` and `b`. Type variables may be
further constrained through traits, in a similar way to Rust's traits
(http://static.rust-lang.org/doc/tutorial.html#traits), allowing you to
operate for instance only on arrays of numbers, or arrays of floating point
values.

Traits
------
Traits specify an interface that value instances implement. Similarly
to Rust's traits and Haskell's type classes, they are a form of bounded
polymorphism, allowing users to constrain type variables ("this
function operates on floating point values only").

They also specify a generic interface that objects can implement. Classes
can declare they belong to a certain trait, allowing any instance of the class
to be used through the trait:

.. code-block:: python

    @jit('(a -> b) -> Iterable[a] -> [b]')
    def map(f, xs):
        ...

Our map now takes an iterable and returns a list. Written this way,
a single map implementation now works for *any* iterable. Any value
implementing the Iterable trait can now be used:

.. code-block:: python

    @jit('Array[Type dtype, Int ndim]')
    class Array(Iterable['dtype']):
        ...

We can now use map() over our array. The generated code must now insert
a `conversion` between ``Array[dtype, ndim]`` and trait ``Iterable[dtype]``,
which concretely means packing up a vtable pointer and a boxed Array pointer.
This form of polymorphism will likely be *incompatible with the GPU backend*.
However, we can still use our generic functions by telling the compiler to
specialize on input types:

.. code-block:: python

    @specialize.argtypes('f', 'xs')
    @jit('(a -> b) -> Iterable[a] -> [b]')
    def map(f, xs):
        ...

Alternatively, we can allow them to simply constrain type variables, and
not actually specify the type as the trait. The type is supplied instead by
the calling context:

.. code-block:: python

    @signature('(it:Iterable[a]) => (a -> b) -> it -> [b]')
    def map(f, xs):
        ...

The constraints are specified in similar way to Haskell's type classes.
The only implementation required in the compiler to support this is the type
checking feature, otherwise it's entirely the same as generic functions above.
Multiple constraints can be expressed, e.g. ``(it:Iterable[a], a:Integral)``.
Alternative syntax could be '(a -> b) -> lst : Iterable[a] -> [b]', but this
is less clear when 'it' is reused elsewhere as a type variable.

Traits can further use inheritance and have default implementations. This can
be trivially implemented at the Python level, requiring no special knowledge
in the compiler.

Overloading and Multiple-dispatch
---------------------------------
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

.. function:: specialize.argtypes(*args)

    Specialize on trait argument types, potentially "untraiting" the
    specialization.

These decorators should also be supported as extra arguments to ``@signature``
etc.

5. Data-parallel Operators
==========================
Parakeet and copperhead do this really well. We need map, reduce, zip,
list comprehensions, etc.

6. Extension of the Code Generator
==================================
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
    def emit_add(codegen, self, other):
        # 'self' and 'other' are (typed) pykit values
        return codegen.builder.add(self, other)

This can also be useful to retain high-level information, instead of expanding
it out beforehand. This can enable high-level optimizations, e.g. consider
the following code:

.. code-block:: python

    L = []
    for i in range(n):
        L.append(i)

    L = map(f, L)

If we expand ``L = []`` and ``L.append(i)`` into memory allocations and
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

Conclusion
==========
The mechanisms above allow us to easily evaluate how code will be compiled,
and asses the performance implications. Furthermore, we can easily see what
is GPU incompatible, i.e. anything that:

    - uses CFFI (this implies use of Object, which is implemented in terms
      of CFFI)
    - uses traits that don't merely constrain type variables
    - allocates anything mutable

Everything else should still work.