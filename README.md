% flypy - The core language ideas
% 
% 

This document describes the core flypy language, which is designed to
generate efficient code for general, pythonic code. It will allow us to
implement most of the features that currently reside in the compiler
directly in a runtime. On top of this small core language we can write
more advanced features such as subtype polymorphism through method
tables.

I believe we need the following features:

> -   Methods on user-defined types with specified representations
>     (structs or otherwise)
>
>     > -   Careful control over allocation and mutability
>
> -   Polymorphism: Generic functions, subtyping, overloading
> -   User-defined typing rules
> -   Careful control over inlining, unrolling and specialization
> -   Extension of the code generator
>
Support for multi-stage programming would be nice, but is considered a
bonus and deferred to external tools like macropy or mython for now. The
control over optimizations likely provides enough functionality to
generate good code.

This describes a closed environment with an optionally static, inferred,
language. Static typing will help provide better error messages, and can
prevent inintended use.

Polymorphism is provided through:

> -   generic functions
> -   multiple dispatch
> -   subtyping ("python classes")
> -   coercion

This language's goals are ultimate control over performance, and a
language with a well-defined and easily understood subset for the GPU.

This language is inspired by the following languages: Rust, Terra,
RPython, Julia, Parakeet, mypy, copperhead. It focusses on static
dispatch flexibility, allowing specialization for static dispatch, or
allowing generating more generic machine code with runtime dispatch.
Like RPython, it will further allow specialization on constant values,
which allows generic code to turn into essentially static code, enabling
partial evaluation opportunities as well as improved type inference.

What we want in our language is full control over specialization and
memory allocation, and easily-understood semantics for what works on the
GPU and what doesn't. The following sections will detail how the above
features will get us there.

1. User-defined Types
=====================

We want to support user-defined types with:

> -   control over representation
> -   (special) methods
> -   control over mutability
> -   control over stack- vs gc-allocation

User-defined types do not support inheritance, which is left to a
runtime implementation. This means that the callees of call-sites are
static, and can be called directly. This further means they can be
inlined (something we will exploit).

This means that we can even write the most performance-critical parts of
our runtime in this way. The compiler needs to support the following
types natively:

> -   int
> -   float
> -   pointer
> -   struct (with optional methods and properties)
> -   union
> -   array (constant size)

Anything else is written in the runtime:

> -   range
> -   complex
> -   array
> -   string/unicode
> -   etc

This means we can easily experiment with different data representations
and extend functionality. For instance we can wrap and override the
native integer multiply to check for overflow, and raise an exception or
issue a warning, or convert to a BigInt.

Representation
--------------

Type representation can be specified through a type 'layout':

```python
@jit
class Array(object):
    layout = Struct([('data', 'Char *')])
```

Mutability and Allocation
-------------------------

Each individual field can be specified to be immutable, or all can be
specified immutable through a decorator:

```python
@jit(immutable=True)
class Array(object):
    ...
```

If all fields are immutable, the object can be stack allocated. Unless
manually specified with `stack=True`, the compiler is free to decide
where to allocate the object. This decision may differ depending on the
target (cpu or gpu).

The `Array` above can be stack-allocated since its fields are immutable
-even though the contained data may not be.

If data is mutable, it is allocated on the heap. This means that
allocation of such an object is incompatible with a GPU code generator.
Hence, data structures like Arrays must be passed in from the host, and
things like Lists are not supported. However, one can write a List
implementation with static size that supports appending a bounded number
of objects.

We disallow explicit stack allocation for mutable types for the
following reason:

```python
x = mutable() # stack allocate
y = x         # copy x into y
y.value = 1   # update y.value, which does not affect x.value
```

To make this work one would need to track the lifetimes of the object
itself and all the variables the object is written into, at which point
we defer you to the Rust programming language. We leave stack allocation
of mutable objects purely as a compile-time optimization.

Destructors
-----------

Destructors are supported only for heap-allocated types, irrespective of
mutability. If a \_\_del\_\_ method is implemented, the object will be
automatically heap-allocated (unless escape analysis can say otherwise).

Ownership
---------

Ownership is tied to mutability:

> -   Data is owned when (recursively) immutable
> -   Data is shared when it, or some field is mutable (recursively)

Owned data may be send over a channel to another thread or task. Shared
data cannot be send, unless explicitly marked as a safe operation:

    channel.send(borrow(x))

The user must guarantee that 'x' stays alive while it is consumed. This
is useful for things like parallel computation on arrays.

Type Parameters
---------------

User-defined types are parameterizable:

```python
@jit('Array[Type dtype, Int ndim]')
class Array(object):
    ...
```

Parameters can be types or values of builtin type int. This allows
specialization for values, such as the dimensionality of an array:

```python
@jit('Array[Type dtype, Int ndim]')
class Array(object):

    layout = Struct([('data', 'Char *'), ('strides', 'Tuple[Int, ndim]')])

    @signature('Tuple[Int, ndim] -> T')
    def __getitem__(self, indices):
        ...
```

This specifies that we take a `Tuple` of `Int`s an size `ndim` as
argument, and return an item of type `T`. The `T` and `ndim` are
resolved as type parameters, which means they specify concrete types in
the method signature.

The type can now be used as follows:

```python
myarray = Array[Double, 2]()
```

This will mostly appear in (flypy) library code, and not in user-written
code, which uses higher-level APIs that ultimately construct these
types. E.g.:

```python
@overload(np.ndarray)
def typeof(array):
    return Array[typeof(array.dtype), array.ndim]

@overload(np.dtype)
def typeof(array):
    return { np.double: Double, ...}[array.dtype]
```

2. Polymorphism
===============

Supported forms of polymorphism are generic functions, overloading and
subtyping.

Generic Functions and Subtyping
-------------------------------

For additional details, including implementation details, see
:ref:\`polymorphism\`.

Generic functions allow code to operate over multiple types
simultaneously. For instance, we can type the \`map\` function,
specifying that it maps values of type \`a\` to type \`b\`.

```python
@jit('(a -> b) -> [a] -> [b]')
def map(f, xs):
    ...
```

Type variables may be further constrained by sets of types or by
classes, e.g.:

```python
@jit('Array[A : Float[nbits]] -> A')
def sum(xs):
    ...
```

which allows `sum` to accept any array with floating point numbers or
any subtype is Float. By default, typed code will accept subtypes, e.g.
if we have a typed argument `A`, then we will also accept a subtype `B`
for that argument.

With parameterized types, we have to be more careful. By default, we
allow only invariant parameters, e.g. `B <: A` does not imply
`C[B] <: C[A]`. That is, even though `B` may be a subtype of `A`, a
class `C` parameterized by `B` is not a subtype of class `C`
parameterized by `A`. In generic functions, we may however indicate
variance using `+` for \`covariance\` and `-` for \`contra-variance\`:

```python
@jit('Array[A : +Number] -> A')
def sum(array):
    ...
```

This indicates we will accept an array of `Number`s, or any subtypes of
`Number`. This is natural for algorithms that read data, e.g if you can
read objects of type `A`, you can also read objects of subtype `B` of
`A`.

However, if we were writing objects, this would break! Consider the
following code:

```python
@jit('Array[T : +A] -> Void')
def write(array):
    array[0] = B()
```

Here we write an `B`, which clearly satisfies being an `A`. However, if
we also have `C <: B`, and if we provide `write` with a `Array[C]`, we
cannot write a `B` into this array!

Instead, this code must have a contra-variant parameter, that is, it may
accept an array of `B` and an array of any super-type of `B`.

Generic functions may be specialized or generic, depending on the
decorator used.

Overloading and Multiple-dispatch
---------------------------------

These mechanisms provide compile-time selection for our language. It is
required to support the compiled `convert` from section 3, and necessary
for many implementations, e.g.:

```python
@jit('a : integral -> a')
def int(x):
    return x

@jit('String -> Int')
def int(x):
    return parse_int(x)
```

3. User-defined Typing Rules
============================

I think Julia does really well here. Analogously we define three
functions:

> -   typeof(pyobj) -\> Type
> -   convert(Type, Value) -\> Value
> -   unify(Type, Type) -\> Type

The `convert` function may make sense as a method on the objects
instead, which is more pythonic, e.g. `__convert__`. `unify` does not
really make sense as a method since it belongs to neither of the two
arguments.

Unify takes two types and returns the result type of the given types.
This result type can be specified by the user. For instance, we may
determine that `unify(Int, Float)` is `Union(Int, Float)`, or that it is
`Float`. The union will give the same result as Python would, but it is
also more expensive in the terms of the operations used on it (and
potentially storage capacity). Unify is used on types only at control
flow merge points.

A final missing piece are a form of ad-hoc polymophism, namely
coercions. This is tricky in the presence of overloading, where multiple
coercions are possible, but only a single coercion is preferable. E.g.:

```python
@overload('Float32 -> Float32 -> Float32')
def add(a, b):
    return a + b

@overload('Complex64 -> Complex64 -> Complex64')
def add(a, b):
    return a + b
```

Which implementation is `add(1, 2)` supposed to pick, `Int` freely
coerces to both `Float32` and `Complex64`? Since we don't want built-in
coercion rules, which are not user-overridable or extensible, we need
some sort of coercion function. We choose a function
`coercion_distance(src_type, dst_type)` which returns the supposed
distance between two types, or raises a TypeError. Since this is not
compiled, we decide to not make it a method of the source type.

```python
@overload(Int, Float)
def coercion_distance(int_type, float_type):
    return ...
```

These functions are used at compile time to determine which conversions
to insert, or whether to issue typing errors.

4. Optimization and Specialization
==================================

We need to allow careful control over optimizations and code
specialization. This allows us to use the abstractions we need, without
paying them if we know we can't afford it. We propose the following
intrinsics exposed to users:

> -   `for x in unroll(iterable): ...`
> -   `@specialize.arg(0)`

Unrolling
---------

The first compiler intrinsic allows unrolling over constant iterables.
For instance, the following would be a valid usage:

```python
x = (1, 2, 3)
for i in unroll(x):
    ...
```

An initial implementation will likely simply recognize special container
types (Tuple, List, etc). Later we may allow arbitrary (user-written!)
iterables, where the result of `len()` must be ultimately constant
(after inlining and register promotion).

Specialization
--------------

The ability to specialize on various things, similar to specialization
in rpython (`rpython/rlib/objectmodel.py`).

These decorators should also be supported as extra arguments to
`@signature` etc.

5. Extension of the Code Generator
==================================

We can support an `@opaque` decorator that marks a function or method as
"opaque", which means it must be resolved by the code generator. A
decorator `@codegen(thefunc)` registers a code generator function for
the function or method being called:

```python
@jit('Int[Int size]')
class Int(object):
    @opague('Int -> Int', eval_if_const=True)
    def __add__(self, other):
        return a + b

@codegen(Int.__add__)
def emit_add(func, argtypes):
    # return a new typed function...
```

Conclusion
==========

The mechanisms above allow us to easily evaluate how code will be
compiled, and asses the performance implications. Furthermore, we can
easily see what is GPU incompatible, i.e. anything that:

> -   uses CFFI (this implies use of Object, which is implemented in
>     terms of CFFI)
> -   uses specialize.generic()
> -   allocates anything mutable

Everything else should still work.
Polymorphism
============

As mentioned in :ref:\`core\`, we support the following forms of
polymorphism:

> -   generic functions
> -   multiple dispatch
> -   subtyping ("python classes")
> -   coercion

This section discusses the semantics and implementation aspects, and
goes into detail about the generation of specialized and generic code.

Semantics
---------

A generic function is a function that abstracts over the types of its
parameters. The simplest function is probably the identity function,
which returns its argument unmodified:

```python
@jit('a -> a')
def id(x):
    return x
```

This function can act over any value, irregardless of its type.

Type \`A\` is a subtype of type \`B\` if it is a python subclass.

Specialization and Generalization
---------------------------------

In flypy-lang we need the flexibility to choose between highly
specialized and optimized code, but also more generic, modular, code.
The reason for the latter is largely compilation time and memory use
(i.e. avoiding "code bloat"). But it may even be that code is
distributed or deployed pre-compiled, without any source code (or
compiler!) available.

We will first explain the implementation of the polymophic features
below in a specialized setting, and then continue with how the same
features will work when generating more generic code. We then elaborate
on how these duals can interact.

### Specialized Implementation

Our polymorphic features can be implemented by specializing everything
for everything.

For generic functions, we "monomorphize" the function for the cartesian
product of argument types, and do so recursively for anything that it
uses in turn.

We also specialize for subtypes, e.g. if our function takes a type
\`A\`, we can also provide subtype \`B\`. This allows us to always
statically know the receiver of a method call, allowing us to
devirtualize them.

Finally, multiple-dispatch is statically resolved, since all input types
to a call are known at compile time. This is essentially overloading.

### Generic Implementation

There are many ways to generate more generic code. Generally, to
generate generic code for polymorphic code, we need to represent data
uniformly. We shall do this through pointers. We pack every datum into a
generic structure called a box, and we pass the boxes around. In order
to implement on operation on the box, we need to have some notion of
what the datum in this box "looks like". If we know nothing about the
contents of a box, all we can do is pass it around, and inspect its
type.

We support generic code through the `@gjit` decorator ('generic jit'),
which can annotate functions or entire classes, turning all methods into
generic methods.

We define the following semantics:

> -   Inputs are boxed, unless fully typed
>
>     > -   This guarantees that we only have to generate a single
>     >     implementation of the function
>
> -   Boxes are automatically unboxed to specific types with a runtime
>     type check, unless a bound on the box obsoletes the check
>
>     > -   This allows interaction with more specialized code
>
> -   Bounds on type variables indicate what operations are allowed over
>     the instances
>
> -   Subtyping trumps overloading
>
Further, if we have boxes with unknown contents, they must match the
constraints of whereever they are passed exactly. For instance:

```python
@jit('List[a] -> a -> void')
def append(lst, x):
    ...

@jit('List[a] -> b -> void')
def myfunc(lst, x):
    append(lst, x)  # Error! We don't know if type(a) == type(b)
```

Issuing an error in such situations allows us to avoid runtime type
checks just for passing boxes around. The same goes for return types,
the inferred bounds must match any declared bounds or a type error is
issued.

We implement subtyping through virtual method tables, similar to C++,
Cython and a wide variety of other languages. To provide varying arity
for multiple-dispatch and overriding methods, arguments must be packed
in tuples or arrays of pointers along with a size. Performing dispatch
is the responsibility of the method, not the caller, which eliminates an
indirection and results only in slower runtime dispatch where needed.

Finally, multiple dispatch is statically resolved if possible, otherwise
it performs a runtime call to a generic function that resolves the right
function given the signature object, a list of overloads and the runtime
arguments.

Coercion is supported only to unbox boxes with a runtime check if
necessary.

#### Bounds

Users may specify type bounds on objects, in order to provide operations
over them. For instance, we can say:

```python
@jit('a <: A[] -> a')
def func(x):
    ...
```

Alternatively, one could write 'A[] -\> A[]', which has a subtly
different meaning if we put in a subtype \`B\` of class \`A\` (instead
of getting back a \`B\`, we'd only know that we'd get back an object of
type \`A\`).

We realize that we don't want to be too far removed from python
semantics, and in order to compare to objects we don't want to inherit
from a say, a class \`Comparable\`. So by default we implement the Top
in the type lattice, which we know as \`object\`. This has default
implementations for most special methods, raising a NotImplementedError
where implementation is not sensible.

Interaction between Specialized and Generic Code
------------------------------------------------

In order to understand the interaction between specialized and generic
code, we explore the four bridges between the two:

### Generic \<-\> Generic

Pass around everything in type-tagged boxes, retain pointer to vtable in
objects. If there are fully typed parameters, allow those to be passed
in unboxed, and generate a wrapper function that takes those arguments
as boxes and unboxes them.

### Generic \<-\> Specialized

Generally generic code can call specialized functions or methods of
objects of known type directly. Another instance of this occurs when
instances originate from specialized classes. Consider populating a list
of an int, string and float. Generic wrappers are generated around the
specialized methods, and a vtable is populated. The wrappers are
implemented as follows:

```python
@gjit('a -> a -> bool')
def wrapper_eq(int_a, int_b):
    return box(specialized_eq(unbox(a), unbox(b)))
```

We further need to generate properties that box specialized instance
data on read, and unbox boxed values on write.

### Specialized \<-\> Generic

Generally specialized code can call generic functions or methods of
objects that are not statically known (e.g. "an instance of A or some
subtype"). The specialized code will need to box arguments in order to
apply such a function. This means that generic wrapper classes need to
be available for specialized code. For parameterized types this means we
get a different generic class for every different combination of
parameters of that type.

We may further allow syntax to store generic objects in specialized
classes, e.g.

```python
@jit
class MyClass(object):
    layout = [('+A[]', 'obj')]
```

Which indicates we can store a generic instance of \`A\` or any subtype
in the \`obj\` slot.

### Specialized \<-\> Specialized

Static dispatch everywhere.

Variance
========

Finally, we return to the issue of variance. For now we disallow subtype
bounds on type variables of parameterized types, allowing only
invariance on parameters. This avoids the read/write runtime checks that
would be needed to guarantee type safety, as touched on in
:ref:\`core\`.

For bonus points, we can allow annotation of variance in the type
syntax, allowing more generic code over containers without excessive
runtime type checks:

```python
@jit('List[+-a]')
class List(object):

    @jit('List[a] -> int64 -> +a)
    def __getitem__(self, idx):
        ...

    @jit('List[a] -> int64 -> -a -> void)
    def __setitem__(self, idx, value):
        ...
```

This means that if we substitute a \`List[b]\` for a \`List[a]\`, then
for a read operations we have the constraint that \`b \<: a\`, since
\`b\` can do everything \`a\` does. For a write operation we have that
\`a \<: b\`, since if we are to write objects of type \`a\`, then the
\`b\` must not be more specific than \`a\`.

This means the type checker will automatically reject any code that does
not satisfy the contraints originated by the operations used in the
code.
% Fusion
% 
% 

We want to fuse operations producing intermediate structures such as
lists or arrays. Fusion or deforestation has been attempted in various
ways, we will first cover some of the existing research in the field.

Deforestation
=============

build/foldr
-----------

Rewrite rules can be used to specify patterns to perform fusion ([1]\_,
[2]\_, [3]\_), e.g.:

    map f (map g xs) = map (f . g) xs

The dot represents the composition operator. To avoid the need for a
pattern for each pair of operators, we can express fusable higher-order
functions in terms of a small set of combinators. One approach is
build/foldr, where `build` generates a list, and `foldr` (reduce)
consumes it ([3]). Foldr can be defined as follows:

```haskell
foldr f z []     = z
foldr f z (x:xs) = f x (foldr f z xs)
```

`build` is the dual of `foldr`, instead of reducing a list it generates
one. Using just build and foldr, a single rewrite rule can be used for
deforestation:

> foldr k z (build g) = g k z

This is easy to understand considering that build generates a list, and
foldr then consumes it, so there's no point in building it in the first
place. Build is specified as follows:

```haskell
build g (:) []
```

This means `g` is applied to the `cons` constructor and the empty list.
We can define a range function (`from` in [3]) as follows:

```haskell
range a b = if a > b then []
            else a : (range (a + 1) b)
```

Abstracting over cons and nil (the empty list) [3], we get:

```haskell
range' a b = \ f lst -> if a > b then lst
                        else f a (range' (a + 1) b f lst)
```

It's easy to see the equivalence to `range` above by substituting `(:)`
for `f` and `[]` for lst. We can now use `range'` with `build` ([3]):

```haskell
range a b = build (range' a b)
```

Things like `map` can now be expressed as follows ([3]):

```haskell
map f xs = build (\ cons lst -> foldr (\ a b -> cons (f a) b) lst xs)
```

However, some functions cannot be expressed in this framework, like zip
([4]\_).

Streams
-------

Another major approach is based on stream fusion ([4]\_, [5]\_). It
expresses the higher-order functions in terms of streams ([4]\_):

```haskell
map f = unstream . map' f . stream
```

`unstream` converts a stream back to a list, and stream converts a list
to a stream. Under composition, like `map f (map g xs)`, we get
`unsteam . map' f . stream . unsteam . map' g . stream`. The fusion then
relies on eliminating the composition of `stream` with `unstream`:

> stream (unstream s) = s

A stream consists of a stepper function and a state. Stepper functions
produce new step states. The states are `Done`, `Yield` or `Skip`.
`Done` signals that the stream is consumed, `Yield` yields a new value
and state, and `Skip` signals that a certain value needs to be skipped
(for things like filter).

Let's see this in action ([5]):

```haskell
stream :: [a] -> Stream a
stream xs0 = Stream next xs0
    where
        next []     = Done
        next (x:xs) = Yield x xs
```

This converts a list to a Stream. It constructs a Stream with a new
stepper function `next` and the initial state (the given list). The
`next` stepper function produces a new step state every time it is
called. Streams can be consumed as follows:

```haskell
map f (Stream next0 s0) = Stream next s0
    where
        next s = case next0 s of
            Done        -> Done
            Skip s'     -> Skip s'
            Yield x s'  -> Yield (f x) s'
```

Here we specify a new stepper function `next` that, given a state,
advances the stream it consumes with the new state, and yields new
results. It wraps this stepper function in a new stream. [5]\_ further
extends this work to allow operation over various kinds of streams:

> -   Chunked streams for bulk memory operations
> -   Vector (multi) streams for SIMD computation
> -   Normal streams that yield one value at a time

It bundles the various streams together in a product type. The idea is
that all streams are available at the same time. Hence a producer can
produce in the most efficient way, and the consumer can consume in the
most efficient way. These concepts don't always align, in which case
fallbacks are in place, for instance a chunked stream can be processed
as a scalar stream, or vice-versa. In addition to inlining and other
optimizations it relies heavily on call-pattern specialization ([6]),
allowing the compiler to eliminate pattern matching of consumer sites.

Fusion in flypy
===============

The concept of a stream encapsulating a state and a stepper function is
akin to iterators in Python, where the state is part of the iterator and
the stepping functionality is provided by the `__next__` method.
Although iterators can be composed and specialized on static callee
destination ( the \_\_next\_\_ method of another iterator), they are
most naturally expressed as generators:

    def map(f, xs):
        for x in xs:
            yield f(xs)

The state is naturally captured in the generator's stack frame. To allow
fusion we need to inline producers into consumers. This is possible only
if we can turn the lazy generator into a non-lazy producer, i.e. the
consumer must immediately consume the result. This introduces a
restriction:

> -   The generator may not be stored, passed to other functions or
>     returned. We can capture this notion by having `iter(generator)`
>     create a `stream`, and disallowing the rewrite rule
>     `stream (unstream s) = s` to trigger when the `unstream` has
>     multiple uses.
>
>     This means the value remains \`unstreamed\` (which itself is lazy,
>     but effectively constitutes a fusion boundary).
>
Since we can express many (all?) higher-order fusable functions as
generator, we have a powerful building block (in the same way as the
previously outlined research methods), that will give us rewrite rules
for free. I.e., we will not need to state the following:

```python
map(f, map(g, xs)) = map(f . g, xs)
```

since this automatically follows from the definition of map:

```python
@signature('(a -> b) -> Stream a -> Stream b')
def map(f, xs):
    for x in xs:
        yield f(x)
```

The two things that need to be addressed are 1) how to inline generators
and 2) how do we specialize on argument "sub-terms".

1. Inlining Generators
======================

The inlining pattern is straightforward:

> -   remove the loop back-edge
> -   promote loop index to stack variable
> -   inline generator
> -   transform 'yield val' to 'i = val'
> -   replace each 'yield' from the callee with a copy of the loop body
>     of the caller

Now consider a set of generators that have multiple yield expressions:

```python
def f(x):
    yield x
    yield x
```

Inlining of the producer into the consumer means duplicating the body
for each yield. This can lead to exponential code explosion in the size
of the depth of the terms:

```python
for i in f(f(f(x))):
    print i
```

Will result in a function with 8 print statements. However, it is not
always possible to generate static code without multiple yields,
consider the concatenation function:

```python
def concat(xs, ys):
    for x in xs:
        yield x
    for y in ys:
        yield ys
```

This function has two yields. If we rewrite it to use only one yield:

```python
def concat(xs, ys):
    for g in (xs, ys):
        for x in g:
            yield x
```

We have introduced dynamicity that cannot be eliminated without
specialization on the values (i.e. unrolling the outer loop, yielding
the first implementation). This not special in any way, it is inherent
to inlining and we and treat it as such (by simply using an inlining
threshold). Crossing the threshold simply means temporaries are not
eliminated -- in this case this means generator "cells" remain.

If this proves problematic, functions such as concat can instead always
unstream their results. Even better than fully unstreaming, or sticking
with a generator cell, is to use a buffering generator fused with the
expression that consumes N iterations and buffers the results. This
divides the constant overhead of generators by a constant factor.

2. Specialization
-----------------

Specialization follows from inlining, there are two cases:

> -   internal terms
> -   boundary terms
> -   `stream (unstream s)` is rewritten, the result is fused

Internal terms are rewritten according to the `stream (unstream s)`
rule. What eventually follows at a boundary is a) consumption through a
user-written loop or b) consumption through the remaining unstream. In
either case the result is consumed, and the inliner will start inlining
top-down (reducing the terms top-down).

SIMD Producers
==============

For simplicity we exclude support for chunked streams. Analogous to
[5]\_ we can expose a SIMD vector type to the user. This vector can be
yielded by a producer to a consumer.

How then, does a consumer pick which stream to operate on? For instance,
zip can only efficiently be implemented if both inputs are the same, not
if one returns vectors and the other scalars (or worse, switching back
and forth mid-way):

```python
def zip(xs, ys):
    while True:
        try:
            yield (next(xs), next(ys))
        except StopIteration:
            break
```

For functions like zip, which are polymorphic in their arguments, we can
simply constrain our inputs:

```python
@overload('Stream[Vector a] -> Stream[Vector b] -> Stream[(Vector a, Vector b)]')
@overload('Stream a -> Stream b -> Stream (a, b)')
def zip(xs, ys):
    ...
```

Of course, this means if one of the arguments produces vectors, and the
other scalars, we need to convert one to the other:

```python
@overload('Stream[Vector a] -> Stream a')
def convert(stream):
    for x in stream:
        yield x
```

Which basically unpacks values from the SIMD register.

Alternatively, a mixed stream of vectors and scalars can be consumed.
[5]\_ distinguises between two vector streams:

> -   a producer stream, which can yield Vector | Scalar
> -   a consumer stream, where the consumer chooses whether to read
>     vectors or scalars. A consumer can start with vectors, and when
>     the vector stream is consumed read from the scalar stream.

A producer stream is useful for producers that mostly yield vectors, but
sometimes need to yield a few scalars. This class includes functions
like concat that concatenates two streams, or e.g. a stream over a
multi-dimensional array where inner-contiguous dimensions have a number
of elements not 0 modulo the vector size.

A consumer stream on the other hand is useful for functions like zip,
allowing them to vectorize part of the input. However, this does not
seem terribly useful for multi-dimensional arrays with contiguous rows,
where it would only vectorize the first row and then fall back to
scalarized code.

However, neither model really makes sense for us, since we would already
manually specialize our loops:

```python
@overload('Array a 2 -> Stream a')
def stream_array(array, vector_size):
    for row in array:
        for i in range(len(row) / vector_size):
            yield load_vector(row.data + i * 4)

        for i in range(i * 4, len(row)):
            yield row[i]
```

This means code consuming scalars and code consuming vectors can be
matched up through pattern specialiation (which is not just type-based
branch pruning).

To keep things simple, we will stick with a producer stream, yielding
either vectors or scalars. Consumers then pattern-match on the produced
values, and pattern specialization can then switch between the two
alternatives:

```python
def sum(xs):
    vzero = Vector(zero)
    zero = 0
    for x in xs:
        if isinstance(x, Vector):
            vzero += x
        else:
            zero += x
    return zero + vreduce(add, vzero)
```

To understand pattern specialization, consider `xs` is a
`stream_array(a)`. This results in approximately the following code
after inlining:

```python
stream_array(array, vector_size):
    for row in array:
        for i in range(len(row) / vector_size):
            x = load_vector(row.data + i * 4)
            if isinstance(x, Vector):
                vzero += x
            else:
                zero += x

        for i in range(i * 4, len(row)):
            x = row[i]
            if isinstance(x, Vector):
                vzero += x
            else:
                zero += x
```

It is now easy to see that we can eliminate the second pattern in the
first loop, and the first pattern in the second loop.

Compiler Support
================

To summarize, to support fusion in a general and pythonic way can be
modelled on generators. To support this we need:

> -   generator inlining
> -   For SIMD and bulk operations, call pattern specialization. For us
>     this means branch pruning and branch merging based on type.

The most important optimization is the fusion, SIMD is a useful
extension. Depending on the LLVM vectorizer (or possibly our own), it
may not be necessary.

References
==========
% Typing
% 
% 

This section discusses typing for flypy. There is plenty of literature
on type inference, most notable is the Damas-Hindley-Milner Algorithm W.
for lambda calculus [1]\_, and an extension for ML. The algorithm
handles let-polymorphism (a.k.a. ML-polymorphism), a form of parametric
polymorphism where type variables themselves may not be polymorphic. For
example, consider:

```python
def f(g, x):
    g(x)
    g(0)
```

We can call `f` with a function, which must accept `x` and an value of
type int. Since `g` is a monotype in `f`, the second call to `g`
restricts what we accept for `x`: it must be something that promotes
with an integer. In other words, the type for `g` is `a -> b` and not
`∀a,b.a -> b`.

Although linear in practise, the algorithm's worst case behaviour is
exponential ([2]\_), since it does not share results for different
function invocations. The cartesian product algorithm ([3]\_) avoids
this by sharing monomorphic template instantiations. It considers all
possible receivers of a message send, and takes the union of the results
of all instances of the cartesian product substitution. The paper does
not seem to address circular type dependencies, where the receiver can
change based on the input types:

```python
def f(x):
    for i in range(10):
        x = g(x)
```

leading to

```llvm
define void f(X0 %x0) {
cond:
    %0 = lt %i 10
    cbranch %0 body exit

body:
    %x1 = phi(x0, x2)
    %x2 = call g(%x0)
    br cond

exit:
    ret void
}
```

However, this can be readily solved through fix-point iteration. If we
assign type variables throughout the function first, we get the
following constraints:

    [ X1 = Union(X0, X2), G = X1 -> T2 , X2 = T2 ]

We can represent a function as a set of overloaded signatures. However,
the function application is problematic, since we send X1 (which will be
assigned a union type). WIthout using the cartesian product this would
lead to exponential behaviour since there are 2\^N subsets for N types.

Type inference in flypy
=======================

We use the cartesian product algorithm on a constraint network based on
the dataflow graph. To understand it, we need to understand the input
language. Since we put most functionality of the language in the
user-domain, we desugar operator syntax through special methods, and we
further support overloaded functions.

The front-end generates a simple language that can conceptually be
described through the syntax below:

    e = x                           variable
      | x = a                       assignment
      | const(x)                    constants
      | x.a                         attribute
      | f(x)                        application
      | jump/ret/exc_throw/...      control flow
      | (T) x                       conversion

As you'll notice, there are no operators, loops, etc. Control flow is
encoded through jumps, exception raising, return, etc. Loops can be
readily detected through a simple analysis (see
pykit/analysis/loop\_detection.py).

We take this input grammar and generate a simpler constraint network,
that looks somewhat like this:

    e = x.a             attribute
      | f(x)            application
      | flow(a, b)      data flow

This is a directed graph where each node classifies the constraint on
the inputs. Types propagate through this network until no more changes
can take place. If there is an edge `A -> B`, then whenever `A` is
updated, types are propagated to `B` and processed according to the
constraint on `B`. E.g. if `B` is a function call, and `A` is an input
argument, we analyze the function call with the new values in the
cartesian product.

Coercions
=========

Coercions may happen in two syntactic constructs:

> -   application
> -   control flow merges (phi nodes)

For application we have a working implementation in Blaze that
determines the best match for polymorphic type signatures, and allows
for coercions. For control flow merges, the user can choose whether to
promote values, or whether to create a sum-type. A post-pass can simply
insert coercions where argument types do not match parameter types.

Subtyping
=========

We intend to support subtyping in the runtime through python
inheritance. When a class B inherits from a class A, we check for a
compatible interface for the methods (argument types are contravariant
and return types covariant). When typing, the only thing we need to
implement are coercion and unification:

> Type B coerces to type A if B is a subtype of A Type A coerces to type
> B if B is a subtype of A with a runtime check only

Then class types A and B unify iff A is a subtype of B or vice-versa.
The result of unification is always the supertype.

Finally, parameteric types will be classified invariant, to avoid
unintended mistakes in the face of mutable containers. Consider e.g.
superclass `A` and subclass `B`. Assume we have the function that
accepts an argument typed `A[:]`. If we treat the dtype as covariant,
then we may pass an array `B[:]` for that argument. However, the code
can legally write `A`s into the array, violating the rule that we can
only assign subtypes. The problem is that reading values is covariant,
whereas writing is contravariant. In other words, the parameter must be
covariant as well as contravariant at the same time, which is only
satisfied when `A = B`.

The exception is maybe function types, for which we have built-in
variance rules.

Parameterization
================

Types can only be parameterized by variables and user-defined or
built-in types.

References
==========
% flypy Runtime
% 
% 

Nearly all built-in data types are implemented in the runtime.

Garbage Collector
=================

To support mutable heap-allocated types, we need a garbage collector. To
get started quickly we can use Boehm or reference counting. We will want
to port one of the available copying collectors and use a shadowstack or
a lazy pointer stack (for bonus points). The GC should then be local to
each thread, since there is no shared state between threads (only owned
and borrowed data is allowed).

Garbage collection is abstracted by pykit.

Exceptions
==========

Exceptions are also handled by pykit. We can implement several models,
depending on the target architecture:

> -   costful (error return codes)
>     :   -   This will be used on the GPU
>
> -   zero-cost
>     :   -   This should be used where supported. We will start with
>             costful
>
> -   setjmp/longjmp
>     :   -   This will need to happen for every stack frame in case of
>             a shadow stack
>
Local exception handling will be translated to jumps. This is not
contrived, since we intend to make heavy use of inlining:

```python
while 1:
    try:
        i = x.__next__()
    except StopIteration:
        break
```

`x.__next__()` may be inlined (and will be in many instances, like
range()), and the `raise StopIteration` will be translated to a jump.
Control flow simplification can further optimize the extra jump (jump to
break, break to loop exit).

Threads
=======

As mentioned in the core language overview, memory is not shared unless
borrowed. This process is unsafe and correctness must be ensured by the
user. Immutable data can be copied over channels between threads. Due to
a thread-local GC, all threads can run at the same time and allocate
memory at the same time.

We will remove prange and simply use a parallel map with a closure.

Extension Types
===============

Extension types are currently built on top of CPython objects. This
should be avoided. We need to decouple flypy with anything CPython, for
the sake of portability as well as pycc.

Extension types can also easily be written in the runtime:

> -   `unify()` needs to return the supertype or raise a type error
> -   `convert(obj, Object)` needs to do a runtime typecheck
> -   `coerce_distance` needs to return a distance for how far the
>     supertype is up the inheritance tree

The approach is simple: generate a wrapper method for each method in the
extension type that does a vtable lookup.

Closures
========

This time we will start with the most common case: closures consumed as
inner functions. This means we don't need dynamic binding for our cell
variables, and we can do simple lambda lifting instead of complicated
closure conversion. This also trivially works on the GPU, allowing one
to use map, filter etc, with lambdas trivially.
