Polymorphism
============

As mentioned in :ref:`core`, we support the following forms of polymorphism:

    - generic functions
    - multiple dispatch
    - subtyping ("python classes")
    - coercion

This section discusses the semantics and implementation aspects, and goes
into detail about the generation of specialized and generic code.

Semantics
---------

A generic function is a function that abstracts over the types of its
parameters. The simplest function is probably the identity function,
which returns its argument unmodified:

.. code-block:: python

    @jit('a -> a')
    def id(x):
        return x

This function can act over any value, irregardless of its type.

Type `A` is a subtype of type `B` if it is a python subclass.

Specialization and Generalization
---------------------------------
In flypy-lang we need the flexibility to choose between highly specialized and
optimized code, but also more generic, modular, code. The reason for the
latter is largely compilation time and memory use (i.e. avoiding "code bloat").
But it may even be that code is distributed or deployed pre-compiled, without
any source code (or compiler!) available.

We will first explain the implementation of the polymophic features below
in a specialized setting, and then continue with how the same features will
work when generating more generic code. We then elaborate on how these duals
can interact.

Specialized Implementation
++++++++++++++++++++++++++
Our polymorphic features can be implemented by specializing everything for
everything.

For generic functions, we "monomorphize" the function for the
cartesian product of argument types, and do so recursively for anything that
it uses in turn.

We also specialize for subtypes, e.g. if our function takes
a type `A`, we can also provide subtype `B`. This allows us to always
statically know the receiver of a method call, allowing us to devirtualize
them.

Finally, multiple-dispatch is statically resolved, since all input types to
a call are known at compile time. This is essentially overloading.

Generic Implementation
++++++++++++++++++++++

There are many ways to generate more generic code. Generally, to generate
generic code for polymorphic code, we need to represent data uniformly.
We shall do this through pointers. We pack every datum into a generic
structure called a box, and we pass the boxes around. In order to implement
on operation on the box, we need to have some notion of what the datum in this
box "looks like". If we know nothing about the contents of a box, all we can
do is pass it around, and inspect its type.

We support generic code through the ``@gjit`` decorator ('generic jit'),
which can annotate functions or entire classes, turning all methods into
generic methods.

We define the following semantics:

    * Inputs are boxed, unless fully typed

        - This guarantees that we only have to generate a single implementation
          of the function

    * Boxes are automatically unboxed to specific types with a runtime
      type check, unless a bound on the box obsoletes the check

        - This allows interaction with more specialized code

    * Bounds on type variables indicate what operations are allowed over
      the instances

    * Subtyping trumps overloading


Further, if we have boxes with unknown contents, they must match the
constraints of whereever they are passed exactly. For instance:

.. code-block:: python

    @jit('List[a] -> a -> void')
    def append(lst, x):
        ...

    @jit('List[a] -> b -> void')
    def myfunc(lst, x):
        append(lst, x)  # Error! We don't know if type(a) == type(b)

Issuing an error in such situations allows us to avoid runtime type checks
just for passing boxes around. The same goes for return types, the
inferred bounds must match any declared bounds or a type error is issued.

We implement subtyping through virtual method tables, similar to C++, Cython
and a wide variety of other languages. To provide varying arity for
multiple-dispatch and overriding methods, arguments
must be packed in tuples or arrays of pointers along with a size. Performing
dispatch is the responsibility of the method, not the caller, which eliminates
an indirection and results only in slower runtime dispatch where needed.


Finally, multiple dispatch is statically resolved if possible, otherwise it
performs a runtime call to a generic function that resolves the right function
given the signature object, a list of overloads and the runtime arguments.


Coercion is supported only to unbox boxes with a runtime check if necessary.


Bounds
~~~~~~
Users may specify type bounds on objects, in order to provide operations over
them. For instance, we can say:

.. code-block:: python

    @jit('a <: A[] -> a')
    def func(x):
        ...

Alternatively, one could write 'A[] -> A[]', which has a subtly different
meaning if we put in a subtype `B` of class `A` (instead of getting back a
`B`, we'd only know that we'd get back an object of type `A`).

We realize that we don't want to be too far removed from python semantics,
and in order to compare to objects we don't want to inherit from a say,
a class `Comparable`. So by default we implement the Top in the type lattice,
which we know as `object`. This has default implementations for most special
methods, raising a NotImplementedError where implementation is not sensible.

Interaction between Specialized and Generic Code
------------------------------------------------
In order to understand the interaction between specialized and generic code,
we explore the four bridges between the two:

Generic <-> Generic
+++++++++++++++++++
Pass around everything in type-tagged boxes, retain pointer to vtable in
objects.
If there are fully typed parameters, allow those to be passed in
unboxed, and generate a wrapper function that takes those arguments as
boxes and unboxes them.

Generic <-> Specialized
+++++++++++++++++++++++
Generally generic code can call specialized functions or methods of objects
of known type directly. Another instance of this occurs when instances
originate from specialized classes. Consider populating a list of an
int, string and float. Generic wrappers are generated around the specialized
methods, and a vtable is populated. The wrappers are implemented as follows:

.. code-block:: python

    @gjit('a -> a -> bool')
    def wrapper_eq(int_a, int_b):
        return box(specialized_eq(unbox(a), unbox(b)))

We further need to generate properties that box specialized
instance data on read, and unbox boxed values on write.

Specialized <-> Generic
+++++++++++++++++++++++
Generally specialized code can call generic functions or methods of objects
that are not statically known (e.g. "an instance of A or some subtype").
The specialized code will need to box arguments in order to apply such a
function. This means that generic wrapper classes need to be available for
specialized code. For parameterized types this means we get a different
generic class for every different combination of parameters of that type.

We may further allow syntax to store generic objects in specialized classes,
e.g.

.. code-block:: python

    @jit
    class MyClass(object):
        layout = [('+A[]', 'obj')]

Which indicates we can store a generic instance of `A` or any subtype in the
`obj` slot.

Specialized <-> Specialized
+++++++++++++++++++++++++++
Static dispatch everywhere.


Variance
========
Finally, we return to the issue of variance. For now we disallow subtype bounds
on type variables of parameterized types, allowing only invariance on
parameters. This avoids the read/write runtime checks that would be needed to
guarantee type safety, as touched on in :ref:`core`.

For bonus points, we can allow annotation of variance in the type syntax,
allowing more generic code over containers without excessive runtime type
checks:

.. code-block:: python

    @jit('List[+-a]')
    class List(object):

        @jit('List[a] -> int64 -> +a)
        def __getitem__(self, idx):
            ...

        @jit('List[a] -> int64 -> -a -> void)
        def __setitem__(self, idx, value):
            ...

This means that if we substitute a `List[b]` for a `List[a]`, then for a read
operations we have the constraint that `b <: a`, since `b` can do everything
`a` does. For a write operation we have that `a <: b`, since if we are to write
objects of type `a`, then the `b` must not be more specific than `a`.

This means the type checker will automatically reject any code that does not
satisfy the contraints originated by the operations used in the code.
