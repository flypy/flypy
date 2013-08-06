Typing
======
This section discusses typing for numba. There is plenty of literature on type
inference, most notable is the Damas-Hindley-Milner Algorithm W. for
lambda calculus [1]_, and an extension for ML. The algorithm handles
let-polymorphism (a.k.a. ML-polymorphism), a form of parametric polymorphism
where type variables themselves may not be polymorphic. For example, consider:

.. code-block:: python

    def f(g, x):
        g(x)
        g(0)

We can call ``f`` with a function, which must accept ``x`` and an value of
type int. Since ``g`` is a monotype in ``f``, the second call to ``g``
restricts what we accept for ``x``: it must be something that promotes with
an integer. In other words, the type for ``g`` is ``a -> b`` and not
``∀a,b.a -> b``.

Although linear in practise, the algorithm's worst case behaviour is
exponential ([2]_), since it does not share results for different function
invocations. The cartesian product algorithm ([3]_) avoids this by sharing
monomorphic template instantiations. It considers all possible
receivers of a message send, and takes the union of the results of all
instances of the cartesian product substitution. They do not seem to address
circular type dependencies (?), where the receiver can change based on the input
types:

.. code-block:: python

    def f(x):
        for i in range(10):
            x = g(x)

leading to

.. code-block:: llvm

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

If we assign type variables throughout the function first, we get the following
constraints:

::

    [ X1 = Union(X0, X2), G = X1 -> T2 , X2 = T2 ]

We can represent a function as a set of overloaded signatures. However,
the function application is problematic, since we send X1 (which will be
assigned a union type). This will lead to exponential behaviour (there are
2^N subsets for N types). Instead we can expand polymorphic calls with the
cartesian product and generate new constraints during unification. This
essentially constitutes a fix-point on polymorphic call-sites with circular
data-dependences. The only way to make sure this terminates is to use an
additional cache that excludes previously considered types for each of the
input types in the union.

Subtyping
=========
We intend to implement subtyping in the runtime through inheritance. When
a class B inherits from a class A, we check for a compatible interface for
the methods (argument types are contravariant and return types covariant).
When typing, the only thing we need to implement are coercion and unification:

    Type B coerces to type A if B is a subtype of A
    Type A coerces to type B if B is a subtype of A with a runtime check only

Then class types A and B unify iff A is a subtype of B or vice-versa. The
result of unification is always the supertype.

Parameterization
================
Types can only be parameterized by traits and non-class user-defined or
built-in types. This allows us to avoid dealing with covariant and
contravariant issues. The exception is function types, for which we have
built-in rules.

References
==========
.. [1]: A Theory of Type Polymorphism in Programming Languages, Milner
.. [2]: A proof of correctness for the Hindley-Milner type inference algorithm
.. [3]: The Cartesian Product Algorithm
