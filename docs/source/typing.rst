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
instances of the cartesian product substitution. The paper does not seem to
address circular type dependencies, where the receiver can change based on
the input types:

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

However, this can be readily solved through fix-point iteration. If we assign
type variables throughout the function first, we get the following
constraints:

::

    [ X1 = Union(X0, X2), G = X1 -> T2 , X2 = T2 ]

We can represent a function as a set of overloaded signatures. However,
the function application is problematic, since we send X1 (which will be
assigned a union type). WIthout using the cartesian product this would lead
to exponential behaviour since there are 2^N subsets for N types.

Type inference in Numba
=======================
We use the cartesian product algorithm on a constraint network based on the
dataflow graph. To understand it, we need to understand the input language.
Since we put most functionality of the language in the user-domain, we
desugar operator syntax through special methods, and we further support
overloaded functions.

The front-end generates a simple language that can conceptually be described
through the syntax below::

    e = x                           variable
      | x = a                       assignment
      | const(x)                    constants
      | x.a                         attribute
      | f(x)                        application
      | jump/ret/exc_throw/...      control flow
      | (T) x                       conversion

As you'll notice, there are no operators, loops, etc. Control flow is encoded
through jumps, exception raising, return, etc. Loops can be readily detected
through a simple analysis (see pykit/analysis/loop_detection.py).

We take this input grammar and generate a simpler constraint network, that
looks somewhat like this::

    e = x.a             attribute
      | f(x)            application
      | flow(a, b)      data flow

This is a directed graph where each node classifies the constraint on the
inputs. Types propagate through this network until no more changes can take
place. If there is an edge ``A -> B``, then whenever ``A`` is updated, types
are propagated to ``B`` and processed according to the constraint on ``B``.
E.g. if ``B`` is a function call, and ``A`` is an input argument, we analyze
the function call with the new values in the cartesian product.

Coercions
=========
Coercions may happen in two syntactic constructs:

    * application
    * control flow merges (phi nodes)

For application we have a working implementation in Blaze that determines
the best match for polymorphic type signatures, and allows for coercions.
For control flow merges, the user can choose whether to promote values, or
whether to create a sum-type. A post-pass can simply insert coercions where
argument types do not match parameter types.

Subtyping
=========
We intend to support subtyping in the runtime through python inheritance. When
a class B inherits from a class A, we check for a compatible interface for
the methods (argument types are contravariant and return types covariant).
When typing, the only thing we need to implement are coercion and unification:

    Type B coerces to type A if B is a subtype of A
    Type A coerces to type B if B is a subtype of A with a runtime check only

Then class types A and B unify iff A is a subtype of B or vice-versa. The
result of unification is always the supertype.

Finally, parameteric types will be classified invariant, to
avoid unintended mistakes in the face of mutable containers. Consider e.g.
superclass ``A`` and subclass ``B``. Assume we have the function that accepts an
argument typed ``A[:]``. If we treat the dtype as covariant, then we may
pass an array ``B[:]`` for that argument. However, the code can legally
write ``A``s into the array, violating the rule that we can only assign
subtypes. The problem is that reading values is covariant, whereas writing
is contravariant. In other words, the parameter must be covariant as well as
contravariant at the same time, which is only satisfied when ``A = B``.

The exception is maybe function types, for which we have built-in variance
rules.

Parameterization
================
Types can only be parameterized by variables and user-defined or
built-in types. Type variables may be constrained through traits (type
sets can readily be constructed by implementing (empty) traits).

References
==========
.. [1]: A Theory of Type Polymorphism in Programming Languages, Milner
.. [2]: A proof of correctness for the Hindley-Milner type inference algorithm
.. [3]: The Cartesian Product Algorithm
