Fusion
======
We want to fuse operations producing intermediate structures such as lists
or arrays. Fusion or deforestation has been attempted in various ways, we
will first cover some of the existing research in the field.

Deforestation
-------------

build/foldr
+++++++++++
Rewrite rules can be used to specify patterns to perform fusion ([1]_,
[2]_, [3]_), e.g.::

    map f (map g xs) = map (f . g) xs

The dot represents the composition operator. To avoid the need for a
pattern for each pair of operators, we can express fusable higher-order functions
in terms of a small set of combinators. One approach is build/foldr, where
``build`` generates a list, and ``foldr`` (reduce) consumes it ([3]).
Foldr can be defined as follows:

.. code-block:: haskell

    foldr f z []     = z
    foldr f z (x:xs) = f x (foldr f z xs)

``build`` is the dual of ``foldr``, instead of reducing a list it generates
one. Using just build and foldr, a single rewrite rule can be used for
deforestation:

    foldr k z (build g) = g k z

This is easy to understand considering that build generates a list, and foldr
then consumes it, so there's no point in building it in the first place.
Build is specified as follows:

.. code-block:: haskell

    build g (:) []

This means ``g`` is applied to the ``cons`` constructor and the empty list.
We can define a range function (``from`` in [3]) as follows:

.. code-block:: haskell

    range a b = if a > b then []
                else a : (range (a + 1) b)

Abstracting over cons and nil (the empty list) [3], we get:

.. code-block:: haskell

    range' a b = \ f lst -> if a > b then lst
                            else f a (range' (a + 1) b f lst)

It's easy to see the equivalence to ``range`` above by substituting
``(:)`` for ``f`` and ``[]`` for lst. We can now use ``range'`` with
``build`` ([3]):

.. code-block:: haskell

    range a b = build (range' a b)

Things like ``map`` can now be expressed as follows ([3]):

.. code-block:: haskell

    map f xs = build (\ cons lst -> foldr (\ a b -> cons (f a) b) lst xs)

However, some functions cannot be expressed in this framework, like zip ([4]_).

Streams
+++++++
Another major approach is based on stream fusion ([4]_, [5]_).
It expresses the higher-order functions in terms of streams ([4]_):

.. code-block:: haskell

    map f = unstream . map' f . stream

``unstream`` converts a stream back to a list, and stream converts a list to
a stream. Under composition, like ``map f (map g xs)``, we get
``unsteam . map' f . stream . unsteam . map' g . stream``. The fusion then
relies on eliminating the composition of ``stream`` with ``unstream``:

    stream (unstream s) = s

A stream consists of a stepper function and a state. Stepper functions
produce new step states. The states are ``Done``, ``Yield`` or ``Skip``.
``Done`` signals that the stream is consumed, ``Yield`` yields a new value
and state, and ``Skip`` signals that a certain value needs to be skipped
(for things like filter).

Let's see this in action ([5]):

.. code-block:: haskell

    stream :: [a] -> Stream a
    stream xs0 = Stream next xs0
        where
            next []     = Done
            next (x:xs) = Yield x xs

This converts a list to a Stream. It constructs a Stream with a new stepper
function  ``next`` and the initial state (the given list). The ``next``
stepper function produces a new step state every time it is called.
Streams can be consumed as follows:

.. code-block:: haskell

    map f (Stream next0 s0) = Stream next s0
        where
            next s = case next0 s of
                Done        -> Done
                Skip s'     -> Skip s'
                Yield x s'  -> Yield (f x) s'

Here we specify a new stepper function ``next`` that, given a state, advances
the stream it consumes with the new state, and yields new results. It wraps
this stepper function in a new stream. [5]_ further extends this work to allow
operation over various kinds of streams:

    - Chunked streams for bulk memory operations
    - Vector (multi) streams for SIMD computation
    - Normal streams that yield one value at a time

It bundles the various streams together in a product type. The idea is that
all streams are available at the same time. Hence a producer can produce
in the most efficient way, and the consumer can consume in the
most efficient way. These concepts don't always align, in which case fallbacks
are in place, for instance a chunked stream can be processed as a scalar
stream, or vice-versa. In addition to inlining and other optimizations it
relies heavily on call-pattern specialization ([6]), allowing the
compiler to eliminate pattern matching of consumer sites.

Fusion in flypy
---------------
The concept of a streams encapsulating a state and a stepper function is akin
to iterators in Python, where the state is part of the iterator and the
stepping functionality is provided by the ``__next__`` method. Although
iterators can be composed and specialized on static callee destination (
the __next__ method of another iterator), they are most naturally expressed
as generators::

    def map(f, xs):
        for x in xs:
            yield f(xs)

The state is naturally captured in the generator's stack frame. To allow
fusion we need to inline producers into consumers. This is possible only
if we can turn the lazy generator into a non-lazy producer, i.e. the consumer
must be the only consumer of the result. This introduces a restriction:

    * The generator may not be stored, passed to other functions
      or returned. We can capture this notion by having ``iter(generator)``
      create a ``stream``, and disallowing the rewrite rule
      ``stream (unstream s) = s`` to trigger when the ``unstream s`` has
      multiple uses.

      This means the value remains `unstreamed` (which itself is lazy, but
      effectively constitutes a fusion boundary).

Since we can express many (all?) higher-order fusable functions as generators,
we have a powerful building block (in the same way as the previously outlined
research methods), that will give us rewrite rules for free.
I.e., we will not need to state the following:

.. code-block:: python

    map(f, map(g, xs)) = map(f . g, xs)

since this automatically follows from the definition of map:

.. code-block:: python

    @signature('(a -> b) -> Stream a -> Stream b')
    def map(f, xs):
        for x in xs:
            yield f(x)

Below we discuss how to realize our fusion scheme.

Generator Fusion
----------------
We start by assuming a restrictor generator model, where all the producer and
consumer sites are statically identified.

For instance, this places a restriction on how the generator can be
instantiated, which must be from a statically identified function or
method receiver.

It places a restriction on how the generator is consumed, which must also be
statically known, in order to specialize consumer sites to producers.

In the most straightforward situation we have a single producer site and a
single consumer site. In more complicated cases, we can have multiple
producer sites (e.g. multiple yields) or multiple consumer sites (e.g.
multiple loops over a single generator).

The multiple producer problem is complicated by the consumer not statically
knowing which dispatch site the next item retrieval will have to jump to.

In addition to any producer sites, there is an additional generator entry
point, which is the start of the function. Additional exit points of the
producer are the return points (including the end of the function). The
exit points correspond to the points where a StopIteration is conceptually
raised.

We start by addressing the most general case we want to handle, which is
a statically identified generator producer with any number of producer sites
and and any number of local consumer sites.

The General Case
++++++++++++++++
The general case is P producer sites, 1 entry point, and C consumer sites.
Specializing the consumer to the producer can mean the following:

    1. track two tokens, which correspond respectively to the currently
       active producer and consumer sites. Retrieving a new item means using
       a dispatch table that jumps to the right program in the generator, the
       token for which we'll call a producer token.

       Yielding means dispatching to any of the C consumer sites, using what
       we'll call the consumer token. These tokens can be shared in a stack
       variable between the consumer and producer, the producer updates the
       producer token and the consumer the consumer token.

    2. Specialize the consumer to the producer, i.e. for each producer site P,
       generate a special consumer C' that only jumps only to a specific producer
       program point (a specific yield). All consumers copies need to share stack
       variables. This eliminates the token used to dispatch to the right
       producer program point. This requires P + 1 specializations of C.

    3. Specialize the producer to the consumer, that is, for each consumer site
       generate a producer P' that jumps to C only. All these producers must
       share the same stack variables. We need C copies of the producer. This
       eliminates the dispatch used to dispatch back to the right consumer
       site. This requires C specializations of P.

    4. Specialize all producers to all consumers and all consumers to all
       producers. This effectively flattens both dispatch tables, and requires
       (P + 1) * C specializations.


We noted P + 1 specializations to incorporate the entry point as a special
point. The entry point can be eliminated from consideration if we have a
`for` loop that consumes the result:

.. code-block:: python

    for i in produce(x):
        ...

We can eliminate the entry point from consideration since we statically know
where this entry point must be inserted in the control flow: it must be
right at the call to `next()` in the loop header block. Further, since we
are the only consumer of the generator, we have P * 1 specializations in the
latter case. The fully degenerate case also has additionally one yield point:

.. code-block:: python

    def produce(x):
        sum = 0
        for i in range(x):  # produce.header
            yield sum       # produce.body0
            sum += i        # produce.body1

    for i in produce(x):
        f(i)                # consumer.body

     # This becomes

     sum = 0
     for i in range(x):     # produce.header
         i = sum            # produce.body0
         f(i)               # consumer.body
         sum += i           # producer.body1

The transitions between the different basic blocks above are actually jumps.
Since the jumps in the body are all unconditional, they are simply eliminated
by the control flow simplier pass.

Note that the third pattern, where we specialize producers to consumers, is a
simple instance of inlining the `next()` call a traditional generator
implementation, which has a dispatch table to jump to the different resume
points in the generator. Hence our first pattern is simply a generalization
of inlining, where we inline a function only once and subsequently connect
arguments to parameters by assigning local stack variables, with a dispatcher
at the end to resume at the right point in the consumer.

.. digraph:: inlining

    P0 -> Cdispatch
    P1 -> Cdispatch
    PN -> Cdispatch

    Cdispatch -> C0
    Cdispatch -> C1
    Cdispatch -> CN

    C0 -> Pdispatch
    C1 -> Pdispatch
    CN -> Pdispatch

    Pdispatch -> P0
    Pdispatch -> P1
    Pdispatch -> PN

Currently, we only handle the degenerate case of generator inlining, with
a single consumer and a single producer point, and further with a statically
known entry point. This looks as follows:

.. code-block:: python

    for i in produce(x):
        f(i)

However, when we use a while loop the entry point insertion cannot be determined
straightforwardly, since we may have an arbitrary condition on our loop:

.. code-block:: python

    while something_complicated:
        i = next(g)
        f(i)

Hence the length of our loop is no longer dictated by the generator. It is now
dictated by both the generator and the loop condition.

To handle the generation of efficient code while retaining generality, it
seems desirable to implement generic generators (i.e. case 3). Through
straightforward inlining, combined with stack-allocation of the activation
record, scalar replacement of aggregates and control flow simplification
we are very close to the optimal form we desire.

The only runtime overhead introduced is the producer dispatch at entry to
the producer! We can simply eliminate this dispatch by implementing an
optimization that can specialize control flow paths. This is a straightforward
application of partial evaluation, where we categorize only the producer token
as static. Our special case is really a value-specialization for a single
variable of a single basic block. This has the effect of simply unrolling
the while loop one iteration.

To summarize, if we implement the general form of generators we can
derive some of our cases above:

    1. This is simply de-duplicated inlining
    2. This is de-duplicated inlining combined with the value specialization
       of the producer token (case 1 + value specialization)
    3. This is regular inlining, except we now share stack variables between
       all inlined versions
    4. This is regular inlining with stack variable sharing combined with
       value specialization

Note further that the inlining with stack variable sharing is a special case
of deduplicated inlining with value specialization, this time on the consumer
token. This means the last case really means inlining the generator once,
and subsequently value-specializing both the producer and consumer tokens.

Of course, we must be careful with specialization. In practice it is likely
that the number of consumer sites is only one, but multiple producer sites
(multiple yield statements is common). Recursive application of this
optimization will generate an exponential amount of code in the depth of
the generator chain.

.. code-block:: python

    for i in f(f(f(x))):
        print i

Will result in a function with 8 print statements. However, it is not always
possible to generate static code without multiple yields, consider
the concatenation function:

.. code-block:: python

    def concat(xs, ys):
        for x in xs:
            yield x
        for y in ys:
            yield ys

This function has two yields. If we rewrite it to use only one yield:

.. code-block:: python

    def concat(xs, ys):
        for g in (xs, ys):
            for x in g:
                yield x

We have introduced dynamicity that cannot be eliminated without specialization
on the values (i.e. unrolling the outer loop, yielding the first
implementation). This not new in any way, it is inherent to inlining and
value specialization, and we treat it as such (by simply using sensible
thresholds). Crossing the threshold simply means temporaries are not
eliminated -- in this case this means generator "cells" remain.

2. Specialization
+++++++++++++++++
Specialization follows from inlining, there are two cases:

    - internal terms
    - boundary terms
    - ``stream (unstream s)`` is rewritten, the result is fused

Internal terms are rewritten according to the ``stream (unstream s)`` rule.
What eventually follows at a boundary is a) consumption through a
user-written loop or b) consumption through the remaining unstream. In either
case the result is consumed, and the inliner will start inlining top-down
(reducing the terms top-down).

SIMD Producers
--------------
For simplicity we exclude support for chunked streams. Analogous to [5]_ we
can expose a SIMD vector type to the user. This vector can be yielded by a
producer to a consumer.

How then, does a consumer pick which stream to operate on? For instance,
zip can only efficiently be implemented if both inputs are the same, not if
one returns vectors and the other scalars (or worse, switching back and forth
mid-way):

.. code-block:: python

    def zip(xs, ys):
        while True:
            try:
                yield (next(xs), next(ys))
            except StopIteration:
                break

For functions like zip, which are polymorphic in their arguments, we can
simply constrain our inputs:

.. code-block:: python

    @overload('Stream[Vector a] -> Stream[Vector b] -> Stream[(Vector a, Vector b)]')
    @overload('Stream a -> Stream b -> Stream (a, b)')
    def zip(xs, ys):
        ...

Of course, this means if one of the arguments produces vectors, and the other
scalars, we need to convert one to the other:

.. code-block:: python

    @overload('Stream[Vector a] -> Stream a')
    def convert(stream):
        for x in stream:
            yield x

Which basically unpacks values from the SIMD register.

Alternatively, a mixed stream of vectors and scalars can be consumed. [5]_
distinguises between two vector streams:

    - a producer stream, which can yield Vector | Scalar
    - a consumer stream, where the consumer chooses whether to read vectors
      or scalars. A consumer can start with vectors, and when the vector stream
      is consumed read from the scalar stream.

A producer stream is useful for producers that mostly yield vectors, but
sometimes need to yield a few scalars. This class includes functions like
concat that concatenates two streams, or e.g. a stream over a multi-dimensional
array where inner-contiguous dimensions have a number of elements not 0 modulo
the vector size.

A consumer stream on the other hand is useful for functions like zip, allowing
them to vectorize part of the input. However, this does not seem terribly
useful for multi-dimensional arrays with contiguous rows, where it would only
vectorize the first row and then fall back to scalarized code.

However, neither model really makes sense for us, since we would already
manually specialize our loops:

.. code-block:: python

    @overload('Array a 2 -> Stream a')
    def stream_array(array, vector_size):
        for row in array:
            for i in range(len(row) / vector_size):
                yield load_vector(row.data + i * 4)

            for i in range(i * 4, len(row)):
                yield row[i]

This means code consuming scalars and code consuming vectors can be matched
up through pattern specialiation (which is not just type-based branch pruning).


To keep things simple, we will stick with a producer stream, yielding either
vectors or scalars. Consumers then pattern-match on the produced values,
and pattern specialization can then switch between the two alternatives:

.. code-block:: python

    def sum(xs):
        vzero = Vector(zero)
        zero = 0
        for x in xs:
            if isinstance(x, Vector):
                vzero += x
            else:
                zero += x
        return zero + vreduce(add, vzero)

To understand pattern specialization, consider ``xs`` is a ``stream_array(a)``.
This results in approximately the following code after inlining:

.. code-block:: python

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

It is now easy to see that we can eliminate the second pattern in the first
loop, and the first pattern in the second loop.

References
----------
.. [1]_ Deforestation: Transforming programs to eliminate trees
.. [2]_ Playing by the Rules: Rewriting as a practical optimisation technique in GHC
.. [3]_ A short-cut to deforestation
.. [4]_ Stream Fusion: From Lists to Streams to Nothing at All
.. [5]_ Exploiting Vector Instructions with Generalized Stream Fusion
.. [6]_ Call-pattern Specialisation for Haskell Programs
