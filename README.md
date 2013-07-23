Next iteration of Numba
=======================

In order to get a more sustainable compiler development process,
and a more extensible and usable language, I believe we need
the following features:

    1) Support for multi-stage programming
    2) Structs with methods
    3) Overriding of typing semantics through typeof() convert() and unify()
    4) Multiple-dispatch or traits for methods typeof(), convert(), etc

1. MSP
======
Quicly assemble program fragments

2. Structs with methods
=======================
I believe this one is critical. The advantage of structs with methods is that
inheritance is disallowed. This means that the callees are static, and can be
inlined (something we will exploit aggresively). Furthermore, even when not
inlined the calls are less expensive than virtual calls.

This means that we can write our runtime in this way. This means the compiler
needs to support the following types natively:

    - int
    - float
    - pointer
    - struct (with optional methods and properties)

Anything else is written in the runtime:

    - range
    - complex
    - array
    - string/unicode
    - etc

This means we can easily experiment with different data representations and
extend functionality. For instance we can override the native integer multiply
to check for overflow, and raise an exception or issue a warning, or convert to
a BigInt.

Approach
========

Write a new pipeline that performs a few transformations:

    - Rewrite all operations using the __*__ equivalents
    - Add some more __*__ methods, like __assign__, __is__, etc

Type inference then simply infers all operations over method calls on the structs,
and on basic native objects.
