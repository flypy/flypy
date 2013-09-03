# Numba the Next Iteration

# Last Month

    * Bytecode frontend (npm2)
    * Typing algorithm
    * Backend (pykit)

# Bytecode -> untyped SSA

Reduced syntax:

    * special methods
    * (compile-time) overloading
    * low-level control flow

Output:

    e = x                           variable
      | x = a                       assignment
      | const(x)                    constants
      | x.a                         attribute
      | f(x)                        application
      | jump/ret/exc_throw/...      control flow
      | (T) x                       conversion

# Typing algorithm (Cartesian Product Algorithm)

Generate simplified constraint network (dataflow graph):

    e = x.a             attribute
      | f(x)            application
      | flow(a, b)      data flow

Typing is performed on this graph.

# Backend (pykit)

* Typed or untyped SSA
* Inlining, loop detection, data flow, ...
* Lowering passes, LLVM backend
* Tools: C front-end (testing), IR interpreter, verifier, ...


# Coming Months:

# New language

# DataShape array syntax:

Single type system:

    T, T, int32         -> square 2D array of int32
    A, B, A : numeric   -> 2D array of numbers
    ..., double         -> ND-array of doubles

# Generic functions, polymorphic type signatures, overloading:

```python
@overload('A, B, dtype : floating -> dtype -> A, B, dtype')
def __add__(self, scalar):
    ...
```

# User-defined types, parameterized types:

```python
@jit('Array[T]', immutable=True)
class Array(object):
    layout = Struct([('data', 'T *')])
```

# Generator fusion under composition:

```python
def map(f, xs):
    for x in xs:
        yield f(x)
```

* General model
* Pythonic
* Minimal number of rewrite rules

# Zero-cost abstraction:

* control over inlining, specialization, mutability, stack-allocation

# Github:

https://github.com/ContinuumIO/numba-nextgen