# -*- coding: utf-8 -*-

r"""
Persistent cache for IR.

Fine-grained Caching
====================

IR is cached at each stage. The key into the cache is a tuple (func, argtypes).
For `func` we hash on the code object and default argument values, the
values of any cell variables and the values of referenced globals.

Tables:

    Version
    =======
    version_id   py_version    flypy_version     llvmpy_version
    -------------------------------------------------------------
    0x0          '2.7.6'       '0.20.0.f298c5f'  '0.12.1.c199881'

    Functions
    =========
    func_id  version_id  module_name    qualified_name  code_blob  timestamp
    -------------------------------------------------------------------------
    0x1      0x0         'package.mod'  'MyCls.method'  '\x.....'  Mon 2013..

    Dependences
    ===========
    func_id     dep_func_id     stage
    ----------------------------------
    0x1         0x2             'llvm'

    IR
    ==
    func_id     stage           ir          env
    -------------------------------------------------
    0x1         'llvm'          '\x.....'   '\x.....'


This fine-grained, per-function based approach allows the compiler to
recompile only those parts that are needed. The `stage` column allows us
to serialize IR at different stages:

    Consider function `f` that uses part of some library. If `f` and all its
    dependences are unmodified and previously compiled, we can immediately
    load the LLVM IR for `f`. However, if `f` changes, we need to re-compile
    `f` and have the IR available for its dependences. Hence to provide
    fast compilation times we also need to cache code from intermediate
    stages.

Each code_blob in the `Functions` table is an individual Module containing
the IR.

Coarse-grained Caching
======================

A function may have many (indirect) dependences, for instance code allocating
a heap object will depend on the garbage collector. To make dependence checks
and loading more efficient, we can pursue a more coarse-grained dependence
model, based on modules:

    Module
    ======
    module_id   filename       modification_time
    --------------------------------------------
    0x5         '/home/...'    Mon 2013..

    ModuleIR
    ========
    module_id    code_block
    -----------------------
    0x5          '\x.....'

    ModuleDeps
    ==========
    module_id   dep_module_id    stage
    -----------------------------------
    0x5         0x6              'llvm'

In the common case, where library code is precompiled and remains unmodified,
the compiler can load in entire modules and link them together quickly. This
will fall back to the fine-grained approach where-ever code or files are
modified.

Difficulties
------------
The main difficulties implementing the coarse-grained part of the cache is
that functions are specialized on demand, and hence it is unclear what a
module is at any given time.

So when module M1 depends on module M2, it really depends on a set of
specialized functions in the current incarnation of flypy. Later, polymorphic
functions may be generated, in which case it becomes clearer what a module
constitutes. Finally, function generation (e.g. through closures returning new
jitted functions) may complicate what a module is.

A module dependence is then only satisfied when the set of (type- and value-)
specialized functions needed by the dependent module are provided by the
supporting module.


Garbage Collection
==================

To ensure the database does not keep growing forever, we need to garbage
collect functions that are no longer needed. This can be based on the
recorded modification times. For bonus points, infrequently used
specializations may be pruned in order to properly manage usage of installed
libraries with flypy code. An additional table may be used:

    Usage
    =====
    func_id     last_used
    ---------------------
    0x1         Mon Dec..

IR Portability
==============

Initially, we will address only locally-shared code. This means IR needs
to be portable across process invocations, which means that:

    * Constant pointers must be rewritten in the IR

This further implies that non-primitive constants (e.g. constants of
heap-allocated objects) must be reconstructed properly. This also means that
we need to incorporate potentially arbitrary object graphs in our hash. For
now we will simply avoid caching functions dependent on such state.

Literals and Constants
----------------------
To maximize portability, we should make sure to generate IR for literals in
code (such as tuples, strings, slice objects, etc), to not simply inject
pointers to constants, but rather to generate code that calls constructors.
Static buffers (such as strings) can be allocated as module globals in the IR.

External Symbols
----------------
FFI calls are a special case of handling constants. We should ensure we
use global symbols in the IR and record the library a symbol is from. We
can then attach to global symbols a runtime address loaded as a runtime
library.

We may need an additional table:

    Symbols
    =======
    func_id     ir_symname     library      lib_symname
    ---------------------------------------------------
    0x1         'printf'       'libc.so'    'printf'
"""

from __future__ import print_function, division, absolute_import

import os
from os.path import expanduser, join
import sys
import sqlite3 as db

import flypy
from . import keys, serializers

import pykit
import llvm

db_file = expanduser(join('~', 'example.db'))

def open_cache(db_file):
    cache = connect_cache(db_file)
    cache.setup_tables()
    serializers.register(cache)
    return cache

def connect_cache(db_file):
    conn = db.connect(db_file)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.commit()
    return CodeCache(db_file, conn)

def delete_cache(db_file):
    try:
        os.remove(db_file)
    except EnvironmentError:
        pass

class CodeCache(object):

    def __init__(self, db_file, conn):
        self.db_file = db_file
        self.conn = conn
        self.cursor = conn.cursor()
        self._version_id = None

        self.serializers = {}
        self.deserializers = {}

    def setup_tables(self):
        """Initialize db tables"""
        self.cursor.execute("""
            CREATE TABLE Version (
                id INTEGER PRIMARY KEY autoincrement NOT NULL,
                versions TEXT NOT NULL
            )""")

        self.cursor.execute("""
            CREATE TABLE Function (
                id INTEGER PRIMARY KEY autoincrement NOT NULL,
                version_id INTEGER NOT NULL,
                module_name TEXT NOT NULL,
                qname TEXT NOT NULL,
                key BLOB NOT NULL,
                mtime TIMESTAMP NOT NULL,

                FOREIGN KEY(version_id) REFERENCES Version(id) ON DELETE CASCADE
            )""")

        self.cursor.execute("""
            CREATE TABLE Dependence (
                func_id INTEGER NOT NULL,
                dep_func_id INTEGER NOT NULL,
                stage TEXT NOT NULL,

                FOREIGN KEY(func_id) REFERENCES Function(id) ON DELETE CASCADE,
                FOREIGN KEY(dep_func_id) REFERENCES Function(id) ON DELETE CASCADE
            )""")

        self.cursor.execute("""
            CREATE TABLE Code (
                func_id INTEGER NOT NULL,
                stage TEXT NOT NULL,
                symname TEXT NOT NULL,
                ir BLOB NOT NULL,
                env BLOB NOT NULL

                -- FOREIGN KEY(func_id) REFERENCES Function(func_id) ON DELETE CASCADE
            )""")

        self.conn.commit()

    def add_serializers(self, stage, serializer, deserializer):
        assert stage not in self.serializers, stage
        self.serializers[stage] = serializer
        self.deserializers[stage] = deserializer

    def lookup(self, py_func, argtypes, stage):
        """
        Look up a cached version for (py_func, argtypes) for the given
        compilation phase.
        """
        func_id = self.func_id(py_func, argtypes)
        if func_id is None:
            return None

        self.cursor.execute("""
            SELECT symname, ir, env FROM Code
            WHERE (func_id = ? AND stage = ?)""", (func_id, stage))
        result = fetchone(self.cursor)

        if result:
            symname, ir_blob, env_blob = map(str, result)

            # De-serialize
            deserializer = self.deserializers[stage]
            mod, func = deserializer.deserialize_code(ir_blob, symname, stage)
            env  = deserializer.deserialize_env(env_blob, stage)
            return mod, func, env
        else:
            return None

    def insert(self, py_func, argtypes, stage, code, env, mtime):
        """
        Cache IR (`code`) and an environment (`env`) for (py_func, argtypes)
        for the given phase.
        """
        func_id = self.func_id(py_func, argtypes)

        # -- Insert (py_func, argtypes) in Function if not present -- #
        if func_id is None:
            self.insert_func(py_func, argtypes, env, mtime)
            func_id = self.func_id(py_func, argtypes)
        else:
            assert self.lookup(py_func, argtypes, stage) is None

        assert func_id is not None
        symname = code.name

        # -- Persist IR in Code -- #
        serializer = self.serializers[stage]
        code_blob = serializer.serialize_code(code, symname, stage)
        env_blob  = serializer.serialize_env(env, stage)

        self.cursor.execute("""
            INSERT INTO Code
            VALUES (?, ?, ?, ?, ?)""",
            (func_id, stage, symname,
             code_blob,
             env_blob))
        self.conn.commit()

    @property
    def version_id(self):
        """
        Get the current version identifier for our compiler infrastructure.
        """
        if self._version_id is None:
            result = get_version_id(self.cursor)
            if result is None:
                update_version_id(self.conn)
                result = get_version_id(self.cursor)
            self._version_id = result

        return self._version_id

    def func_id(self, py_func, argtypes):
        """
        Retrieve the function ID for this (py_func, argtypes) combination
        from the db.
        """
        # TODO: check modification times of each type implementation in
        # `argtypes`
        blob = self.blobify(py_func, argtypes)
        self.cursor.execute("""
            SELECT id FROM Function
            WHERE (version_id = ? AND key = ?)""", (self.version_id, blob))
        return fetchone(self.cursor)

    def insert_func(self, py_func, argtypes, env, mtime):
        self.cursor.execute("""
            INSERT INTO Function (version_id, module_name, qname,
                                  key, mtime)
            VALUES (?, ?, ?, ?, ?)
            """, (self.version_id, env['flypy.state.func_module'],
                  env['flypy.state.func_qname'],
                  self.blobify(py_func, argtypes), mtime))
        self.conn.commit()


    def blobify(self, py_func, argtypes):
        """Create a structural representation for the python function"""
        # TODO: Cache blobs
        return keys.make_code_blob(py_func, argtypes)


def get_version_id(cursor):
    cursor.execute("""SELECT id FROM Version WHERE (versions = ?)""",
                   (version_tuple(),))

    return fetchone(cursor)

def update_version_id(conn):
    conn.execute("""
        INSERT INTO Version (versions)
        VALUES  (?)""", (version_tuple(),))
    conn.commit()

#===------------------------------------------------------------------===
# Versioning
#===------------------------------------------------------------------===

def version_tuple():
    return str((py_version(), flypy_version(), pykit_version(),
                llvmpy_version(), llvm_version()))

def py_version():
    return tuple(sys.version_info[:3])

def flypy_version():
    return flypy.__version__

def pykit_version():
    return pykit.__version__

def llvmpy_version():
    return llvm.__version__

def llvm_version():
    return llvm.version

#===------------------------------------------------------------------===
# Utils
#===------------------------------------------------------------------===

def fetchone(cursor):
    result = cursor.fetchone()
    if result is not None and len(result) == 1:
        return result[0] # unpack single item
    return result # row (tuple) or None

#===------------------------------------------------------------------===
# Example
#===------------------------------------------------------------------===

def example_db():
    db_file = expanduser(join('~', 'example.db'))
    delete_cache(db_file)
    cache = open_cache(db_file)
    return cache