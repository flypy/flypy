# -*- coding: utf-8 -*-

"""
IR and native code serializers.
"""

from __future__ import print_function, division, absolute_import
import io
import pickle

import llvm.core as lc

#===------------------------------------------------------------------===
# interface
#===------------------------------------------------------------------===

class Serializer(object):
    """Serialize IR and a compilation environment"""

    def serialize_code(self, code, symname, stage):
        raise NotImplementedError

    def serialize_env(self, env, stage):
        raise NotImplementedError

class DeSerializer(object):
    """De-serialize IR and a compilation environment"""

    def deserialize_code(self, code_blob, symname, stage):
        raise NotImplementedError

    def deserialize_env(self, env_blob, stage):
        raise NotImplementedError

#===------------------------------------------------------------------===
# llvm
#===------------------------------------------------------------------===

class LLVMSerializer(object):

    def serialize_code(self, lfunc, symname, stage):
        bitcode = lfunc.module.to_bitcode()
        return bitcode

    def serialize_env(self, env, stage):
        return ""
        #return pickle.dumps(env)

class LLVMDeSerializer(object):

    def deserialize_code(self, bitcode, symname, stage):
        f = io.StringIO()
        f.write(bytes(bitcode, "latin-1"))
        f.seek(0)

        module = lc.Module.from_bitcode(f)
        return module, module.get_function_named(symname)

    def deserialize_env(self, env_blob, stage):
        #return pickle.loads(env_blob)
        return ""

#===------------------------------------------------------------------===
# Registration
#===------------------------------------------------------------------===

def register(cache):
    cache.add_serializers('llvm', LLVMSerializer(), LLVMDeSerializer())