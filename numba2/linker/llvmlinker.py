"""
Defines a LLVM specific linker.
"""
from __future__ import print_function
import llvm.core
import llvm.passes

# Symbols with the same name and definition can be merged during linkage.
LINKAGE_DEFAULT = llvm.core.LINKAGE_WEAK_ODR

class LinkerUnit(object):
    def __init__(self, module):
        self.module = module
        self.symbols = frozenset(_find_all_symbols(self.module))
        self.declarations = frozenset(_find_all_undefined(self.module))
        self.definitions = self.symbols - self.declarations

    def correct_linkage(self):
        # TODO deal with other global values
        for func in self.module.functions:
            func.linkage = LINKAGE_DEFAULT

    def link_in(self, other):
        self.module.link_in(other.module, preserve=True)

    def __repr__(self):
        return '<LinkUnit: module %s>' % self.module.id


def _find_all_symbols(module):
    """Returns all global symbols in the module
    """
    return [f.name for f in module.functions]

def _find_all_undefined(module):
    # TODO deal with other global values
    return [f.name for f in module.functions if f.is_declaration]


class Linker(object):
    """A LLVM Linker

    Linking is transitive so that symbols are propagated.
    Transitive linking can reduce the number of module requires to be linked.

    For example:

        A defines foo.
        B defines bar.
        bar references foo.
        C references foo and bar.
        If B exposes foo, C need to link to B only.
        Thus, reducing the need to link to A.
    """

    def __init__(self, result):
        """
        Modules are linked into result

        Parameters
        -----------

        result: llvm Module
        """
        self.result = LinkerUnit(result)
        self.modules = []

    def add_module(self, module):
        """
        Parameters
        ----------

        module: llvm Module
            Add a module to be linked into ``self.result``
        """
        self.modules.append(LinkerUnit(module))

    def link(self):
        """Perform the linking
        """
        if not self.modules:
            return self.result.module

        modules = set(self.modules)
        missing = set(self.result.declarations)

        while modules and missing:
            scoreboard = []
            for m in modules:
                avail = m.definitions & missing
                rank = len(avail)
                scoreboard.append((rank, avail, m))

            ordered = sorted(scoreboard)

            best = ordered[-1]
            lu_best = best[2]
            provided = best[1]
            missing -= provided
            self.result.link_in(lu_best)
            modules.remove(lu_best)

            for rank, _, m in ordered:
                if rank == 0:
                    # Remove zero ranked modules
                    modules.remove(m)
                else:
                    break

        self.result.correct_linkage()
        post_link_optimize(self.result.module)


def post_link_optimize(mod, inline=200):
    """Inline small functions
    """
    pm = llvm.passes.PassManager.new()
    pmb = llvm.passes.PassManagerBuilder.new()
    pmb.opt_level = 1      # allow minimum optimization to reduce code size
    pmb.use_inliner_with_threshold(inline)
    pmb.populate(pm)
    pm.run(mod)
