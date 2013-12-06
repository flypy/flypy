from pykit.analysis import callgraph

def run(func, env):
    envs = env["numba.state.envs"]
    dependences = [d for d in callgraph.callgraph(func).node]
    env["numba.state.dependences"] = dependences

