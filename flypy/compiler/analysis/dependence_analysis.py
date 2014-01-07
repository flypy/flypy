from pykit.analysis import callgraph

def run(func, env):
    envs = env["flypy.state.envs"]
    dependences = [d for d in callgraph.callgraph(func).node]
    env["flypy.state.dependences"] = dependences

