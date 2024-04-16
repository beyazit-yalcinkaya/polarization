import numpy as np
import networkx as nx
import gymnasium as gym
from copy import deepcopy

# An observation is a networkx graph with static and dynamic node attributes.
# An action is an edge on the same networkx graph.

class Population(gym.Env):

    def __init__(self, n=4, m=10, k=2, network=None):
        self.n = n # Number of dynamic features
        self.m = m # Upper bound for dynamic features
        self.k = k # Upper bound for static features
        self.network = network
        self.state = None

    def reset(self):
        self.state = deepcopy(self.network)
        nx.set_node_attributes(self.state, {node: {"static": np.random.randint(self.k), "dynamic": np.random.randint(self.m, size=(self.n,))} for node in self.state.nodes})
        return self.state

    def step(self, action):
        source, target = action

        differences = np.array([1 if self.state.nodes[source]["dynamic"][i] != self.state.nodes[target]["dynamic"][i] else 0 for i in range(self.n)])

        if sum(differences) == 0:
            return self.state
        
        differences = differences / sum(differences)
        dimension = np.random.choice(self.n, p=differences)

        self.state.nodes[target]["dynamic"][dimension] = self.state.nodes[source]["dynamic"][dimension]
        return self.state

if __name__ == "__main__":
    network = nx.grid_graph(dim=(10, 10))
    env = Population(network=network)
    obs = env.reset()
    for e in obs.edges:
        env.step(e)
