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

    def _interact(self, target, neighbors):
        index = np.random.choice(len(neighbors))
        source = neighbors[index]

        differences = np.array([1 if self.state.nodes[source]["dynamic"][i] != self.state.nodes[target]["dynamic"][i] else 0 for i in range(self.n)])

        target_dynamic_features = np.array(self.state.nodes[target]["dynamic"])

        if sum(differences) == 0:
            return target_dynamic_features
        
        differences = differences / sum(differences)
        dimension = np.random.choice(self.n, p=differences)

        target_dynamic_features[dimension] = self.state.nodes[source]["dynamic"][dimension]

        return target_dynamic_features

    def step(self, action):
        assert isinstance(action, nx.DiGraph)
        connect_graph = action
        updates = {}
        for target in connect_graph.nodes:
            neighbors = self._get_in_edges(target, connect_graph)
            if len(neighbors) > 0:
                updates[target] = self._interact(target, neighbors)
        for target in updates:
            self.state.nodes[target]["dynamic"] = updates[target]
        return self.state

    def _get_in_edges(self, node, G):
        return list(map(lambda e: e[0], filter(lambda e: e[1] == node, G.edges)))

if __name__ == "__main__":
    network = nx.grid_graph(dim=(10, 10))
    env = Population(network=network)
    obs = env.reset()
    for e in obs.edges:
        env.step(e)
