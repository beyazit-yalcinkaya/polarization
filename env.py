import random
import numpy as np
import networkx as nx
import gymnasium as gym
from utils import *
from copy import deepcopy

# An observation is a networkx graph with static and dynamic node attributes.
# An action is an edge on the same networkx graph.

class Population(gym.Env):

    def __init__(self, n=4, m=10, k=2, network=None, transform=None):
        self.n = n # Number of dynamic features
        self.m = m # Upper bound for dynamic features
        self.k = k # Upper bound for static features
        self.speed = 0.1
        self.tolerance = 100

        if network is None:
            self.network = grid_2d_moore_graph(w, w, periodic=periodic)
        else:
            self.network = network
        if transform is None:
            self.transform = self._transform_tolerance
        else:
            self.transform = transform
        self.state = deepcopy(self.network)

    def reset(self):
        nx.set_node_attributes(self.state, {node: {"static": np.random.randint(self.k), "dynamic": np.random.randint(self.m, size=(self.n,)), "confidence": np.random.normal(0.5, 0.1)} for node in self.state.nodes})
        self.within = [(n1, n2) for i1, n1 in enumerate(self.state.nodes) for i2, n2 in enumerate(self.state.nodes) if self.state.nodes[n1]["static"] == self.state.nodes[n2]["static"] and i1 < i2]
        self.between = [(n1, n2) for i1, n1 in enumerate(self.state.nodes) for i2, n2 in enumerate(self.state.nodes) if self.state.nodes[n1]["static"] != self.state.nodes[n2]["static"] and i1 < i2]
        return self.state

    def _transform_tornberg(self, target, neighbors):
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

    def _transform_middle(self, target, neighbors):
        index = np.random.choice(len(neighbors))
        source = neighbors[index]

        differences = np.array([1 if self.state.nodes[source]["dynamic"][i] != self.state.nodes[target]["dynamic"][i] else 0 for i in range(self.n)])

        target_dynamic_features = np.array(self.state.nodes[target]["dynamic"])

        if sum(differences) == 0:
            return target_dynamic_features

        differences = differences / sum(differences)
        dimension = np.random.choice(self.n, p=differences)

        old_value = target_dynamic_features[dimension]
        new_value = (old_value + self.state.nodes[source]["dynamic"][dimension]) // 2

        target_dynamic_features[dimension] = new_value

        return target_dynamic_features

    def _transform_uniform_random(self, target, neighbors):
        index = np.random.choice(len(neighbors))
        source = neighbors[index]

        differences = np.array([1 if self.state.nodes[source]["dynamic"][i] != self.state.nodes[target]["dynamic"][i] else 0 for i in range(self.n)])

        target_dynamic_features = np.array(self.state.nodes[target]["dynamic"])

        if sum(differences) == 0:
            return target_dynamic_features
        
        differences = differences / sum(differences)
        dimension = np.random.choice(self.n, p=differences)

        target_old_value = target_dynamic_features[dimension]
        source_value = self.state.nodes[source]["dynamic"][dimension]

        new_value = random.randint(*sorted([source_value, target_old_value])) # Don't use numpy here

        target_dynamic_features[dimension] = new_value

        return target_dynamic_features

    def _transform_tolerance(self, target, neighbors):
        n_neighbors = len(neighbors)
        index = np.random.choice(n_neighbors)
        source = neighbors[index]

        tolerance_lower_bound = self.state.nodes[source]["dynamic"] - self.tolerance*(1 - self.state.nodes[source]["confidence"])
        tolerance_upper_bound = self.state.nodes[source]["dynamic"] + self.tolerance*(1 - self.state.nodes[source]["confidence"])

        differences = np.array([1 if (tolerance_lower_bound[i] <= self.state.nodes[target]["dynamic"][i] and tolerance_upper_bound[i] >= self.state.nodes[target]["dynamic"][i]) else 0 for i in range(self.n)])

        target_dynamic_features = np.array(self.state.nodes[target]["dynamic"])

        if sum(differences) == 0:
            return target_dynamic_features
        
        probabilities = differences / sum(differences)
        dimension = np.random.choice(self.n, p=probabilities)

        old_value = target_dynamic_features[dimension]
        source_value = self.state.nodes[source]["dynamic"][dimension]

        new_value = old_value + int(2*self.speed*(source_value - old_value))

        target_dynamic_features[dimension] = new_value

        self.state.nodes[source]["confidence"] = self.state.nodes[source]["confidence"] + (self.speed*(np.sum(differences == 1)/n_neighbors - 0.5))/3

        return target_dynamic_features

    def step(self, action):
        assert isinstance(action, nx.DiGraph)
        connect_graph = action
        updates = {}
        for target in connect_graph.nodes:
            neighbors = self._get_in_neighbors(target, connect_graph)
            if len(neighbors) > 0:
                updates[target] = self.transform(target, neighbors)
        for target in updates:
            self.state.nodes[target]["dynamic"] = updates[target]
        return self.state

    def _get_in_neighbors(self, node, G):
        return list(map(lambda e: e[0], G.in_edges(node)))

    def _fraction_shared_dynamic(self, n1, n2):
        return np.sum(self.state.nodes[n1]["dynamic"] == self.state.nodes[n2]["dynamic"]) / self.n

    # TODO: Optimize this
    def calculate_sorting(self):
        within_sorting_value = np.mean([self._fraction_shared_dynamic(a, b) for a,b in self.within])
        between_sorting_value = np.mean([self._fraction_shared_dynamic(a, b) for a,b in self.between])
        return within_sorting_value - between_sorting_value











