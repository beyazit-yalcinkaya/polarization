import random
import numpy as np
import networkx as nx
from copy import deepcopy
from env import Population
from utils import *

class Recommender():

    def __init__(self, n, gamma=0.0, h=8, c=1):
        self.n = n
        self.gamma = gamma
        self.h = h
        self.c = c

    def __call__(self, state):
        assert isinstance(state, nx.Graph) or isinstance(state, nx.DiGraph)
        while True:
            target = random.choice(list(state.nodes))

            neighbors = self._get_in_edges(target, state)
            if isinstance(state, nx.Graph):
                neighbors += self._get_out_edges(target, state)

            n_random_neighbors = int(self.gamma * len(neighbors))
            neighbors = random.sample(neighbors, len(neighbors) - n_random_neighbors) + random.sample(list(state.nodes), n_random_neighbors)

            weights = np.array([self._similarity(source, target, state)**self.h for source in neighbors])
            if sum(weights) == 0:
                continue

            weights = weights / np.sum(weights)
            
            source_index = np.random.choice(len(neighbors), p=weights)
            source = neighbors[source_index]
            return (source, target)

    def _similarity(self, node1, node2, state):
        return ((self.c if state.nodes[node1]["static"] == state.nodes[node2]["static"] else 0) + np.sum(state.nodes[node1]["dynamic"] == state.nodes[node2]["dynamic"])) / (self.c+self.n)

    def _get_in_edges(self, node, G):
        return list(map(lambda e: e[0], filter(lambda e: e[1] == node, G.edges)))

    def _get_out_edges(self, node, G):
        return list(map(lambda e: e[1], filter(lambda e: e[0] == node, G.edges)))
