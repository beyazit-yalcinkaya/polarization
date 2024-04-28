import sys
import numpy as np
from env import Population
from agent import Recommender
from utils import *

# Explore random graphs in the context of this research. You can use the knowledge from randomness and computation. This might be the opputinity to use probabilistic proof method FINALLY.

if __name__ == "__main__":
    n_samples = sys.argv[1]
    max_len = sys.argv[2]
    n_samples = int(n_samples)
    max_len = int(max_len)
    assert n_samples > 0 and max_len > 0
    n = 3
    m = 256
    k = 2
    w = 10
    gamma = 1.0
    periodic = True
    network = grid_2d_moore_graph(w, w, periodic=periodic)
    # network = nx.grid_2d_graph(w, w, periodic=periodic)

    env = Population(n=n, m=m, k=k, network=network)
    agent = Recommender(n=n, gamma=gamma)

    samples = []
    for i in range(n_samples):
        obs = env.reset()
        last_step = None
        sorting_value = None
        sorting_values = []
        for j in range(max_len):
            action = agent(obs)
            obs = env.step(action)
            # sorting_value = env.calculate_sorting()
            # sorting_values.append(sorting_value)
            # if sorting_value == 1.0:
                # last_step = j
                # break
        samples.append(sorting_value)
        print(env.calculate_sorting())
        # if last_step is None:
            # print(sorting_values)

        for node in obs.nodes:
            obs.nodes[node]["dynamic"] = obs.nodes[node]["dynamic"] / m
        draw(G=obs, show_fig=True)

    # for node in obs.nodes:
    #     obs.nodes[node]["dynamic"] = obs.nodes[node]["dynamic"] / m

    # draw(G=obs, show_fig=True)

    # print(len(samples), np.mean(samples), np.std(samples))

    # sorting_model = SortingModel(n=3, m=256, gamma=1.0)
	# sorting_model.run(100000)
	# print(sorting_model.calculate_sorting())
	# sorting_model.draw(show_fig=True)