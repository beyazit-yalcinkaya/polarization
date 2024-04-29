import sys
import random
import numpy as np
from env import Population
from agent import Recommender
from utils import *

# Explore random graphs in the context of this research. You can use the knowledge from randomness and computation. This might be the opputinity to use probabilistic proof method FINALLY.

if __name__ == "__main__":
    gamma = float(sys.argv[1])
    h = int(sys.argv[2])
    tolerance = int(sys.argv[3])
    seed = int(sys.argv[4])
    assert gamma >= 0.0 and gamma <= 1.0 and h >= 0 and tolerance >= 0 and seed >= 0
    random.seed(seed)
    np.random.seed(seed)
    max_len = 1_000
    n = 3
    m = 256
    k = 2
    w = 10
    periodic = True
    network = grid_2d_moore_graph(w, w, periodic=periodic)

    csv_path = "exps/csvs/"
    fig_path = "exps/figs/"
    file_name = f"exp_gamma_{gamma}_h_{h}_tolerance_{tolerance}_seed_{seed}"

    env = Population(n=n, m=m, k=k, tolerance=tolerance, network=network)
    agent = Recommender(n=n, gamma=gamma, h=h)

    obs = env.reset()
    draw(G=obs, m=m, file_name=fig_path+file_name+"_init")
    last_step = None
    sorting_values = []
    for j in range(max_len):
        action = agent(obs)
        obs = env.step(action)
        sorting_values.append(env.calculate_sorting())
        if all([i >= 1.0 for i in sorting_values[-10:]]):
            break

    draw(G=obs, m=m, file_name=fig_path+file_name+"_final")

    with open(csv_path + file_name + ".csv", 'w+') as f:
        f.write("step,psi")
        f.writelines([f"\n{i+1},{sorting_values[i]}" for i in range(len(sorting_values))])


