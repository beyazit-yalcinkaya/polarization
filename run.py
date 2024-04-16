from sorting_model import SortingModel
from env import Population
from agent import Recommender
from utils import *

if __name__ == "__main__":
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

    obs = env.reset()
    for _ in range(10000):
        action = agent(obs)
        obs = env.step(action)

    for node in obs.nodes:
        obs.nodes[node]["dynamic"] = obs.nodes[node]["dynamic"] / m

    draw(G=obs, show_fig=True)

    # sorting_model = SortingModel(n=3, m=256, gamma=1.0)
	# sorting_model.run(100000)
	# print(sorting_model.calculate_sorting())
	# sorting_model.draw(show_fig=True)