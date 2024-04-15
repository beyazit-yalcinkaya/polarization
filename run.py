from sorting_model import SortingModel

sorting_model = SortingModel(n=3, m=256, gamma=1)
sorting_model.run(10000)
sorting_model.draw()