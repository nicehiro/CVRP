from ga import GA, get_route_cost, route_to_subroute
from deap import creator, base, tools
import os
import utils
import random


class MultiObjectiveCVRP(GA):
    def __init__(
        self,
        cx_prob,
        mut_prob,
        num_gen=150,
        pop_size=400,
        **kwargs,
    ) -> None:
        super().__init__(cx_prob, mut_prob, num_gen, pop_size)
        self.name = "q4"

    def create_creators(self):
        creator.create("FitnessMin", base=base.Fitness, weights=(-1.0, -1.0))
        creator.create("Chromosome", base=list, fitness=creator.FitnessMin)

        self.toolbox.register(
            "indexes",
            random.sample,
            range(1, self.num_chromosome + 1),
            self.num_chromosome,
        )
        self.toolbox.register(
            "chromosome", tools.initIterate, creator.Chromosome, self.toolbox.indexes
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.chromosome
        )

        self.toolbox.register("evaluate", self._eval_fitness, unit_cost=1)
        self.toolbox.register("cxover", self._cxover)
        self.toolbox.register("mutate", self.mutation, indpb=self.mut_prob)

    def _eval_fitness(self, chromosome, unit_cost=1):
        vehicles = self._get_num_vehicles_required(chromosome)
        subroutes = route_to_subroute(self.data, chromosome)
        route_cost = get_route_cost(self.data, subroutes, unit_cost)
        return (vehicles, route_cost)


if __name__ == "__main__":
    cx_prob = 0.9
    mut_prob = 0.01
    num_gen = 4000
    pop_size = 400

    for select in utils.selection_methods:
        ga = MultiObjectiveCVRP(cx_prob, mut_prob, num_gen, pop_size, select)

        dir_path = os.path.join(utils.BASE_DIR, "data", "CVRP_MOP")
        paths = os.listdir(path=dir_path)

        for txt_path in paths:
            txt_path = os.path.join(dir_path, txt_path)
            ga.load_data(txt_path)
            ga.run()
            ga.plot()
