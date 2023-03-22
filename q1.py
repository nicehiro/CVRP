from ga import GA, route_to_subroute, get_route_cost
from deap import creator, base, tools
import utils
import os
import random
import plot


class SimpCVRP(GA):
    def __init__(
        self,
        cx_prob,
        mut_prob,
        num_gen=150,
        pop_size=400,
        **kwargs,
    ) -> None:
        super().__init__(cx_prob, mut_prob, num_gen, pop_size)
        self.name = "q1"

    def create_creators(self):
        creator.create("FitnessMin", base=base.Fitness, weights=(-1.0,))
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
        subroutes = route_to_subroute(self.data, chromosome)
        route_cost = get_route_cost(self.data, subroutes, unit_cost)
        return (route_cost,)
