from ga import GA, route_to_subroute
from deap import creator, base, tools
import os
import utils
import random


class CVRPTW(GA):
    def __init__(
        self,
        cx_prob,
        mut_prob,
        num_gen=150,
        pop_size=400,
        **kwargs,
    ) -> None:
        super().__init__(cx_prob, mut_prob, num_gen, pop_size)
        self.name = "q3"

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
        # check vehicle condition
        num_vehicle = self._get_num_vehicles_required(chromosome)
        total_vehicle = self.data["max_vehicle_number"]
        if num_vehicle > total_vehicle:
            return 10e32
        subroutes = route_to_subroute(self.data, chromosome)
        route_cost = get_route_cost(self.data, subroutes, unit_cost, 1, 1)
        return (route_cost,)


def get_route_cost(data, sub_routes, unit_cost, wait_cost=0, delay_cost=0):
    total_cost = 0

    for sub_route in sub_routes:
        sub_route_time_cost = 0
        elapsed_time = 0
        # Initializing the subroute distance to 0
        sub_route_distance = 0
        # Initializing customer id for depot as 0
        last_customer_id = 0

        for customer_id in sub_route:
            # Distance from the last customer id to next one in the given subroute
            distance = data["distance_matrix"][last_customer_id][customer_id]
            sub_route_distance += distance

            # calc time cost
            # arrival_time <= ready_time, need to wait, has cost
            # arrival_time > due_time, cannot serve, has cost
            arrival_time = elapsed_time + distance
            time_cost = wait_cost * max(
                data[f"customer_{customer_id}"]["ready_time"] - arrival_time, 0
            ) + delay_cost * max(
                arrival_time - data[f"customer_{customer_id}"]["due_time"], 0
            )
            sub_route_time_cost += time_cost
            elapsed_time = (
                arrival_time + data[f"customer_{customer_id}"]["service_time"]
            )
            # Update last_customer_id to the new one
            last_customer_id = customer_id

        # After adding distances in subroute, adding the route cost from last customer to depot
        # that is 0
        sub_route_distance = (
            sub_route_distance + data["distance_matrix"][last_customer_id][0]
        )

        # Cost for this particular sub route
        sub_route_transport_cost = unit_cost * sub_route_distance

        sub_route_cost = sub_route_time_cost + sub_route_transport_cost

        # Adding this to total cost
        total_cost = total_cost + sub_route_cost

    return total_cost


if __name__ == "__main__":
    cx_prob = 0.9
    mut_prob = 0.01
    num_gen = 10000
    pop_size = 400

    for select in utils.selection_methods:
        ga = CVRPTW(cx_prob, mut_prob, num_gen, pop_size, select)

        dir_path = os.path.join(utils.BASE_DIR, "data", "CVRP")
        paths = os.listdir(path=dir_path)

        for txt_path in paths:
            txt_path = os.path.join(dir_path, txt_path)
            ga.load_data(txt_path)
            ga.run()
            ga.plot()
