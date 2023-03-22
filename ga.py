import random
import os

from deap import base, creator, tools

from plot import plot_route
from utils import create_stats_objs, export_csv, txt2json, BASE_DIR


class GA:
    def __init__(
        self,
        cx_prob,
        mut_prob,
        num_gen=150,
        pop_size=400,
    ) -> None:
        self.name = "original"
        # typically 0.8 ~ 0.95
        self.cx_prob = cx_prob
        # typically 0.001 ~ 0.01
        self.mut_prob = mut_prob
        self.pop_size = pop_size
        self.num_gen = num_gen

        self.toolbox = base.Toolbox()

        self.logbook, self.stats = create_stats_objs()

    def load_data(self, data_path):
        self.data = txt2json(data_path)
        self.num_chromosome = self.data["Number_of_customers"]

        self.csv_title = (
            f"{self.data['instance_name']}_"
            f"selection_{self.select_method}_"
            f"pop_{self.pop_size}_crossProb_{self.cx_prob}"
            f"_mutProb_{self.mut_prob}_numGen_{self.num_gen}"
        )
        self.logbook.clear()

    def register_select(self, select_method):
        if select_method == "roulette-wheel":
            self.toolbox.register("select", tools.selRoulette)
        elif select_method == "tournament":
            self.toolbox.register("select", tools.selTournament, tournsize=3)
        elif select_method == 'random':
            self.toolbox.register("select", tools.selRandom)
        self.select_method = select_method

    def create_creators(self):
        raise NotImplementedError

    def _cxover(self, input_ind1, input_ind2):
        ind1 = [x - 1 for x in input_ind1]
        ind2 = [x - 1 for x in input_ind2]
        size = min(len(ind1), len(ind2))
        a, b = random.sample(range(size), 2)
        if a > b:
            a, b = b, a

        # print(f"The cutting points are {a} and {b}")
        holes1, holes2 = [True] * size, [True] * size
        for i in range(size):
            if i < a or i > b:
                holes1[ind2[i]] = False
                holes2[ind1[i]] = False

        # We must keep the original values somewhere before scrambling everything
        temp1, temp2 = ind1, ind2
        k1, k2 = b + 1, b + 1
        for i in range(size):
            if not holes1[temp1[(i + b + 1) % size]]:
                ind1[k1 % size] = temp1[(i + b + 1) % size]
                k1 += 1

            if not holes2[temp2[(i + b + 1) % size]]:
                ind2[k2 % size] = temp2[(i + b + 1) % size]
                k2 += 1

        # Swap the content between a and b (included)
        for i in range(a, b + 1):
            ind1[i], ind2[i] = ind2[i], ind1[i]

        # Finally adding 1 again to reclaim original input
        ind1 = [x + 1 for x in ind1]
        ind2 = [x + 1 for x in ind2]
        return ind1, ind2

    def mutation(self, chromosome, indpb):
        """
        Inputs : Individual route
                 Probability of mutation betwen (0,1)
        Outputs : Mutated individual according to the probability
        """
        size = len(chromosome)
        for i in range(size):
            if random.random() < indpb:
                swap_indx = random.randint(0, size - 2)
                if swap_indx >= i:
                    swap_indx += 1
                chromosome[i], chromosome[swap_indx] = (
                    chromosome[swap_indx],
                    chromosome[i],
                )

        return chromosome

    def _get_num_vehicles_required(self, chromosome):
        """
        Get number of vehicles required by current solution: chromosome.
        """
        sub_routes = route_to_subroute(self.data, chromosome)
        return len(sub_routes)

    def printRoute(self, route, merge=False):
        route_str = "0"
        sub_route_count = 0
        for sub_route in route:
            sub_route_count += 1
            sub_route_str = "0"
            for customer_id in sub_route:
                sub_route_str = f"{sub_route_str} - {customer_id}"
                route_str = f"{route_str} - {customer_id}"
            sub_route_str = f"{sub_route_str} - 0"
            if not merge:
                print(f"  Vehicle {sub_route_count}'s route: {sub_route_str}")
            route_str = f"{route_str} - 0"
        if merge:
            print(route_str)

    def generate_pop_fitness(self):
        self.pop = self.toolbox.population(n=self.pop_size)
        self.invalid_chrom = [chrom for chrom in self.pop if not chrom.fitness.valid]
        self.fitnesses = list(map(self.toolbox.evaluate, self.invalid_chrom))

        for chrom, fit in zip(self.invalid_chrom, self.fitnesses):
            chrom.fitness.values = fit

        self.pop = self.toolbox.select(self.pop, len(self.pop))
        self.record_stat(self.invalid_chrom, gen=0)

    def check_valid(self):
        raise NotImplementedError

    def run_generations(self):
        for gen in range(self.num_gen):
            print(f"{20*'#'} Currently Evaluating {gen} Generation {20*'#'}")
            invalid_ind = self._run_one_generation()
            self.record_stat(invalid_ind, gen)

        print(f"{20 * '#'} End of Generations {20 * '#'} ")

    def _run_one_generation(self):
        # select
        offspring = self.toolbox.select(self.pop, len(self.pop))
        offspring = list(map(self.toolbox.clone, offspring))
        # cross over
        for chrom1, chrom2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.cx_prob:
                self.toolbox.cxover(chrom1, chrom2)
                del chrom1.fitness.values
                del chrom2.fitness.values
        # mutation
        for chrom in offspring:
            if random.random() < self.mut_prob:
                self.toolbox.mutate(chrom)
                del chrom.fitness.values
        # evaluate fitness of invalid
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        self.pop[:] = offspring
        return invalid_ind

    def get_best(self):
        self.best_chrom = tools.selBest(self.pop, 1)[0]
        best_subroutes = route_to_subroute(self.data, self.best_chrom)
        self.printRoute(best_subroutes)
        return best_subroutes

    def record_stat(self, invalid_ind, gen):
        record = self.stats.compile(self.pop)
        best_individual = tools.selBest(self.pop, 1)[0]
        record["best_one"] = best_individual
        record["fitness_best_one"] = best_individual.fitness
        record["best_num_vehicle"] = self._get_num_vehicles_required(best_individual)
        self.logbook.record(Generation=gen, evals=len(invalid_ind), **record)
        print(self.logbook.stream)

    def export(self, sub_dir=None):
        if sub_dir is None:
            csv_dir_path = os.path.join(BASE_DIR, "results", self.name)
        else:
            csv_dir_path = os.path.join(BASE_DIR, "results", self.name, sub_dir)
        if not os.path.exists(csv_dir_path):
            os.makedirs(csv_dir_path)
        csv_path = os.path.join(csv_dir_path, self.csv_title + ".csv")
        export_csv(csv_path, self.logbook)

    def plot(self, sub_dir=None):
        best_subroutes = self.get_best()
        if sub_dir is None:
            fig_dir_path = os.path.join(BASE_DIR, "figures", self.name)
        else:
            fig_dir_path = os.path.join(BASE_DIR, "figures", self.name, sub_dir)
        if not os.path.exists(fig_dir_path):
            os.makedirs(fig_dir_path)
        fig_path = os.path.join(fig_dir_path, self.csv_title + ".png")
        plot_route(self.data, best_subroutes, self.csv_title, fig_path)

    def run(self, sub_dir=None):
        self.create_creators()
        self.generate_pop_fitness()
        self.run_generations()
        self.get_best()
        self.export(sub_dir)


def get_route_cost(data, sub_routes, unit_cost=1):
    total_cost = 0

    for sub_route in sub_routes:
        # Initializing the subroute distance to 0
        sub_route_distance = 0
        # Initializing customer id for depot as 0
        last_customer_id = 0

        for customer_id in sub_route:
            # Distance from the last customer id to next one in the given subroute
            distance = data["distance_matrix"][last_customer_id][customer_id]
            sub_route_distance += distance
            # Update last_customer_id to the new one
            last_customer_id = customer_id

        # After adding distances in subroute, adding the route cost from last customer to depot
        # that is 0
        sub_route_distance = (
            sub_route_distance + data["distance_matrix"][last_customer_id][0]
        )

        # Cost for this particular sub route
        sub_route_transport_cost = unit_cost * sub_route_distance

        # Adding this to total cost
        total_cost = total_cost + sub_route_transport_cost

    return total_cost


def route_to_subroute(data, chromosome):
    """
    Inputs: Sequence of customers that a route has
            Loaded instance problem
    Outputs: Route that is divided in to subroutes
             which is assigned to each vechicle.
    """
    route = []
    sub_route = []
    vehicle_load = 0
    last_customer_id = 0
    vehicle_capacity = data["vehicle_capacity"]

    for customer_id in chromosome:
        # print(customer_id)
        demand = data[f"customer_{customer_id}"]["demand"]
        # print(f"The demand for customer_{customer_id}  is {demand}")
        updated_vehicle_load = vehicle_load + demand

        if updated_vehicle_load <= vehicle_capacity:
            sub_route.append(customer_id)
            vehicle_load = updated_vehicle_load
        else:
            route.append(sub_route)
            sub_route = [customer_id]
            vehicle_load = demand

        last_customer_id = customer_id

    if sub_route != []:
        route.append(sub_route)

    # Returning the final route with each list inside for a vehicle
    return route
