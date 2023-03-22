from ga import GA
from q1 import SimpCVRP
from deap import creator, base, tools
import os
import utils
from utils import txt2json, create_stats_objs, BASE_DIR, export_csv
from plot import plot_route
import random
import numpy as np


class EMTCVRP(GA):
    def __init__(
        self,
        cx_prob,
        mut_prob,
        num_gen=150,
        pop_size=400,
        **kwargs,
    ) -> None:
        super().__init__(cx_prob, mut_prob, num_gen, pop_size)
        self.op1 = SimpCVRP(cx_prob, mut_prob, num_gen, pop_size)
        self.op2 = SimpCVRP(cx_prob, mut_prob, num_gen, pop_size)
        self.name = "q5"

        self.logbook1, self.stats1 = create_stats_objs()
        self.logbook2, self.stats2 = create_stats_objs()

    def register_select(self, select_method):
        super().register_select(select_method)
        self.op1.register_select(select_method)
        self.op2.register_select(select_method)

    def load_data(self, data_path1, data_path2):
        self.op1.load_data(data_path1)
        self.op2.load_data(data_path2)

        self.op1.csv_title = (
            f"{self.op1.data['instance_name']}_"
            f"selection_{self.op1.select_method}"
            f"pop{self.op1.pop_size}_crossProb{self.op1.cx_prob}"
            f"_mutProb{self.op1.mut_prob}_numGen{self.op1.num_gen}"
        )

        self.op2.csv_title = (
            f"{self.op2.data['instance_name']}_"
            f"selection_{self.op2.select_method}"
            f"pop{self.op2.pop_size}_crossProb{self.op2.cx_prob}"
            f"_mutProb{self.op2.mut_prob}_numGen{self.op2.num_gen}"
        )
        self.logbook1.clear()
        self.logbook2.clear()

    def create_creators(self):
        self.op1.create_creators()
        self.op2.create_creators()

    def generate_pop_fitness(self):
        self.op1.generate_pop_fitness()
        self.op2.generate_pop_fitness()

    def run_generations(self):
        for gen in range(self.num_gen):
            print(f"{20*'#'} Currently Evaluating {gen} Generation {20*'#'}")
            invalid_ind1, invalid_ind2 = self._run_one_generation_with_transfer(gen)
            self.record_stat(invalid_ind1, invalid_ind2, gen)
        print(f"{20 * '#'} End of Generations {20 * '#'} ")

    def _run_one_generation_with_transfer(self, gen):
        invalid_inds = []
        # select
        offspring1 = self.op1.toolbox.select(self.op1.pop, len(self.op1.pop))
        offspring1 = list(map(self.op1.toolbox.clone, offspring1))

        offspring2 = self.op2.toolbox.select(self.op2.pop, len(self.op2.pop))
        offspring2 = list(map(self.op2.toolbox.clone, offspring2))
        # cross over
        for chrom1, chrom2 in zip(offspring1[::2], offspring1[1::2]):
            if random.random() < self.cx_prob:
                self.op1.toolbox.cxover(chrom1, chrom2)
                del chrom1.fitness.values
                del chrom2.fitness.values

        for chrom1, chrom2 in zip(offspring2[::2], offspring2[1::2]):
            if random.random() < self.cx_prob:
                self.op2.toolbox.cxover(chrom1, chrom2)
                del chrom1.fitness.values
                del chrom2.fitness.values
        # mutation
        for chrom in offspring1:
            if random.random() < self.mut_prob:
                self.op1.toolbox.mutate(chrom)
                del chrom.fitness.values

        for chrom in offspring2:
            if random.random() < self.mut_prob:
                self.op2.toolbox.mutate(chrom)
                del chrom.fitness.values
        # knowledge transfer
        if gen > 0 and gen % 10 == 0:
            inject_num = 10
            # find inject_num best solution of op2
            curr_pop1 = tools.selBest(self.op1.pop, len(self.op1.pop))
            his_pop1 = tools.selBest(self.op2.pop, len(self.op2.pop))
            his_best1 = his_pop1[:inject_num]
            # convert to numpy.array
            curr_pop1 = np.array(curr_pop1)
            his_pop1 = np.array(his_pop1)
            his_best1 = np.array(his_best1)

            curr_pop2 = tools.selBest(self.op2.pop, len(self.op2.pop))
            his_pop2 = tools.selBest(self.op1.pop, len(self.op1.pop))
            his_best2 = his_pop2[:inject_num]
            # convert to numpy.array
            curr_pop2 = np.array(curr_pop2)
            his_pop2 = np.array(his_pop2)
            his_best2 = np.array(his_best2)

            # generate inject
            inject1 = DA(curr_pop1, his_pop1, his_best1)

            inject2 = DA(curr_pop2, his_pop2, his_best2)

            # map to Chromosome
            inject1 = np.rint(inject1).astype(int).tolist()
            inject1 = np.clip(inject1, 1, 100)
            inject1 = [creator.Chromosome(x) for x in inject1]

            inject2 = np.rint(inject2).astype(int).tolist()
            inject2 = np.clip(inject2, 1, 100)
            inject2 = [creator.Chromosome(x) for x in inject2]

            # save to offspring1
            for i in range(len(inject1)):
                idx = random.randint(0, len(offspring1) - 1)
                offspring1[idx] = inject1[i]
            for i in range(len(inject2)):
                idx = random.randint(0, len(offspring2) - 1)
                offspring2[idx] = inject2[i]
        # evaluate fitness of invalid
        invalid_ind1 = [ind for ind in offspring1 if not ind.fitness.valid]
        fitnesses = map(self.op1.toolbox.evaluate, invalid_ind1)
        for ind, fit in zip(invalid_ind1, fitnesses):
            ind.fitness.values = fit
        self.op1.pop[:] = offspring1
        invalid_inds.append(invalid_ind1)
        # for op2
        invalid_ind2 = [ind for ind in offspring2 if not ind.fitness.valid]
        fitnesses = map(self.op2.toolbox.evaluate, invalid_ind2)
        for ind, fit in zip(invalid_ind2, fitnesses):
            ind.fitness.values = fit
        self.op2.pop[:] = offspring2
        invalid_inds.append(invalid_ind2)
        return invalid_inds

    # def _run_one_generation_with_transfer(self, gen):
    #     invalid_inds = []
    #     # select
    #     offspring1 = self.op1.toolbox.select(self.op1.pop, len(self.op1.pop))
    #     offspring1 = list(map(self.op1.toolbox.clone, offspring1))
    #     # cross over
    #     for chrom1, chrom2 in zip(offspring1[::2], offspring1[1::2]):
    #         if random.random() < self.cx_prob:
    #             self.op1.toolbox.cxover(chrom1, chrom2)
    #             del chrom1.fitness.values
    #             del chrom2.fitness.values
    #     # mutation
    #     for chrom in offspring1:
    #         if random.random() < self.mut_prob:
    #             self.op1.toolbox.mutate(chrom)
    #             del chrom.fitness.values
    #     # knowledge transfer
    #     if gen > 0 and gen % 10 == 0:
    #         inject_num = 10
    #         # find inject_num best solution of op2
    #         curr_pop = tools.selBest(self.op1.pop, len(self.op1.pop))
    #         his_pop = tools.selBest(self.op2.pop, len(self.op2.pop))
    #         his_best = his_pop[:inject_num]
    #         # convert to numpy.array
    #         curr_pop = np.array(curr_pop)
    #         his_pop = np.array(his_pop)
    #         his_best = np.array(his_best)
    #         # generate inject
    #         inject = DA(curr_pop, his_pop, his_best)
    #         # map to Chromosome
    #         inject = np.rint(inject).astype(int).tolist()
    #         inject = np.clip(inject, 1, 100)
    #         inject = [creator.Chromosome(x) for x in inject]
    #         # save to offspring1
    #         for i in range(len(inject)):
    #             idx = random.randint(0, len(offspring1) - 1)
    #             offspring1[idx] = inject[i]
    #     # evaluate fitness of invalid
    #     invalid_ind1 = [ind for ind in offspring1 if not ind.fitness.valid]
    #     fitnesses = map(self.op1.toolbox.evaluate, invalid_ind1)
    #     for ind, fit in zip(invalid_ind1, fitnesses):
    #         ind.fitness.values = fit
    #     self.op1.pop[:] = offspring1
    #     invalid_inds.append(invalid_ind1)
    #     # for op2
    #     invalid_inds.append(self.op2._run_one_generation())
    #     return invalid_inds

    def record_stat(self, invalid_ind1, invalid_ind2, gen):
        self.op1.record_stat(invalid_ind1, gen)
        self.op2.record_stat(invalid_ind2, gen)

    def get_best(self):
        best1 = self.op1.get_best()
        best2 = self.op2.get_best()
        return (best1, best2)

    def export(self):
        csv_dir_path = os.path.join(BASE_DIR, "results", self.name)
        if not os.path.exists(csv_dir_path):
            os.mkdir(csv_dir_path)
        csv_path1 = os.path.join(csv_dir_path, self.op1.csv_title + ".csv")
        csv_path2 = os.path.join(csv_dir_path, self.op2.csv_title + ".csv")
        export_csv(csv_path1, self.op1.logbook)
        export_csv(csv_path2, self.op2.logbook)

    def plot(self):
        best_subroutes1, best_subroutes2 = self.get_best()
        fig_dir_path = os.path.join(BASE_DIR, "figures", self.name)
        if not os.path.exists(fig_dir_path):
            os.mkdir(fig_dir_path)
        fig_path1 = os.path.join(fig_dir_path, self.op1.csv_title + ".png")
        plot_route(self.op1.data, best_subroutes1, self.op1.csv_title, fig_path1)
        fig_path2 = os.path.join(fig_dir_path, self.op2.csv_title + ".png")
        plot_route(self.op1.data, best_subroutes2, self.op2.csv_title, fig_path2)

    def run(self):
        self.create_creators()
        self.generate_pop_fitness()
        self.run_generations()
        self.get_best()
        self.export()


def DA(curr_pop, his_pop, his_bestSolution):
    # curr_pop and his_pop denote the current population and
    # population from another domain. Both in the form of n*d matrix.
    # n is the number of individual, and d is the variable dimension.
    # They do not have to be with the same d. We assume they have the
    # same n (same population size)

    # his_bestSolution is the best solutions from one domain.
    # output is the transformed solution.
    curr_len = curr_pop.shape[1]
    tmp_len = his_pop.shape[1]
    if curr_len < tmp_len:
        curr_pop = np.pad(curr_pop, ((0, 0), (0, tmp_len - curr_len)), mode="constant")
    elif curr_len > tmp_len:
        his_pop = np.pad(his_pop, ((0, 0), (0, curr_len - tmp_len)), mode="constant")

    xx = curr_pop.T
    noise = his_pop.T
    d, n = xx.shape
    xxb = np.vstack((xx, np.ones((1, n))))
    noise_xb = np.vstack((noise, np.ones((1, n))))
    Q = np.dot(noise_xb, noise_xb.T)
    P = np.dot(xxb, noise_xb.T)
    reg = 1e-5 * np.eye(d + 1)
    reg[-1, -1] = 0
    W = np.dot(P, np.linalg.inv(Q + reg))

    # tmmn = W.shape[0]
    # W = np.delete(W, tmmn - 1, axis=0)
    # W = np.delete(W, tmmn - 1, axis=1)
    best_n = his_bestSolution.shape[0]

    if curr_len <= tmp_len:
        a = his_bestSolution.T
        b = np.vstack((a, np.ones((1, best_n))))
        tmp_solution = np.dot(W, b).T
        inj_solution = tmp_solution[:, :curr_len]
        inj_solution = curr_pop[:len(his_bestSolution)]
    elif curr_len > tmp_len:
        new_array = np.expand_dims(np.array([0] * (curr_len - tmp_len)), axis=0)
        his_bestSolution = np.concatenate((his_bestSolution, new_array), axis=1)
        a = his_bestSolution.T
        b = np.vstack((a, np.ones((1, best_n))))
        inj_solution = np.dot(W, b).T

    return inj_solution


def test_DA():
    curr_pop = np.random.rand(100, 13)
    his_pop = np.random.rand(100, 7)
    his_bestSolution = his_pop[1, :].reshape((1, 7))
    transformedsolution = DA(curr_pop, his_pop, his_bestSolution)
    print(transformedsolution)


if __name__ == "__main__":
    # test_DA()
    cx_prob = 0.8
    mut_prob = 0.04
    num_gen = 10000
    pop_size = 400
    p = EMTCVRP(cx_prob, mut_prob, num_gen, pop_size)
    p.register_select("tournament")
    p.load_data("data/CVRP_s/c201.txt", "data/CVRP_s/c202.txt")
    p.run()
    p.plot()
