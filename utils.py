import os
import io
import numpy as np
from deap import tools
import csv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def calculate_distance(customer1, customer2):
    return (
        (customer1["coordinates"]["x"] - customer2["coordinates"]["x"]) ** 2
        + (customer1["coordinates"]["y"] - customer2["coordinates"]["y"]) ** 2
    ) ** 0.5


def export_csv(csv_path, logbook):
    csv_columns = logbook[0].keys()
    try:
        with open(csv_path, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in logbook:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def create_stats_objs():
    # Method to create stats and logbook objects
    """
    Inputs : None
    Outputs : tuple of logbook and stats objects.
    """
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Methods for logging
    logbook = tools.Logbook()
    logbook.header = (
        "Generation",
        "evals",
        "avg",
        "std",
        "min",
        "max",
        "best_one",
        "fitness_best_one",
        "best_num_vehicle",
    )
    return logbook, stats


def txt2json(txt_path):
    """
    Parse test case txt file into json format with calculated distance matrix.
    """
    # txt_path = os.path.join(BASE_DIR, "data", txt_path)
    if not txt_path.endswith(".txt"):
        raise "file is not a txt!"

    json_data = {}
    numCustomers = 0
    with io.open(txt_path, "rt", newline="") as file_object:
        for line_count, line in enumerate(file_object, start=1):
            if line_count in [2, 3, 4, 6, 7, 8, 9]:
                pass

            # Instance name details, input text file name
            elif line_count == 1:
                json_data["instance_name"] = line.strip()

            # Vehicle capacity and max vehicles details
            elif line_count == 5:
                values = line.strip().split()
                for i in range(len(values)):
                    if values[i] == "infinite" or values[i] == "Infinite":
                        values[i] = 10e6
                json_data["max_vehicle_number"] = int(values[0])
                json_data["vehicle_capacity"] = float(values[1])

            # Depot details
            elif line_count == 10:
                # This is depot
                values = line.strip().split()
                json_data["depart"] = {
                    "coordinates": {
                        "x": float(values[1]),
                        "y": float(values[2]),
                    },
                    "demand": float(values[3]),
                    "ready_time": float(values[4]),
                    "due_time": float(values[5]),
                    "service_time": float(values[6]),
                }

            # Customer details
            else:
                # Rest all are customers
                # Adding customer to number of customers
                numCustomers += 1
                values = line.strip().split()
                json_data[f"customer_{values[0]}"] = {
                    "coordinates": {
                        "x": float(values[1]),
                        "y": float(values[2]),
                    },
                    "demand": float(values[3]),
                    "ready_time": float(values[4]),
                    "due_time": float(values[5]),
                    "service_time": float(values[6]),
                }

    customers = ["depart"] + [f"customer_{x}" for x in range(1, numCustomers + 1)]

    # Writing the distance_matrix
    json_data["distance_matrix"] = [
        [
            calculate_distance(json_data[customer1], json_data[customer2])
            for customer1 in customers
        ]
        for customer2 in customers
    ]

    # Writing the number of customers details
    json_data["Number_of_customers"] = numCustomers

    return json_data


def test_selection(GA, data_path, **ga_kwargs):
    selection_methods = ["random", "roulette-wheel", "tournament"]
    ga = GA(**ga_kwargs)

    for selection_method in selection_methods:
        dir_path = os.path.join(BASE_DIR, "data", data_path)
        paths = os.listdir(path=dir_path)

        for txt_path in paths:
            txt_path = os.path.join(dir_path, txt_path)
            ga.register_select(selection_method)
            ga.load_data(txt_path)
            ga.run(sub_dir="selection")
            ga.plot(sub_dir="selection")
            break


def test_cxover_prob(GA, data_path, **ga_kwargs):
    selection_method = "tournament"

    cxover_probs = [0.5, 0.6, 0.7, 0.8, 0.9]

    if "cx_prob" in ga_kwargs:
        ga_kwargs.pop("cx_prob")

    for cxover_prob in cxover_probs:
        ga = GA(cx_prob=cxover_prob, **ga_kwargs)
        dir_path = os.path.join(BASE_DIR, "data", data_path)
        paths = os.listdir(path=dir_path)

        for txt_path in paths:
            txt_path = os.path.join(dir_path, txt_path)
            ga.register_select(selection_method)
            ga.load_data(txt_path)
            ga.run(sub_dir="cxover")
            ga.plot(sub_dir="cxover")
            break


def test_mut_prob(GA, data_path, **ga_kwargs):
    selection_method = "tournament"

    mut_probs = [0.02, 0.04, 0.06, 0.08, 0.1]

    if "mut_prob" in ga_kwargs:
        ga_kwargs.pop("mut_prob")

    for mut_prob in mut_probs:
        ga = GA(mut_prob=mut_prob, **ga_kwargs)
        dir_path = os.path.join(BASE_DIR, "data", data_path)
        paths = os.listdir(path=dir_path)

        for txt_path in paths:
            txt_path = os.path.join(dir_path, txt_path)
            ga.register_select(selection_method)
            ga.load_data(txt_path)
            ga.run(sub_dir="mutation")
            ga.plot(sub_dir="mutation")
            break


def test_pop_size(GA, data_path, **ga_kwargs):
    selection_method = "tournament"

    pop_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    if "pop_size" in ga_kwargs:
        ga_kwargs.pop("pop_size")

    for pop_size in pop_sizes:
        ga = GA(pop_size=pop_size, **ga_kwargs)
        dir_path = os.path.join(BASE_DIR, "data", data_path)
        paths = os.listdir(path=dir_path)

        for txt_path in paths:
            txt_path = os.path.join(dir_path, txt_path)
            ga.register_select(selection_method)
            ga.load_data(txt_path)
            ga.run(sub_dir="pop")
            ga.plot(sub_dir="pop")
            break


def test(GA, data_path, **ga_kwargs):
    selection_method = "tournament"

    ga = GA(**ga_kwargs)
    dir_path = os.path.join(BASE_DIR, "data", data_path)
    paths = os.listdir(path=dir_path)

    for txt_path in paths:
        txt_path = os.path.join(dir_path, txt_path)
        ga.register_select(selection_method)
        ga.load_data(txt_path)
        ga.run(sub_dir="all")
        ga.plot(sub_dir="all")


if __name__ == "__main__":
    d = txt2json("data/CVRP_s/c101.txt")
    print(d)
