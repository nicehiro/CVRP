import os
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


label_index = {
    "selection": 2,
    "pop": 4,
    "cxover": 6,
    "mutation": 8,
}


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


def getCoordinatesDframe(json_instance):
    num_of_cust = json_instance["Number_of_customers"]
    # Getting all customer coordinates
    customer_list = [i for i in range(1, num_of_cust + 1)]
    x_coord_cust = [
        json_instance[f"customer_{i}"]["coordinates"]["x"] for i in customer_list
    ]
    y_coord_cust = [
        json_instance[f"customer_{i}"]["coordinates"]["y"] for i in customer_list
    ]
    # Getting depot x,y coordinates
    depot_x = [json_instance["depart"]["coordinates"]["x"]]
    depot_y = [json_instance["depart"]["coordinates"]["y"]]
    # Adding depot details
    customer_list = [0] + customer_list
    x_coord_cust = depot_x + x_coord_cust
    y_coord_cust = depot_y + y_coord_cust
    df = pd.DataFrame(
        {"X": x_coord_cust, "Y": y_coord_cust, "customer_list": customer_list}
    )
    return df


def plotSubroute(subroute, dfhere, color):
    totalSubroute = [0] + subroute + [0]
    subroutelen = len(subroute)
    for i in range(subroutelen + 1):
        firstcust = totalSubroute[0]
        secondcust = totalSubroute[1]
        plt.plot(
            [dfhere.X[firstcust], dfhere.X[secondcust]],
            [dfhere.Y[firstcust], dfhere.Y[secondcust]],
            c=color,
        )
        totalSubroute.pop(0)


def plot_route(data, subroutes, csv_title, fig_path):
    # Loading the instance
    # ga = GA(data_path, cx_prob=0.8, mut_prob=0.02)
    # subroutes = ga._route_to_subroute(route)
    colorslist = [
        "blue",
        "green",
        "red",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "#008000",
        "#FF69B4",
        "#6495ED",
        "#CD5C5C",
    ]
    colorindex = 0

    # getting df
    dfhere = getCoordinatesDframe(data)

    # Plotting scatter
    plt.figure(figsize=(10, 10))

    for i in range(dfhere.shape[0]):
        if i == 0:
            plt.scatter(dfhere.X[i], dfhere.Y[i], c="red", s=200)
            plt.text(dfhere.X[i], dfhere.Y[i], "depot", fontsize=12)
        else:
            plt.scatter(dfhere.X[i], dfhere.Y[i], c="orange", s=200)
            plt.text(dfhere.X[i], dfhere.Y[i], f"{i}", fontsize=12)

    # Plotting routes
    for route in subroutes:
        plotSubroute(route, dfhere, color=colorslist[colorindex])
        colorindex = (colorindex + 1) % len(colorslist)

    # Plotting is done, adding labels, Title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(csv_title)
    plt.savefig(fig_path)


def plot_best_route(txt_path, csv_path, fig_path):
    data = utils.txt2json(txt_path)
    df = pd.read_csv(csv_path)
    temp = df["best_one"]
    temp = temp.iloc[-1]
    temp[1:-1]
    t = temp[1:-1].split(",")
    individual = [int(x) for x in t]
    subroutes = route_to_subroute(data, individual)
    csv_title = ""
    plot_route(data=data, subroutes=subroutes, csv_title=csv_title, fig_path=fig_path)


def plot_one_fitness(csv_path, save_path):
    df = pd.read_csv(csv_path)
    df['Fitness'] = df['fitness_best_one'].map(lambda x: float(x[1:-2]))
    df.plot(x="Generation", y="Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig(save_path)


def plot_fitness(csv_dir, label, title, save_path):
    df_res = pd.DataFrame()
    # for multi objective
    df_res2 = pd.DataFrame()

    for csv_path in os.listdir(csv_dir):
        if not csv_path.endswith(".csv"):
            continue
        x_label = csv_path.split("_")[label_index[label]]
        csv_path = os.path.join(csv_dir, csv_path)
        df = pd.read_csv(csv_path)

        if "q4" not in csv_dir:
            df_res[x_label] = df["fitness_best_one"].map(lambda x: float(x[1:-2]))
        else:
            # remove first ( and last )
            df_res[x_label] = df["fitness_best_one"].map(
                lambda x: float(x[1:-1].split(",")[1])
            )
            df_res2[x_label] = df["fitness_best_one"].map(
                lambda x: float(x[1:-1].split(",")[0])
            )

    df_res.plot()
    # plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig(save_path)

    if "q4" not in csv_dir:
        df_res2.plot()
        plt.xlabel("Generation")
        plt.ylabel("Number of Vehicles")
        save_path = os.path.join(csv_dir, "vehicle.png")
        plt.savefig(save_path)


def plot_last(csv_dir, label, x, y="Last Fitness"):
    res = {}
    res2 = {}

    for csv_path in os.listdir(csv_dir):
        if not csv_path.endswith(".csv"):
            continue
        x_label = csv_path.split("_")[label_index[label]]
        csv_path = os.path.join(csv_dir, csv_path)
        df = pd.read_csv(csv_path)

        x_label = float(x_label)

        if "q4" not in csv_dir:
            res[x_label] = float(df["fitness_best_one"].iloc[-1][1:-2])
        else:
            # remove first ( and last )
            res[x_label] = float(df["fitness_best_one"].iloc[-1][1:-1].split(",")[1])
            res2[x_label] = float(df["fitness_best_one"].iloc[-1][1:-1].split(",")[0])

    df_res = pd.DataFrame(res.items(), columns=[x, y])
    df_res = df_res.sort_values(by=x)
    df_res.plot(x=x, y=y)
    # plt.title(title)
    plt.ylabel("Fitness")
    save_path = os.path.join(csv_dir, "fitness.png")
    plt.savefig(save_path)


if __name__ == "__main__":
    sns.set_theme()

    # label = "pop"
    # csv_dir = "results/q4/pop"
    # title = "The best fitness value of different population size"
    # x = "Population Size"
    # save_path = os.path.join(csv_dir, "fitness-pop.png")

    # label = "selection"
    # csv_dir = "results/q2/selection"
    # title = "The best fitness value of different selection strategy"
    # save_path = os.path.join(csv_dir, "fitness-selection.png")

    # label = "cxover"
    # csv_dir = "results/q4/cxover"
    # title = "The best fitness value of different crossover probability"
    # x = "Crossover Probability"
    # save_path = os.path.join(csv_dir, "fitness-cxover.png")

    # label = "mutation"
    # csv_dir = "results/q4/mutation"
    # title = "The best fitness value of different mutation probability"
    # x = "Mutation Probability"
    # save_path = os.path.join(csv_dir, "fitness-mutation.png")

    # plot_fitness(csv_dir, label, title, save_path)
    # plot_last(csv_dir, label, x=x)

    # txt_path = "data/CVRP_s/c202.txt"
    # csv_path = "results/q5/C202_selection_tournamentpop400_crossProb0.8_mutProb0.04_numGen10000.csv"
    # fig_path = "figures/q5/q5-2.png"
    # plot_best_route(txt_path, csv_path, fig_path)

    # csv_path = "results/q5/C201_selection_tournamentpop400_crossProb0.8_mutProb0.04_numGen10000.csv"
    # save_path = "results/q5/c201.png"
    # plot_one_fitness(csv_path, save_path)
