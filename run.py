from q1 import SimpCVRP
from q2 import CVRP
from q3 import CVRPTW
from q4 import MultiObjectiveCVRP

from utils import test_selection, test_cxover_prob, test_mut_prob, test_pop_size, test

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default="q1")
    parser.add_argument("--test", type=str, default="select")
    parser.add_argument("--num_gen", type=int, default=1000)
    parser.add_argument("--cx_prob", type=float, default=0.8)
    parser.add_argument("--mut_prob", type=float, default=0.02)
    parser.add_argument("--pop_size", type=int, default=400)
    args = parser.parse_args()

    gas = {"q1": SimpCVRP, "q2": CVRP, "q3": CVRPTW, "q4": MultiObjectiveCVRP}

    data_paths = {
        "q1": "CVRP_s",
        "q2": "CVRP",
        "q3": "CVRP",
        "q4": "CVRP_MOP",
    }

    testcases = {
        "select": test_selection,
        "cxover": test_cxover_prob,
        "mutation": test_mut_prob,
        "pop": test_pop_size,
        "all": test,
    }

    ga = gas[args.question]
    data_path = data_paths[args.question]
    testcase = testcases[args.test]

    testcase(
        ga,
        data_path=data_path,
        cx_prob=args.cx_prob,
        mut_prob=args.mut_prob,
        num_gen=args.num_gen,
        pop_size=args.pop_size,
    )
