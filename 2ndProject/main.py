from Graph import *
from GreedBestFirstSearch import greedy_best_first_search
from WeightedAStar import weighted_a_star_search
from AStarSearch import a_star_search
from BeamSearch import beam_search
from BeamSearch import beam_search
from BranchAndBound import branch_and_bound


def main():
    graph = Graph()
    with open("graph.txt") as file:
        lines = file.readlines()

    for i in range(1, len(lines)):
        origin, destiny, weight = lines[i].split()
        graph.new_edge(origin, destiny, weight)

    heuristics = Graph()
    with open("heuristics.txt") as file:
        lines = file.readlines()

    for i in range(1, len(lines)):
        origin, destiny, weight = lines[i].split()
        heuristics.new_edge(origin, destiny, weight)

    print("Origin: ", end="")
    origin = input()
    destiny = "Bucharest"

    path = greedy_best_first_search(graph, heuristics, origin, destiny)
    print(f"\nGreedy Best First Search: {path}")

    path, total_cost = a_star_search(graph, heuristics, origin, destiny)
    print(f"\nA* Search: {path}")
    print(f"Total cost: {total_cost}")

    w = float(input("\nInsert w: "))
    path, total_cost = weighted_a_star_search(graph, heuristics, origin, destiny, w)
    print(f"\nUsing w = {w}")
    print(f"Weighted A* Search: {path}")
    print(f"Total cost: {total_cost}")

    k = int(input("\nInsert k: "))
    path = beam_search(graph, heuristics, origin, destiny, k)
    print(f"\nBeam Search: {path}")

    path = branch_and_bound(graph, heuristics, origin, destiny)
    print(f"\nBranch and Bound: {path}")


if __name__ == "__main__":
    main()
