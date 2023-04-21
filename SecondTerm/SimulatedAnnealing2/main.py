import random
import math


class Graph:
    def __init__(self):
        self.content = {}

    def new_edge(self, origin, destiny, weight):
        if origin not in self.content:
            self.content[origin] = []
        if destiny not in self.content:
            self.content[destiny] = []
        self.content[origin].append((destiny, weight))
        self.content[destiny].append((origin, weight))


def get_solution_cost(solution):
    cost = 0
    for city in range(len(graph.content) - 1):
        cost += graph.content[solution[city]][solution[city + 1]][1]
    return cost


def generate_random_swap_solution(current_solution):
    indexes = random.sample(range(1, len(current_solution) - 1), 2)
    value1 = current_solution[indexes[0]]
    value2 = current_solution[indexes[1]]

    swapped_solution = current_solution.copy()
    swapped_solution[indexes[0]] = value2
    swapped_solution[indexes[1]] = value1
    return swapped_solution


def decrease_temperature(temperature, percentage_to_reduce_temperature):
    decreased_value = temperature * (percentage_to_reduce_temperature / 100)
    return temperature - decreased_value


def SimulatedAnnealing(initial_solution, initial_temperature=100, stop_temperature=0, iterations=1000,
                       percentage_to_reduce_temperature=2):
    temperature = initial_temperature

    current_solution = initial_solution
    first_solution_cost = get_solution_cost(current_solution)
    current_solution_cost = 0

    while temperature > stop_temperature:
        for iteration in range(iterations):
            new_solution = generate_random_swap_solution(current_solution)
            new_solution_cost = get_solution_cost(new_solution)
            difference = new_solution_cost - current_solution_cost
            if difference >= 0:
                current_solution = new_solution
                current_solution_cost = new_solution_cost
            else:
                probability = math.exp(difference / temperature)
                if random.random() < probability:
                    current_solution = new_solution
                    current_solution_cost = new_solution_cost
        temperature = decrease_temperature(temperature, percentage_to_reduce_temperature)
    return current_solution, first_solution_cost, current_solution_cost


def main():
    global graph
    graph = Graph()
    with open("data.txt") as file:
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

    path = SimulatedAnnealing(graph, heuristics)

    print(f"Path: {path}")


if __name__ == "__main__":
    main()
