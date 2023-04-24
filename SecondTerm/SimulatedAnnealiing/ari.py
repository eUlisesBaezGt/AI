import networkx as nx
import random, math
import matplotlib.pyplot as plt

G = nx.DiGraph()

G.add_weighted_edges_from({
    ("Sibiu", "Rimnicu Vilcea", 80), ("Sibiu", "Fagaras", 99),
    ("Rimnicu Vilcea", "Sibiu", 80), ("Fagaras", "Sibiu", 99),

    ("Rimnicu Vilcea", "Craiova", 146), ("Rimnicu Vilcea", "Pitesti", 97),
    ("Craiova", "Rimnicu Vilcea", 146), ("Pitesti", "Rimnicu Vilcea", 97),

    ("Craiova", "Pitesti", 138),
    ("Pitesti", "Craiova", 138),

    ("Pitesti", "Bucharest", 101),
    ("Bucharest", "Pitesti", 101),

    ("Fagaras", "Bucharest", 211),
    ("Bucharest", "Fagaras", 211),

    ("Sibiu", "Craiova", 1000), ("Sibiu", "Pitesti", 1000), ("Sibiu", "Bucharest", 1000),
    ("Craiova", "Sibiu", 1000), ("Pitesti", "Sibiu", 1000), ("Bucharest", "Sibiu", 1000),

    ("Rimnicu Vilcea", "Fagaras", 1000), ("Rimnicu Vilcea", "Bucharest", 1000),
    ("Fagaras", "Rimnicu Vilcea", 1000), ("Bucharest", "Rimnicu Vilcea", 1000),

    ("Craiova", "Bucharest", 1000), ("Craiova", "Fagaras", 1000),
    ("Bucharest", "Craiova", 1000), ("Fagaras", "Craiova", 1000),

    ("Pitesti", "Fagaras", 1000),
    ("Fagaras", "Pitesti", 1000),

})

def generate_initial_solution(start):
    connections = [edges[1] for edges in G.edges() if edges[0] == start]

    connections.insert(0, start)
    connections.append(start)

    for node in G.nodes():
        if node not in connections:
            return []

    return connections


def decrease_temperature(temperature, percentage_to_reduce):
    decrease_percentage = 100 * float(percentage_to_reduce) / float(temperature)
    return decrease_percentage


def generate_random_swap_solution(current_solution):
    indexes = random.sample(range(1, len(current_solution) - 1), 2)
    value_one = current_solution[indexes[0]]
    value_two = current_solution[indexes[1]]
    swaped_solution = current_solution.copy()
    swaped_solution[indexes[0]] = value_two
    swaped_solution[indexes[1]] = value_one
    return swaped_solution


def get_solution_cost(solution):
    cost = 0

    for i in range(len(solution) - 1):
        cost = cost + G[solution[i]][solution[i + 1]]["weight"]

    cost = cost + G[solution[len(solution) - 2]][solution[len(solution) - 1]]["weight"]

    return cost


def simulated_annealing_result(initial_solution, initial_temperature, number_of_iterations, stop_temperature,
                               percentage_to_reduce_temperature):
    temperature = initial_temperature
    current_solution = initial_solution

    first_solution_cost = get_solution_cost(current_solution)
    current_solution_cost = 0

    while temperature >= stop_temperature:
        for i in range(number_of_iterations):
            new_random_solution = generate_random_swap_solution(current_solution)

            current_solution_cost = get_solution_cost(current_solution)
            new_random_solution_cost = get_solution_cost(new_random_solution)

            diferences_between_costs = current_solution_cost - new_random_solution_cost

            if diferences_between_costs >= 0:
                current_solution = new_random_solution
            else:
                uniform_random_number = random.uniform(0, 1)

                acceptance_probability = math.exp(diferences_between_costs / temperature)

                if uniform_random_number <= acceptance_probability:
                    current_solution = new_random_solution

        alpha = decrease_temperature(temperature, percentage_to_reduce_temperature)
        temperature = int(temperature - alpha)

    return current_solution, first_solution_cost, current_solution_cost


def print_simulated_annealing_result(result, initial_solution):
    tuples = []
    for i in range(len(result[0]) - 1):
        tuples.append((result[0][i], result[0][i + 1]))

    simulated_annealing_result = nx.Graph()

    simulated_annealing_result.add_weighted_edges_from({
        (node_tuple[0], node_tuple[1], G[node_tuple[0]][node_tuple[1]]["weight"]) for node_tuple in tuples
    })

    print('\nInitial solution = ', initial_solution)
    print('\nInitial solution cost = ', result[1])
    print('\nSimulated annealing solution = ', result[0])
    print('\nSimulated annealing solution cost = ', result[2])



def main():
    origin = input('Enter the origin city: ')
    initial_solution = generate_initial_solution(origin)
    initial_temperature = 100
    stop_temperature = 0
    number_of_iterations = 5
    percentage_to_reduce_temperature = 2

    result = simulated_annealing_result(initial_solution, initial_temperature, number_of_iterations, stop_temperature,
                                        percentage_to_reduce_temperature)

    print_simulated_annealing_result(result, initial_solution)


main()
