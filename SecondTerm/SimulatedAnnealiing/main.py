# -*- coding: utf-8 -*-
"""Simulated_Annealing_class.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qlOthUHW-_UbJCVEAXADhxKhHLZn1Pf_
"""

import networkx as nx
import random
import math
import matplotlib.pyplot as plt
# from networkx.algorithms import approximation as approx


def print_graph(G):
    # nodes
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=5000)
    nx.draw_networkx_labels(G, pos, font_size=10,
                            font_family="sans-serif", font_color='white')

    # edges
    nx.draw_networkx_edges(
        G, pos, edgelist=G.edges, width=6, alpha=1, edge_color="black", style="dashed"
    )
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels, font_size=10, font_color='red')

    # ploting
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


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

    ("Sibiu", "Craiova", 1000), ("Sibiu",
                                 "Pitesti", 1000), ("Sibiu", "Bucharest", 1000),
    ("Craiova", "Sibiu", 1000), ("Pitesti",
                                 "Sibiu", 1000), ("Bucharest", "Sibiu", 1000),

    ("Rimnicu Vilcea", "Fagaras", 1000), ("Rimnicu Vilcea", "Bucharest", 1000),
    ("Fagaras", "Rimnicu Vilcea", 1000), ("Bucharest", "Rimnicu Vilcea", 1000),

    ("Craiova", "Bucharest", 1000), ("Craiova", "Fagaras", 1000),
    ("Bucharest", "Craiova", 1000), ("Fagaras", "Craiova", 1000),

    ("Pitesti", "Fagaras", 1000),
    ("Fagaras", "Pitesti", 1000),

})

# print(G.nodes())

# print(G.edges())

# print_graph(G)


def generate_initial_solution(start, g):
    # COMPREHENSIVE LIST
    connections = [edges[1] for edges in G.edges() if edges[0] == start]
    connections.insert(0, start)
    connections.append(start)
    print(connections)
    return connections


def generate_random_swap_solution(current_solution):
    indexes = random.sample(range(1, len(current_solution)-1), 2)
    value1 = current_solution[indexes[0]]
    value2 = current_solution[indexes[1]]

    swapped_solution = current_solution.copy()
    swapped_solution[indexes[0]] = value2
    swapped_solution[indexes[1]] = value1
    return swapped_solution



def simulated_annealing(initial_solution, initial_temperature, stop_temperature, iterations, percentage_to_reduce_temperature):
    temperature = initial_temperature

    current_solution = initial_solution

    while temperature >= stop_temperature:
        for iteration in range(iterations):
            # Generate a random selected solution
            new_ranodm_solution = generate_random_swap_solution(
                current_solution)
            print(new_ranodm_solution)
            break
        break


def main():
    initial_solution = generate_initial_solution('Fagaras', 0)
    print(initial_solution)
    initial_temperature = 100
    stop_temperature = 0
    iterations = 5
    percentage_to_reduce_temperature = 2

    result = simulated_annealing(initial_solution, initial_temperature,
                                 stop_temperature, iterations, percentage_to_reduce_temperature)


if __name__ == "__main__":
    main()