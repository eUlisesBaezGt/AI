import random
from KAGraph import KAGraph as KAg


def initialize_population(graph, origin, destination, population_size):
    population = []
    for _ in range(population_size):
        path = [origin]
        current = origin
        while current != destination:
            neighbors = graph.get_neighbors(current)
            next_city = random.choice(list(neighbors))
            path.append(next_city)
            current = next_city
        population.append(path)
    return population


def fitness_function(graph, path):
    if graph is None or path is None or len(path) < 2:
        return float('inf')

    cost = 0
    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]
        if node2 not in graph.get_neighbors(node1):
            # If node2 is not a neighbor of node1, the edge is missing
            return float('inf')
        weight = graph.get_weight(node1, node2)
        cost += int(weight)

    return cost


def selection(population, fitness_values):
    selected = random.choices(population, weights=fitness_values, k=2)
    return selected[0], selected[1]


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 2)
    offspring1 = parent1[:crossover_point] + [city for city in parent2 if city not in parent1[:crossover_point]]
    offspring2 = parent2[:crossover_point] + [city for city in parent1 if city not in parent2[:crossover_point]]
    return offspring1, offspring2


def mutation(graph, offspring):
    index1, index2 = random.sample(range(1, len(offspring) - 1), 2)
    neighbors1 = list(graph.get_neighbors(offspring[index1]))
    neighbors2 = list(graph.get_neighbors(offspring[index2]))
    if offspring[index1] in neighbors2:
        neighbors2.remove(offspring[index1])
    if offspring[index2] in neighbors1:
        neighbors1.remove(offspring[index2])
    offspring[index1], offspring[index2] = random.choice(list(neighbors1)), random.choice(list(neighbors2))
    return offspring


def genetic_algorithm(graph, origin, destination, population_size=50, num_generations=100, mutation_rate=0.1):
    population = initialize_population(graph, origin, destination, population_size)

    for _ in range(num_generations):
        fitness_values = [1 / fitness_function(graph, path) for path in population]
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population, fitness_values)
            offspring1, offspring2 = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                offspring1 = mutation(graph, offspring1)
            if random.random() < mutation_rate:
                offspring2 = mutation(graph, offspring2)
            new_population.extend([offspring1, offspring2])
        population = new_population

    best_path = min(population, key=lambda x: fitness_function(graph, x))
    best_cost = fitness_function(graph, best_path)
    return best_path, best_cost


def main():
    graph = KAg.Graph()
    with open("data.txt") as file:
        lines = file.readlines()

    for i in range(1, len(lines)):
        origin, destiny, weight = lines[i].split()
        graph.add_edge(origin, destiny, weight)

    path, cost = genetic_algorithm(graph, "Arad", "Bucharest")
    print(f"Path: {path}")
    print(f"Cost: {cost}")


if __name__ == "__main__":
    main()
