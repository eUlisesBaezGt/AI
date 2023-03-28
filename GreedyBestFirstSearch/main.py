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


def greedy_best_first_search(graph, heuristics, start="Neamt", goal="Bucharest"):
    if start == goal:
        return [start]

    frontier = [start]
    explored = []
    path = []

    while frontier:
        node = frontier.pop(0)
        explored.append(node)
        path.append(node)

        if node == goal:
            return path

        for destiny, weight in graph.content[node]:
            if destiny not in explored and destiny not in frontier:
                frontier.append(destiny)

        frontier.sort(key=lambda x: heuristics.content[x][0][1])


def main():
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

    path = greedy_best_first_search(graph, heuristics)

    print(f"Path: {path}")


if __name__ == "__main__":
    main()
