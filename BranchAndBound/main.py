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


def BranchAndBound(graph, heuristics, start="Neamt", goal="Bucharest"):
    if start == goal:
        return [start]

    frontier = [start]
    path = []

    while frontier:
        current = frontier.pop(0)
        path.append(current)

        for neighbor, weight in graph.content[current]:
            if neighbor == goal:
                path.append(neighbor)
                return path
            frontier.append(neighbor)

        for node, weight in heuristics.content[current]:
            if node == goal:
                path.append(node)
                return path


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

    path = BranchAndBound(graph, heuristics)

    print(f"Path: {path}")


if __name__ == "__main__":
    main()
