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


def StochasticHillClimbing(graph, heuristcs, origin="Arad", destination="Bucharest"):
    costs = {origin: 0}
    paths = {origin: []}
    frontier = [origin]

    while frontier:
        current = frontier.pop(0)
        if current == destination:
            print(f"Cost: {costs[current]}")
            return paths[current]

        for destiny, weight in graph.content[current]:
            if destiny not in costs or costs[current] + int(weight) < costs[destiny]:
                costs[destiny] = costs[current] + int(weight)
                paths[destiny] = paths[current] + [destiny]
                frontier.append(destiny)

        frontier.sort(key=lambda x: costs[x] + int(heuristcs.content[x][0][1]))
        frontier = frontier[:1]

    return None


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

    path = StochasticHillClimbing(graph, heuristics)

    print(f"Path: {path}")


if __name__ == "__main__":
    main()
