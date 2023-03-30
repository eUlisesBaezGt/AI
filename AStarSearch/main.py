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


def AStarSearch(graph, heuristcs, origin="Arad", destination="Bucharest"):
    costs = {origin: 0}
    paths = {origin: []}
    frontier = [origin]

    while frontier:
        frontier.sort(key=lambda node: costs[node] + int(heuristcs.content[node][0][1]))
        current = frontier.pop(0)

        if current == destination:
            print(f"Cost: {costs[current]}")
            return paths[current] + [current]

        for next_node, _ in graph.content[current]:
            for i in graph.content[current]:
                if i[0] == next_node:
                    cost = costs[current] + int(i[1])
                if next_node not in costs or cost < costs[next_node]:
                    costs[next_node] = cost
                    paths[next_node] = paths[current] + [current]
                    if next_node not in frontier:
                        frontier.append(next_node)
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

    path = AStarSearch(graph, heuristics)

    print(f"Path: {path}")


if __name__ == "__main__":
    main()
