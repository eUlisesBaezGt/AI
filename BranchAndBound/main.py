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


def BranchAndBound(graph, heuristics, start="Arad", goal="Bucharest"):
    frontier = [(0, start, [])]
    explored = set()

    while frontier:
        frontier.sort(key=lambda node: node[0])
        current = frontier.pop(0)

        if current[1] == goal:
            return current[2] + [current[1]]

        if current[1] not in explored:
            explored.add(current[1])
            for next_node, _ in graph.content[current[1]]:
                for i in graph.content[current[1]]:
                    if i[0] == next_node:
                        cost = current[0] + int(i[1])
                frontier.append((cost, next_node, current[2] + [current[1]]))

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

    path = BranchAndBound(graph, heuristics)

    print(f"Path: {path}")


if __name__ == "__main__":
    main()
