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


def greedy_best_first_search(graph, heuristics, start, goal):
    if start == goal:
        return [start]

    frontier = []
    explored = set()
    parents = {}

    frontier.append(start)
    parents[start] = None

    while frontier:
        frontier.sort(key=lambda x: heuristics.content[x][0][1])
        current = frontier.pop(0)

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parents[current]
            return path[::-1]

        explored.add(current)

        for neighbor, _ in graph.content[current]:
            if neighbor not in explored and neighbor not in frontier:
                frontier.append(neighbor)
                parents[neighbor] = current

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

    path = greedy_best_first_search(graph, heuristics, "Arad", "Bucharest")

    print(f"Path: {path}")


if __name__ == "__main__":
    main()
