class Graph:
    def __init__(self):
        self.content = dict()

    def new_edge(self, origin, destiny, weight):
        if origin not in self.content:
            self.content[origin] = []
        if destiny not in self.content:
            self.content[destiny] = []
        self.content[origin].append((destiny, weight))
        self.content[destiny].append((origin, weight))
        

def WeightedAStarSeach(graph, heuristics, start="Neamt", goal="Bucharest"):
    if start == goal:
        return [start]

    frontier = [start]
    explored = []
    path = []

    path.append(start)

    while frontier:

        current = frontier.pop(0)
        explored.append(current)

        for neighbor, weight in graph.content[current]:
            if neighbor not in explored:
                frontier.append(neighbor)
                path.append(neighbor)
                if goal == neighbor:
                    break

        if goal in frontier:
            break

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

    path = WeightedAStarSeach(graph, heuristics)

    print(f"Path: {path}")


if __name__ == "__main__":
    main()
