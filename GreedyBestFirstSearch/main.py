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

    def view_all(self):
        print("\n\nGraph:\n------")
        print("ORIGIN -> [(DESTINY, WEIGHT), ...]")
        for origin, destiny in self.content.items():
            print(f"{origin} -> {destiny}")
        print("\n")


def greedy_best_first_search(graph, heuristics, start="Neamt", goal="Bucharest"):
    if start == goal:
        return [start]

    path = [start]

    while True:
        current = path[-1]
        children = []
        heuristic = 0
        if current == goal:
            return path

        print(f"Current: {current}")

        # Get children
        for node in graph.content[current]:
            if node[0] not in children:
                children.append(node[0])
                for h in heuristics.content[node[0]]:
                    if h[0] == goal:
                        heuristic = int(h[1])
            print(f"Children: {children}")
            print(f"Heuristic for {node[0]}: {heuristic} ")

        if len(children) == 0:
            return "No path found."

        # Get the child with the lowest heuristic with lambda function
        sorted_heuristics = sorted(children, key=lambda x: int(heuristics.content[x][0][1])) # CHECAR PARA EL 0
        print(f"Sorted Heuristics: {sorted_heuristics}")
        child_with_lowest_heuristic = sorted_heuristics[0]
        print(f"Child with lowest heuristic: {child_with_lowest_heuristic}")
        path.append(child_with_lowest_heuristic)


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

    # graph.view_all()
    # heuristics.view_all()


if __name__ == "__main__":
    main()
