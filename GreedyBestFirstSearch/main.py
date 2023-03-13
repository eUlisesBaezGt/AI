class Graph:
    def __init__(self):
        self.content = dict()

    def new_edge(self, origin, destiny, weight):
        if origin not in self.content:
            self.content[origin] = []
        if destiny not in self.content:
            self.content[destiny] = []
        self.content[origin].append((destiny, weight))

    def view_all(self):
        print("\n\nGraph:\n------")
        print("ORIGIN -> [(DESTINY, WEIGHT), ...]")
        for origin, destiny in self.content.items():
            print(f"{origin} -> {destiny}")
        print("\n")


def greedy_best_first_search(graph, heuristics, start="Arad", goal="Bucharest"):
    if start == goal:
        return [start]

    path = [start]

    while True:
        current = path[-1]
        children = []
        heuristic = 0
        if current == goal:
            return path

        # Get children
        for node in graph.content[current]:
            if node[0] not in children:
                children.append(node[0])
                for h in heuristics.content[node[0]]:
                    if h[0] == goal:
                        heuristic = int(h[1])



            print(f"Current: {current}")
            print(f"Children: {children}")
            print(f"Heuristic for {node[0]}: {heuristic} ")

        # Get the child with the lowest heuristic


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

    graph.view_all()

    path = greedy_best_first_search(graph, heuristics)


if __name__ == "__main__":
    main()
