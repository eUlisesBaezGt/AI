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


def main():
    graph = Graph()
    with open("data.txt") as file:
        lines = file.readlines()
    nodes, edges = lines[0].split()

    for i in range(1, len(lines)):
        origin, destiny, weight = lines[i].split()
        graph.new_edge(origin, destiny, weight)

    graph.view_all()


if __name__ == "__main__":
    main()
