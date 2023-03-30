import time

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


def comp_limited_depth_search(graph):
    print("\n\nLimited Depth Search:\n---------------------")
    print("Origin: ", end="")
    origin = input()
    print("Destiny: ", end="")
    destiny = input()
    path = limited_depth_search(graph, origin, destiny)
    print("\n\nRESULTS:\n--------")
    print("FROM:", origin)
    print("TO:", destiny)
    print("\nPATH FOUND:")
    if path:
        for i in range(len(path)):
            if i == len(path) - 1:
                print(path[i])
            else:
                print(path[i], end=" -> ")
    else:
        print("No path found")


def limited_depth_search(graph, origin, destiny, limit=9):  # Also works with 7,could be an alternative
    if origin == destiny:
        return [origin]
    if limit == 0:
        return None
    for node in graph.content[origin]:
        path = limited_depth_search(graph, node[0], destiny, limit - 1)
        if path:
            return [origin] + path
    return None


def main():
    with open("data.txt", "w") as file:
        file.write("20 23\n")
        file.write("Arad Zerind 5\n")
        file.write("Zerind Oradea 5\n")
        file.write("Oradea Sibiu 5\n")
        file.write("Sibiu Fagaras 5\n")
        file.write("Sibiu RimnicuVilcea 5\n")
        file.write("Fagaras Bucharest 5\n")
        file.write("RimnicuVilcea Pitesti 5\n")
        file.write("RimnicuVilcea Craiova 5\n")
        file.write("Pitesti Bucharest 5\n")
        file.write("Craiova Pitesti 5\n")
        file.write("Arad Sibiu 5\n")
        file.write("Arad Timisoara 5\n")
        file.write("Timisoara Lugoj 5\n")
        file.write("Lugoj Mehadia 5\n")
        file.write("Mehadia Drobeta 5\n")
        file.write("Drobeta Craiova 5\n")
        file.write("Bucharest Giurgiu 5\n")
        file.write("Bucharest Urziceni 5\n")
        file.write("Urziceni Hirsova 5\n")
        file.write("Hirsova Eforie 5\n")
        file.write("Urziceni Vaslui 5\n")
        file.write("Vaslui Iasi 5\n")
        file.write("Iasi Neamt 5\n")

    graph = Graph()
    with open("data.txt") as file:
        lines = file.readlines()
    nodes, edges = lines[0].split()

    for i in range(1, len(lines)):
        origin, destiny, weight = lines[i].split()
        graph.new_edge(origin, destiny, weight)

    # comp_limited_depth_search(graph)
    graph.view_all()

    print("\nFINISHED\n")
    time.sleep(1000)


if __name__ == "__main__":
    main()
