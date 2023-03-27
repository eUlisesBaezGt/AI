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


def breadth_first_search(graph):
    print("\n\nBreadth-First Search:\n---------------------")
    queue = []
    visited = []
    queue.append("Arad")
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            if node == "Neamt" or node == "Eforie":
                break
            for destiny, weight in graph.content[node]:
                queue.append(destiny)

    if "Neamt" in visited or "Eforie" in visited:
        print("Path found!")
    else:
        print("Path not found!")
    print(visited)


def main():
    with open("data.txt", "w") as file:
        file.write("20 23\n")
        # Arad Zerind 5
        # Zerind Oradea 5
        # Oradea Sibiu 5
        # Sibiu Fagaras 5
        # Sibiu RimnicuVilcea 5
        # Fagaras Bucharest 5
        # RimnicuVilcea Pitesti 5
        # RimnicuVilcea Craiova 5
        # Pitesti Bucharest 5
        # Craiova Pitesti 5
        # Arad Sibiu 5
        # Arad Timisoara 5
        # Timisoara Lugoj 5
        # Lugoj Mehadia 5
        # Mehadia Drobeta 5
        # Drobeta Craiova 5
        # Bucharest Giurgiu 5
        # Bucharest Urziceni 5
        # Urziceni Hirsova 5
        # Hirsova Eforie 5
        # Urziceni Vaslui 5
        # Vaslui Iasi 5
        # Iasi Neamt 5
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

    graph.view_all()
    breadth_first_search(graph)


if __name__ == "__main__":
    main()
