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

    def getConnections(self, origin):
        print("[(DESTINY, WEIGHT), ...]")
        print(self.content[origin])

    def getFromNode(self, node):
        paths = []
        for origin, destiny in self.content.items():
            for i in range(len(destiny)):
                if destiny[i][0] == node:
                    paths.append(origin)
        print(paths)

    def getToNode(self, node):
        paths = []
        for origin, destiny in self.content.items():
            for i in range(len(destiny)):
                if origin == node:
                    paths.append(destiny[i][0])
        print(paths)

    def getCost(self, origin, destiny):
        cost = 0
        flag = False
        if origin not in self.content or destiny not in self.content:
            print("Invalid origin or destiny")
            return
        else:
            print("Valid origin and destiny")
        for i in range(len(self.content[origin])):
            if self.content[origin][i][0] == destiny:
                cost = self.content[origin][i][1]
                flag = True
                break
        if not flag:
            print("No path found")
        else:
            print("Cost from", origin, "to", destiny, "is", cost)


def main():
    graph = Graph()
    with open("data.txt") as file:
        lines = file.readlines()
    nodes, edges = lines[0].split()
    for i in range(1, len(lines)):
        origin, destiny, weight = lines[i].split()
        graph.new_edge(origin, destiny, weight)
    while True:
        print("\n\nMAIN MENU")
        print("-------------")
        print("1. Search for a path's cost")
        print("2. Search for a node's connections")
        print("3. Search node's father")
        print("4. Search node's child")
        print("5. Print graph")
        print("0. Exit")
        option = input("\nOption: ")
        if option == "1":
            print("\n\nSearch for a path's cost:")
            origin = input("Origin? ")
            destiny = input("Destiny? ")
            graph.getCost(origin, destiny)
        elif option == "2":
            origin = input("\n\nSearch connections for a node: ")
            graph.getConnections(origin)
        elif option == "3":
            origin = input("\n\nSearch node's father: ")
            graph.getFromNode(origin)
        elif option == "4":
            origin = input("\n\nSearch node's child: ")
            graph.getToNode(origin)
        elif option == "5":
            graph.view_all()
        elif option == "0":
            print("Exiting...")
            break
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()
