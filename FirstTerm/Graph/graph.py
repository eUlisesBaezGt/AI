class Graph:
    """
    A class to represent a directed weighted graph using adjacency lists.
    """
    def __init__(self):
        # Initialize an empty dictionary to hold the adjacency list
        self.content = dict()

    def new_edge(self, origin, destiny, weight):
        """
        Add a new edge to the graph with a specified weight.

        Parameters:
        origin (str): The starting node of the edge.
        destiny (str): The ending node of the edge.
        weight (int/float): The weight of the edge.
        """
        # If the origin node does not exist in the graph, add it with an empty list
        if origin not in self.content:
            self.content[origin] = []
        # If the destiny node does not exist in the graph, add it with an empty list
        if destiny not in self.content:
            self.content[destiny] = []
        # Append the destiny node and weight as a tuple to the origin node's list
        self.content[origin].append((destiny, weight))

    def view_all(self):
        """
        Prints all the edges in the graph with their weights.
        """
        print("\n\nGraph:\n------")
        print("ORIGIN -> [(DESTINY, WEIGHT), ...]")
        for origin, destiny in self.content.items():
            print(f"{origin} -> {destiny}")

    def getConnections(self, origin):
        """
        Prints the connections (edges and weights) of a given node.

        Parameters:
        origin (str): The node whose connections are to be printed.
        """
        print("[(DESTINY, WEIGHT), ...]")
        print(self.content[origin])

    def getFromNode(self, node):
        """
        Prints all nodes that have an edge leading to the specified node.

        Parameters:
        node (str): The node to find the incoming edges for.
        """
        paths = []
        for origin, destiny in self.content.items():
            for i in range(len(destiny)):
                if destiny[i][0] == node:
                    paths.append(origin)
        print(paths)

    def getToNode(self, node):
        """
        Prints all nodes that can be reached directly from the specified node.

        Parameters:
        node (str): The node to find the outgoing edges for.
        """
        paths = []
        for origin, destiny in self.content.items():
            if origin == node:
                paths.extend([d[0] for d in destiny])
        print(paths)

    def getCost(self, origin, destiny):
        """
        Prints the cost of traveling from the origin node to the destiny node, if such a path exists.

        Parameters:
        origin (str): The starting node.
        destiny (str): The ending node.
        """
        cost = 0
        flag = False
        if origin not in self.content or destiny not in self.content:
            print("Invalid origin or destiny")
            return
        else:
            print("Valid origin and destiny")
        for dest, weight in self.content[origin]:
            if dest == destiny:
                cost = weight
                flag = True
                break
        if not flag:
            print("No path found")
        else:
            print(f"Cost from {origin} to {destiny} is {cost}")

def main():
    """
    Main function to interact with the Graph class through a simple CLI.
    """
    graph = Graph()
    with open("data.txt") as file:
        lines = file.readlines()
    nodes, edges = lines[0].split()
    for i in range(1, len(lines)):
        origin, destiny, weight = lines[i].split()
        graph.new_edge(origin, destiny, int(weight))  # Ensure weight is an integer

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
            origin = input("\n\nSearch for a path's cost:\nOrigin? ")
            destiny = input("Destiny? ")
            graph.getCost(origin, destiny)
        elif option == "2":
            origin = input("\n\nSearch connections for a node:\nNode? ")
            graph.getConnections(origin)
        elif option == "3":
            node = input("\n\nSearch node's father:\nNode? ")
            graph.getFromNode(node)
        elif option == "4":
            node = input("\n\nSearch node's child:\nNode? ")
            graph.getToNode(node)
        elif option == "5":
            graph.view_all()
        elif option == "0":
            print("Exiting...")
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()
