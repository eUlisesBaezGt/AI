class Graph:
    """
    A class to represent a directed weighted graph using adjacency lists.
    """
    def __init__(self):
        # Initialize an empty dictionary to store the graph's adjacency list
        self.content = dict()

    def new_edge(self, origin, destiny, weight):
        """
        Adds a new edge to the graph.

        Parameters:
        origin (str): The starting node of the edge.
        destiny (str): The ending node of the edge.
        weight (int/float): The weight of the edge.
        """
        # If the origin or destiny node does not exist, initialize it with an empty list
        if origin not in self.content:
            self.content[origin] = []
        if destiny not in self.content:
            self.content[destiny] = []
        # Add the edge to the origin node's list
        self.content[origin].append((destiny, weight))

    def view_all(self):
        """
        Prints all edges in the graph, showing the connections and their weights.
        """
        print("\n\nGraph:\n------")
        print("ORIGIN -> [(DESTINY, WEIGHT), ...]")
        for origin, destiny in self.content.items():
            print(f"{origin} -> {destiny}")

def comp_limited_depth_search(graph):
    """
    Completes a limited depth search on the graph and prints the result.

    Parameters:
    graph (Graph): The graph to perform the search on.
    """
    print("\n\nLimited Depth Search:\n---------------------")
    origin = input("Origin: ")
    destiny = input("Destiny: ")
    path = limited_depth_search(graph, origin, destiny)
    print("\n\nRESULTS:\n--------")
    print("FROM:", origin)
    print("TO:", destiny)
    if path:
        print("\nPATH FOUND:")
        print(" -> ".join(path))
    else:
        print("No path found")

def limited_depth_search(graph, origin, destiny, limit=9):
    """
    Performs a depth-limited search from the origin to the destiny within a given depth limit.

    Parameters:
    graph (Graph): The graph to perform the search on.
    origin (str): The starting node.
    destiny (str): The destination node.
    limit (int): The maximum depth to search (default is 9).

    Returns:
    list/None: A list of nodes forming the path if found, None otherwise.
    """
    if origin == destiny:
        return [origin]
    if limit == 0:
        return None
    for node in graph.content.get(origin, []):
        path = limited_depth_search(graph, node[0], destiny, limit - 1)
        if path:
            return [origin] + path
    return None

def main():
    """
    Main function to demonstrate graph functionality and depth-limited search.
    """
    # # Writes a predefined graph structure to 'data.txt'
    # with open("data.txt", "w") as file:
    #     file.writelines([
    #         "20 23\n",
    #         "Arad Zerind 5\n", "Zerind Oradea 5\n", "Oradea Sibiu 5\n",
    #         "Sibiu Fagaras 5\n", "Sibiu RimnicuVilcea 5\n", "Fagaras Bucharest 5\n",
    #         "RimnicuVilcea Pitesti 5\n", "RimnicuVilcea Craiova 5\n", "Pitesti Bucharest 5\n",
    #         "Craiova Pitesti 5\n", "Arad Sibiu 5\n", "Arad Timisoara 5\n",
    #         "Timisoara Lugoj 5\n", "Lugoj Mehadia 5\n", "Mehadia Drobeta 5\n",
    #         "Drobeta Craiova 5\n", "Bucharest Giurgiu 5\n", "Bucharest Urziceni 5\n",
    #         "Urziceni Hirsova 5\n", "Hirsova Eforie 5\n", "Urziceni Vaslui 5\n",
    #         "Vaslui Iasi 5\n", "Iasi Neamt 5\n"
    #     ])

    # Initializes a graph and loads edges from 'data.txt'
    graph = Graph()
    with open("data.txt") as file:
        lines = file.readlines()
    for line in lines[1:]:  # Skip the first line as it contains the number of nodes and edges
        origin, destiny, weight = line.split()
        graph.new_edge(origin, destiny, int(weight))

    graph.view_all()  # Display the graph

    print("\nFINISHED\n")
    
    comp_limited_depth_search(graph)  # Perform a limited depth search

if __name__ == "__main__":
    main()
