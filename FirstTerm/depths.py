class Graph:
    def __init__(self):
        # Initialize an empty dictionary to store the graph's adjacency list,
        # where keys are node identifiers and values are lists of tuples (neighbor, weight).
        self.content = dict()

    def new_edge(self, origin, destiny, weight):
        """
        Add a new edge to the graph with a specified weight.
        
        If the origin or destiny nodes do not exist in the graph, they are initialized
        with empty lists. Then, the edge (destiny node and weight) is appended to the
        list of edges for the origin node.
        """
        if origin not in self.content:
            self.content[origin] = []
        if destiny not in self.content:
            self.content[destiny] = []
        self.content[origin].append((destiny, weight))

    def view_all(self):
        """
        Print all edges in the graph, showing each node's connections and the weights
        of those connections. This method iterates through each node and prints its
        adjacency list.
        """
        print("\n\nGraph:\n------")
        print("ORIGIN -> [(DESTINY, WEIGHT), ...]")
        for origin, destiny in self.content.items():
            print(f"{origin} -> {destiny}")

def limited_depth_search(graph, origin, destiny, limit):
    # Base case: if the current node is the destination, a path is found.
    if origin == destiny:
        return [origin]
    
    # Base case: if the limit is reached (0), stop searching and return None.
    if limit == 0:
        return None
    
    # Recursive case: iterate through each neighbor of the current node.
    for node in graph.content[origin]:
        # Perform a depth-limited search from the neighbor node, reducing the limit by 1.
        path = limited_depth_search(graph, node[0], destiny, limit - 1)
        
        # If a path is found from the neighbor to the destination, prepend the current node
        # to the path and return it.
        if path:
            return [origin] + path
    
    # If no path is found from this branch, return None.
    return None


def iterative_depth(graph, start, destiny):
    # Iterate over possible depths starting from 0 up to the number of nodes in the graph.
    for depth in range(len(graph.content)):
        # Attempt a limited depth search with the current depth.
        path = limited_depth_search(graph, start, destiny, depth)
        
        # If a path is found at the current depth, print the depth and return the path.
        if path:
            print("Path found at DEPTH: ", depth)
            return path
    
    # If no path is found at any depth, return None.
    return None


def main():
    graph = Graph()
    with open("data.txt") as file:
        lines = file.readlines()

    for line in lines[1:]:  # Read and add edges to the graph.
        origin, destiny, weight = line.strip().split()
        graph.new_edge(origin, destiny, weight)

    graph.view_all()  # Display the graph's connections.

    # Ask the user for the start and end nodes:
    start_node = input("\nPlease enter the start node: ")
    end_node = input("Please enter the end node: ")

    # Perform iterative depth search.
    path = iterative_depth(graph, start_node, end_node)

    print(f"\nPath: {path}")

if __name__ == "__main__":
    main()
