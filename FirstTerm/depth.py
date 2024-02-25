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

def depth_search(graph, origin, destiny, path=None):
    """
    Perform a depth from the origin node to the destiny node.
    
    This recursive function attempts to find a path from the origin to the destiny
    by exploring as far as possible along each branch before backtracking.
    
    Parameters:
    - graph: The Graph object to search within.
    - origin: The starting node for the search.
    - destiny: The destination node to find.
    - path: The current path being explored. Initialized as None for the first call.
    
    Returns:
    - A list representing the path from origin to destiny, or None if no path exists.
    """
    if path is None:
        path = []
    path.append(origin)

    # Base case: if the current node is the destiny, return the path.
    if origin == destiny:
        print(f"Path found: {path}")
        return path

    # Recursively search through each neighbor not already in the current path.
    for neighbor in graph.content[origin]:
        # neighbor[0] represents the destination node of an edge originating from 'origin'.
        if neighbor[0] not in path:
            # If the neighbor has not been visited (not in the current path),
            # then proceed to explore this new path.

            # Create a new path including this neighbor by copying the current path to avoid
            # altering the original path for subsequent iterations/recursions.
            new_path = depth_search(graph, neighbor[0], destiny, path.copy())

            # If a path to the destination is found (new_path is not None),
            # immediately return this path. This step ensures that as soon as a valid
            # path is found, it is returned without exploring further unnecessary paths.
            if new_path is not None:
                return new_path  # Return the first successful path found.

    
    # If no path is found after exploring all neighbors, return None.
    return None


def main():
    """
    Main function to demonstrate the graph functionality and depth-first search.
    
    It reads a graph from 'data.txt', displays the graph's adjacency list, and then
    performs a depth-first search based on user input for the start and end nodes.
    """
    graph = Graph()
    with open("data.txt") as file:
        lines = file.readlines()

    for line in lines[1:]:  # Read and add edges to the graph.
        origin, destiny, weight = line.strip().split()
        graph.new_edge(origin, destiny, weight)

    graph.view_all()  # Display the graph's connections.

    # Ask the user for the start and end nodes for the depth-first search.
    start_node = input("\nPlease enter the start node: ")
    end_node = input("Please enter the end node: ")

    # Execute the depth-first search with the specified start and end nodes.
    depth_search(graph, start_node, end_node)

if __name__ == "__main__":
    main()
