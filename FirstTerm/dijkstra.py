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

def dijkstra(graph, origin, destiny):
    """
    Implements Dijkstra's algorithm to find the shortest path from an origin node to
    a destiny node within a graph. It calculates the minimum distance to reach each
    node and determines the shortest path to the destination.
    """
    # Initialize distances from the origin to all nodes as infinity, except for the
    # origin itself which is set to 0.
    distances = {node: float('inf') for node in graph.content}
    distances[origin] = 0

    # A list to keep track of visited nodes to avoid revisiting.
    visited = []

    # A dictionary to store the shortest path from the origin to each node.
    shortest_paths = {}

    # Iterates until all nodes are visited.
    while len(visited) != len(graph.content):
        # Selects the unvisited node with the smallest distance from the origin.
        current_node = None
        # Initializes the current_node as None. This variable will hold the node
        # with the smallest distance from the origin that hasn't been visited yet.
        current_distance = float('inf')
        # Initializes current_distance with infinity. This variable represents the
        # smallest distance found so far from the origin to an unvisited node. It's
        # initially set to infinity because we haven't started comparing distances yet.
        for node in graph.content:
            # Iterates over each node in the graph's content. The graph.content dictionary
            # contains all the nodes of the graph as keys.  
            if distances[node] < current_distance and node not in visited:
                # Checks if the current node's distance from the origin is less than
                # the smallest distance found so far (current_distance) and if this node
                # has not been visited yet. The distances dictionary contains the current
                # shortest distances from the origin node to every other node.
                current_node = node
                current_distance = distances[node]

        # Breaks the loop if no unvisited node can be selected (isolated or unreachable nodes).
        if current_node is None:
            break

        # Marks the current node as visited.
        visited.append(current_node)

        # Updates the distances to neighboring nodes if a shorter path is found.
        for neighbor, weight in graph.content[current_node]:
            # Iterates over each neighbor of the current node. The neighbors are represented
            # as tuples containing the neighbor's identifier and the weight of the edge connecting
            # the current node to this neighbor.
            distance = int(current_distance) + int(weight)
            # Calculates the distance to this neighbor from the origin node by adding
            # the current distance to the current node (from the origin) and the weight
            # of the edge from the current node to this neighbor. The distance to the
            # current node is known (current_distance), and the weight of the edge to the
            # neighbor is given, so their sum represents the total distance from the origin
            # to this neighbor via the current node.
            if distance < distances[neighbor]:
                # Checks if this newly calculated distance to the neighbor is less than the
                # previously known shortest distance to this neighbor (stored in the distances
                # dictionary). This condition is true if the newly found path to the neighbor
                # is shorter than any previously known path.
                distances[neighbor] = distance
                # If a shorter path is found, updates the shortest known distance to this
                # neighbor in the distances dictionary. This step ensures that the algorithm
                # always keeps track of the shortest path to each node encountered so far.
                shortest_paths[neighbor] = current_node
                # This line records the current node as the direct predecessor (or "parent") of the
                # neighbor node in the context of the shortest path journey. By doing this, we're essentially
                # noting that the most efficient way to reach this neighbor node, as known so far, is via
                # the current node. This information is crucial for piecing together the sequence of nodes
                # that constitutes the shortest path from the starting point to any given node, once
                # the algorithm concludes.
        
    # Constructs the shortest path to the destiny node, if reachable.
    if distances[destiny] == float('inf'):
        return None, None  # Returns None if the destiny node is unreachable.

    # Initializes an empty list to store the path from the destiny node back to the origin node.
    path = []

    # Starts with the destiny node.
    node = destiny

    # Continues to loop until it reaches the origin node. This loop reconstructs the path
    # by moving from the destiny node up through its predecessors.
    while node != origin:
        # Appends the current node to the path list. Initially, this is the destiny node,
        # and in subsequent iterations, it will be the predecessor nodes leading back to the origin.
        path.append(node)

        # Updates the current node to its predecessor in the shortest path. The shortest_paths
        # dictionary maps each node to its predecessor on the shortest path from the origin node.
        node = shortest_paths[node]

    # Once the loop reaches the origin node (meaning the while condition fails),
    # it appends the origin node to the path. This is done outside the loop because
    # the loop terminates when 'node' is updated to 'origin', not including the origin
    # node itself in the path.
    path.append(origin)

    # The constructed path is in reverse order (from destiny to origin), so it's reversed
    # to present it from origin to destiny.
    path.reverse()

    # Returns the shortest path and its total distance if found.
    return path, distances[destiny]


def main():
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

    # Perform dijkstra search from the start node to the end node.
    path, distance = dijkstra(graph, start_node, end_node)

    print(f"\nPath: {path}")
    print(f"Distance: {distance}")

if __name__ == "__main__":
    main()
