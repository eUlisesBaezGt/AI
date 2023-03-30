def branch_and_bound(graph, heuristics, origin, destiny):
    if origin == destiny:
        return [origin]

    # Initialize the queue with the origin node
    queue = [(heuristics.content[origin][0][1], [origin])]
    visited = set()

    while queue:
        # Get the current path with the smallest estimated cost
        path_cost, path = min(queue, key=lambda x: x[0])
        queue.remove((path_cost, path))
        current_node = path[-1]

        # Check if the current node has already been visited
        if current_node in visited:
            continue

        visited.add(current_node)

        # Check if the current node is the goal node
        if current_node == destiny:
            return path

        # Explore the neighbors of the current node
        for neighbor, weight in graph.content[current_node]:
            if neighbor not in visited:
                # Calculate the estimated cost of the path to the neighbor
                neighbor_path = path + [neighbor]
                neighbor_cost = path_cost + weight + heuristics.content[neighbor][0][1]

                # Add the path to the queue
                queue.append((neighbor_cost, neighbor_path))

    # No path was found
    return None
