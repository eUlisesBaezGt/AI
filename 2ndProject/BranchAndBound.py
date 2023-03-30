def branch_and_bound(graph, heuristics, origin, destiny):
    if origin == destiny:
        return [origin]

    queue = [(heuristics.content[origin][0][1], [origin])]
    visited = set()

    while queue:
        path_cost, path = min(queue, key=lambda x: x[0])
        queue.remove((path_cost, path))
        current_node = path[-1]

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == destiny:
            return path

        for neighbor, weight in graph.content[current_node]:
            if neighbor not in visited:
                neighbor_path = path + [neighbor]
                neighbor_cost = path_cost + weight + heuristics.content[neighbor][0][1]

                queue.append((neighbor_cost, neighbor_path))

    return None
