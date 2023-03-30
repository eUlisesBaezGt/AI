def greedy_best_first_search(graph, heuristics, start, goal):
    if start == goal:
        return [start]

    frontier = []
    explored = set()
    parents = {}

    frontier.append(start)
    parents[start] = None

    while frontier:
        frontier.sort(key=lambda x: heuristics.content[x][0][1])
        current = frontier.pop(0)

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parents[current]
            return path[::-1]

        explored.add(current)

        for neighbor, _ in graph.content[current]:
            if neighbor not in explored and neighbor not in frontier:
                frontier.append(neighbor)
                parents[neighbor] = current

    return None
