def a_star_search(graph, heuristics, start, goal):
    if start == goal:
        return [start]

    frontier = []
    explored = set()
    parents = {}
    costs = {}

    frontier.append((start, 0))
    parents[start] = None
    costs[start] = 0

    while frontier:
        frontier.sort(key=lambda x: x[1] + int(heuristics.content[x[0]][0][1]))
        current, current_cost = frontier.pop(0)

        if current == goal:
            path = []
            total_cost = current_cost
            while current is not None:
                path.append(current)
                current = parents[current]
            return path[::-1], total_cost

        explored.add(current)

        for neighbor, weight in graph.content[current]:
            if neighbor not in explored:
                new_cost = current_cost + int(weight)
                if neighbor not in frontier or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    frontier.append((neighbor, new_cost))
                    parents[neighbor] = current

    return None, None
