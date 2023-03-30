def beam_search(graph, heuristics, start, goal, k):
    if start == goal:
        return [start]

    beam = [start]
    parents = {start: None}
    explored = set()

    while beam:
        new_beam = []
        for node in beam:
            explored.add(node)
            for neighbor, _ in graph.content[node]:
                if neighbor not in explored:
                    if neighbor == goal:
                        path = [goal, node]
                        while parents[node] is not None:
                            node = parents[node]
                            path.append(node)
                        return path[::-1]
                    new_beam.append(neighbor)
                    parents[neighbor] = node

        new_beam.sort(key=lambda x: heuristics.content[x][0][1])
        beam = new_beam[:k]

    return None
