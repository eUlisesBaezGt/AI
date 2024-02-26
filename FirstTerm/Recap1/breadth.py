from KAGraph import KAGraph

def breadth_first_search(graph, origin, destiny):
    # Perform a breadth-first search from the origin to the destiny node.
    queue = [[origin]]  # Initialize a queue with the origin node as the first path.
    # En BFS, la cola se utiliza para mantener un registro de los caminos que necesitamos explorar.
    while queue:
        # Se entra en un bucle while que continúa mientras haya caminos en la cola para explorar. 
        # En cada iteración del bucle, se extrae el primer camino de la cola para su exploración.
        path = queue.pop(0)  # Dequeue the first path.
        node = path[-1]  # Get the last node from the path.
        if node == destiny:
            # Se verifica si el último nodo en el camino actual es el nodo destino.
            # Si lo es, significa que se ha encontrado un camino desde el origen hasta el destino, 
            # y este camino se imprime y devuelve como resultado.
            # If the current node is the destiny, print and return the path.
            print("\n\nPath found:\n-----------")
            print(f"Path: {path}")
            # return path
        # For each neighbor of the current node, construct a new path and enqueue it.
        for neighbor in graph.content[node]:
            # Si el último nodo del camino actual no es el nodo destino, se procede a explorar sus vecinos. 
            # Para cada vecino, se crea un nuevo camino añadiendo este vecino al camino actual y 
            # se encola este nuevo camino para su futura exploración. Esto asegura que todos los 
            # posibles caminos desde el origen se exploren de manera gradual, expandiéndose en amplitud.
            new_path = list(path)
            new_path.append(neighbor[0])
            queue.append(new_path)
    return None  # Return None if no path is found.

def main():
    # Main function to demonstrate the functionality.
    graph = KAGraph()
    with open("data.txt") as file:
        lines = file.readlines()

    # Read graph edges from 'data.txt' and add them to the graph.
    for line in lines[1:]:  # Skip the first line (contains the number of nodes and edges).
        origin, destiny, weight = line.strip().split()
        graph.new_edge(origin, destiny, weight)

    # Display the graph's adjacency list.
    graph.view_all()

    # Ask the user for the start and end nodes for the breadth-first search.
    start_node = input("Please enter the start node: ")
    end_node = input("Please enter the end node: ")

    # Perform breadth-first search with the user-specified nodes.
    breadth_first_search(graph, start_node, end_node)

if __name__ == "__main__":
    main()
