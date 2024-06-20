import numpy as np
from queue import PriorityQueue

def construct_path(visited, end):
    """
    Constructs the path from the start node to the end node using the visited dictionary.
    
    Parameters:
    ---------------------------
    visited: dictionary
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    end: integer
        The end node for which the path needs to be constructed.
    
    Returns
    ---------------------
    visited: dictionary
        The same visited dictionary that was passed as input.
    path: list
        The constructed path from the start node to the end node.
    """
    path = []
    current = end
    while current is not None:
        path.insert(0, current)
        current = visited[current]
    print(f"path: {path}")
    print(f"visited: {visited}")
    return visited, path

def BFS(matrix, start, end):
    """
    BFS algorithm:
    Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    --------------------------------
    Ref: https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/
    """
    frontier = [start]
    visited = {start: None}
    path = []

    print(f"frontier: {frontier}")

    while frontier:
        current = frontier.pop(0)
        if current == end:
            visited, path = construct_path(visited, end)
            return visited, path

        for neighbor in range(len(matrix[current])):
            if matrix[current][neighbor] and neighbor == end:
                visited[neighbor] = current
                visited, path = construct_path(visited, end)
                return visited, path  
            if matrix[current][neighbor] and neighbor not in visited:
                visited[neighbor] = current
                frontier.append(neighbor)
        print(f"frontier: {frontier}")

    print(f"visited: {visited}")
    return visited, path

def DFS(matrix, start, end):
    """
    DFS algorithm:
    Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited 
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    --------------------------------
    Ref: https://www.geeksforgeeks.org/iterative-depth-first-traversal/
    """
    frontier = [(start, None)]
    visited = {}
    path = []

    #print(f"frontier: {frontier}")

    while frontier:
        current, predecessor = frontier.pop()
        visited[current] = predecessor

        while len(path) > 0 and path[-1] != visited[current]:
            path.pop()
        path.append(current)

        if current == end:
            print("-----------------")
            #print(f"frontier: {frontier}")
            print(f"visited: {visited}")
            print(f"path: {path}")
            return visited, path

        for neighbor in range(len(matrix[current]) - 1, -1, -1):
            if matrix[current][neighbor] != 0 and neighbor not in visited:
                frontier.append((neighbor, current))

        #print(f"frontier: {frontier}")

    print(f"visited: {visited}")
    return visited, path


def UCS(matrix, start, end):
    """
    Uniform Cost Search (UCS) algorithm
    Parameters:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    path = []
    visited = {start: None}
    pq = PriorityQueue()
    pq.put((0, start))
    cost = {start: 0}

    frontier = [(start, 0)]
    print(f"frontier: {frontier}")

    while not pq.empty():
        current_cost, node = pq.get()
        frontier = sorted([(n, c) for c, n in pq.queue], key=lambda x: x[1])
        print(f"frontier: {frontier}" + f" queue: {pq.queue}")
        if node == end:
            break
        for neighbor, weight in enumerate(matrix[node]):
            if weight:
                new_cost = current_cost + weight
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    pq.put((new_cost, neighbor))
                    visited[neighbor] = node

    if end in visited:
        node = end
        while node is not None:
            path.insert(0, node)
            node = visited[node]

    print(f"path: {path}")
    print(f"visited: {visited}")
    return visited, path


def GBFS1(matrix, start, end, pos):
    """
    Greedy Best First Search algorithm
    heuristic : Euclidean distance based on positions parameter
    Parameters:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    pos: dictionary. keys are nodes, values are positions
        positions of graph nodes
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node,
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    path = []
    visited = {start: None}
    pq = PriorityQueue()
    pq.put((heuristic(start, end, pos), start))

    frontier = [(start, heuristic(start, end, pos))]
    print(f"frontier: {frontier}")

    while not pq.empty():
        _, node = pq.get()
        frontier = sorted([(n, c) for c, n in pq.queue], key=lambda x: x[1])
        print(f"frontier: {frontier}")
        if node == end:
            break
        for neighbor, connected in enumerate(matrix[node]):
            if connected and neighbor not in visited:
                visited[neighbor] = node
                pq.put((heuristic(neighbor, end, pos), neighbor))

    if end in visited:
        node = end
        while node is not None:
            path.insert(0, node)
            node = visited[node]

    print(f"path: {path}")
    print(f"visited: {visited}")
    return visited, path


def GBFS(matrix, start, end):
    """
    Greedy Best First Search algorithm
    heuristic : edge weights
    Parameters:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node,
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    path = []
    visited = {start: None}
    pq = PriorityQueue()
    pq.put((0, start))

    frontier = [(start, 0)]
    #print(f"frontier: {frontier}")
    while not pq.empty():
        _, node = pq.get()
        # frontier = [(n, c) for c, n in pq.queue]
        print(f"frontier: {frontier}")
        if node == end:
            break

        for neighbor in range(len(matrix[node])):
            weight = matrix[node][neighbor]
            if weight and neighbor not in visited:
                pq.put((weight, neighbor))
                visited[neighbor] = node
        #frontier = [(n, c) for c, n in pq.queue]
        frontier = sorted([(n, c) for c, n in pq.queue], key=lambda x: x[1])

    if end in visited:
        node = end
        while node is not None:
            path.insert(0, node)
            node = visited[node]

    print(f"path: {path}")
    print(f"visited: {visited}")
    return visited, path

# def heuristic(node, end, pos):
#     # Euclidean distance as heuristic
#     return np.linalg.norm(np.array(pos[node]) - np.array(pos[end]))


def heuristic(node, end, pos):
    # Euclidean distance as heuristic
    x1, y1 = pos[node]
    x2, y2 = pos[end]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def Astar(matrix, start, end, pos):
    """
    A* Search algorithm
    heuristic: Euclidean distance based on positions parameter
    Parameters:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    pos: dictionary. keys are nodes, values are positions
        positions of graph nodes
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    path = []
    visited = {start: None}
    pq = PriorityQueue()
    pq.put((0, start))
    cost = {start: 0}

    frontier = [(start, 0)]
    #print(f"frontier: {frontier}")

    while not pq.empty():
        current_cost, node = pq.get()
        # frontier = sorted([(n, c) for c, n in pq.queue], key=lambda x: x[1])
        print(f"frontier: {frontier}")
        if node == end:
            break
        for neighbor, weight in enumerate(matrix[node]):
            if weight:
                new_cost = cost[node] + weight
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, end, pos)
                    pq.put((priority, neighbor))
                    visited[neighbor] = node
        frontier = sorted([(n, c) for c, n in pq.queue], key=lambda x: x[1])


    if end in visited:
        node = end
        while node is not None:
            path.insert(0, node)
            node = visited[node]

    print(f"path: {path}")
    print(f"visited: {visited}")
    return visited, path


def Dijkstra(matrix, start, end):
    """
    Dijkstra's algorithm
    Parameters:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes, each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    path = []
    visited = {start: None}
    pq = PriorityQueue()
    pq.put((0, start))
    cost = {start: 0}

    frontier = [(start, 0)]
    print(f"frontier: {frontier}")

    while not pq.empty():
        current_cost, node = pq.get()
        frontier = sorted([(n, c) for c, n in pq.queue], key=lambda x: x[1])
        print(f"frontier: {frontier}")
        if node == end:
            break
        for neighbor, weight in enumerate(matrix[node]):
            if weight:
                new_cost = current_cost + weight
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    pq.put((new_cost, neighbor))
                    visited[neighbor] = node

    if end in visited:
        node = end
        while node is not None:
            path.insert(0, node)
            node = visited[node]

    print(f"path: {path}")
    print(f"visited: {visited}")
    return visited, path

# def DLS(matrix, start, end, limit):
#     """
#     Depth-Limited Search (DLS) algorithm
#     Parameters:
#     ---------------------------
#     matrix: np array 
#         The graph's adjacency matrix
#     start: integer 
#         starting node
#     end: integer
#         ending node
#     limit: integer
#         depth limit
    
#     Returns
#     ---------------------
#     visited 
#         The dictionary contains visited nodes: each key is a visited node, 
#         each value is the key's adjacent node which is visited before key.
#     path: list
#         Founded path
#     """
#     path = []
#     visited = {start: None}
    
#     def recursive_dls(node, depth):
#         if node == end:
#             return True
#         if depth == limit:
#             return False
#         for neighbor, connected in enumerate(matrix[node]):
#             if connected and neighbor not in visited:
#                 visited[neighbor] = node
#                 if recursive_dls(neighbor, depth + 1):
#                     return True
#         return False
    
#     recursive_dls(start, 0)
    
#     if end in visited:
#         node = end
#         while node is not None:
#             path.insert(0, node)
#             node = visited[node]
    
#     return visited, path