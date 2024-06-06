import numpy as np
from queue import PriorityQueue

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
    """
    # TODO: 
   
    path = []
    visited = {start: None}
    queue = [start]

    while queue:
        node = queue.pop(0)
        if node == end:
            break
        for neighbor, connected in enumerate(matrix[node]):
            if connected and neighbor not in visited:
                visited[neighbor] = node
                queue.append(neighbor)

    if end in visited:
        node = end
        while node is not None:
            path.insert(0, node)
            node = visited[node]

    return visited, path

def DFS(matrix, start, end):
    """
    DFS algorithm
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

    # TODO: 
    
    path = []
    visited = {start: None}
    stack = [start]

    while stack:
        node = stack.pop()
        if node == end:
            break
        for neighbor in range(len(matrix[node]) - 1, -1, -1):
            if matrix[node][neighbor] and neighbor not in visited:
                visited[neighbor] = node
                stack.append(neighbor)

    if end in visited:
        node = end
        while node is not None:
            path.insert(0, node)
            node = visited[node]
   
    return visited, path


def UCS(matrix, start, end):
    """
    Uniform Cost Search algorithm
     Parameters:visited
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
    # TODO:  
    path = []
    visited = {start: None}
    pq = PriorityQueue()
    pq.put((0, start))
    cost = {start: 0}

    while not pq.empty():
        current_cost, node = pq.get()
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
    # TODO: 
    
    path = []
    visited = {start: None}
    pq = PriorityQueue()
    pq.put((0, start))

    while not pq.empty():
        _, node = pq.get()
        if node == end:
            break
        for neighbor, weight in enumerate(matrix[node]):
            if weight and neighbor not in visited:
                visited[neighbor] = node
                heuristic = weight  # Assuming heuristic is the weight itself
                pq.put((heuristic, neighbor))

    if end in visited:
        node = end
        while node is not None:
            path.insert(0, node)
            node = visited[node]
            
    return visited, path

def Astar(matrix, start, end, pos):
    """
    A* Search algorithm
    heuristic: eclid distance based positions parameter
     Parameters:
    ---------------------------
    matrix: np array UCS
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
    # TODO: 

    def heuristic(a, b):
        return np.linalg.norm(np.array(pos[a]) - np.array(pos[b]))

    path = []
    visited = {start: None}
    pq = PriorityQueue()
    pq.put((0, start))
    cost = {start: 0}

    while not pq.empty():
        _, node = pq.get()
        if node == end:
            break
        for neighbor, weight in enumerate(matrix[node]):
            if weight:
                new_cost = cost[node] + weight
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, end)
                    pq.put((priority, neighbor))
                    visited[neighbor] = node

    if end in visited:
        node = end
        while node is not None:
            path.insert(0, node)
            node = visited[node]
            
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
    
    while not pq.empty():
        current_cost, node = pq.get()
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