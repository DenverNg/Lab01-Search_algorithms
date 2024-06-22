from collections import deque
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

    while frontier:
        current, predecessor = frontier.pop()
        visited[current] = predecessor

        while len(path) > 0 and path[-1] != visited[current]:
            path.pop()
        path.append(current)

        if current == end:
            print("-----------------")
            print(f"visited: {visited}")
            print(f"path: {path}")
            return visited, path

        for neighbor in range(len(matrix[current]) - 1, -1, -1):
            if matrix[current][neighbor] != 0 and neighbor not in visited:
                frontier.append((neighbor, current))

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
    while not pq.empty():
        _, node = pq.get()
        print(f"frontier: {frontier}")
        if node == end:
            break

        for neighbor in range(len(matrix[node])):
            weight = matrix[node][neighbor]
            if weight and neighbor not in visited:
                pq.put((weight, neighbor))
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

def DLS(matrix, start, end, limit):
    """
    Depth-Limited Search algorithm:
    Parameters:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    limit: integer
        maximum depth limit
        default limit = 2 (you can change it to any value you want in Animations.py file)
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node,
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """

    def recursive_dls(node, depth):
        if depth > limit:
            return False
        if node == end:
            return True
        for neighbor in range(len(matrix[node])):
            if matrix[node][neighbor] and neighbor not in visited:
                visited[neighbor] = node
                if recursive_dls(neighbor, depth + 1):
                    path.insert(0, neighbor)
                    return True
                # If not successful, backtrack
                visited.pop(neighbor)
        return False

    path = []
    visited = {start: None}

    print(f"Starting DLS with limit {limit}")

    if recursive_dls(start, 0):
        path.insert(0, start)
    else:
        print("No path found within depth limit.")

    print(f"path: {path}")
    print(f"visited: {visited}")
    return visited, path

def DLS_for_IDS(matrix, start, end, limit, visited, frontier):
    """
    Depth-Limited Search (DLS) for IDS
    Parameters:
    ---------------------------
    matrix: list of lists (2D array)
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    limit: integer
        maximum depth limit
    visited: dict
        The dictionary containing visited nodes with their parent nodes
    frontier: set
        The set of nodes being explored at the current depth limit
    
    Returns
    ---------------------
    bool
        True if target is found within limit, False otherwise
    """
    if start == end:
        return True
    if limit <= 0:
        return False

    for neighbor in range(len(matrix[start])):
        if matrix[start][neighbor] and neighbor not in visited:
            visited[neighbor] = start
            if DLS_for_IDS(matrix, neighbor, end, limit - 1, visited, frontier):
                return True
            frontier.add(neighbor)
    return False

def IDS(matrix, start, end):
    """
    Iterative Deepening Search (IDS) algorithm
    Parameters:
    ---------------------------
    matrix: list of lists (2D array)
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
    maxDepth = len(matrix) # Maximum depth is the number of nodes in the graph
    for depth in range(maxDepth + 1):
        visited = {start: None}
        frontier = {start}
        if DLS_for_IDS(matrix, start, end, depth, visited, frontier):
            path = []
            node = end
            while node is not None:
                path.insert(0, node)
                node = visited[node]
            print(f"path: {path}")
            print(f"visited: {visited}")
            return visited, path
        print(f"limit {depth}: {frontier}")
    return {}, []


def bidirectional_search(matrix, start, end):
    """
    Bidirectional Search algorithm
    Parameters:
    ---------------------------
    matrix: list of lists (2D array)
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited: string
        The string contains visited nodes from both searches:
        src_visited and dest_visited.
    path: list
        Found path from start to end
    """

    from collections import deque

    # Initialization
    n = len(matrix)
    if start == end:
        visited = f"{start}: None\n{end}: None"
        return visited, [start]

    src_queue = deque([start])
    dest_queue = deque([end])

    src_visited = {start: None}
    dest_visited = {end: None}

    print("Starting Bidirectional Search...")
    print(f"Searching path from {start} to {end}")
    step = 0
    # Perform BFS from both ends
    while src_queue and dest_queue:
        step += 1
        print(f"\nStep {step}")
        # Forward search from source
        if src_queue:
            print(f"Source queue: {src_queue}")
            current = src_queue.popleft()
            for neighbor, is_connected in enumerate(matrix[current]):
                if is_connected and neighbor not in src_visited:
                    src_queue.append(neighbor)
                    src_visited[neighbor] = current
                    # print(f"Source visited: {src_visited}")
                    
                    if neighbor in dest_visited:
                        # Path found
                        visited = _format_visited(src_visited, dest_visited)
                        path = _construct_path(src_visited, dest_visited, neighbor)
                        print(f"Source visited: {src_visited}")
                        print(f"Path found: {path}")
                        return visited, path
            print(f"Source visited: {src_visited}")
        # Backward search from destination
        if dest_queue:
            print(f"Destination queue: {dest_queue}")
            current = dest_queue.popleft()
            for neighbor, is_connected in enumerate(matrix[current]):
                if is_connected and neighbor not in dest_visited:
                    dest_queue.append(neighbor)
                    dest_visited[neighbor] = current
                    # print(f"Destination visited: {dest_visited}")
                    
                    if neighbor in src_visited:
                        # Path found
                        visited = _format_visited(src_visited, dest_visited)
                        path = _construct_path(src_visited, dest_visited, neighbor)
                        print(f"Destination visited: {dest_visited}")
                        print(f"Path found: {path}")
                        return visited, path
            print(f"Destination visited: {dest_visited}")
    # If no path found
    visited = _format_visited(src_visited, dest_visited)
    print("No path found.")
    return visited, []


def _construct_path(src_visited, dest_visited, intersecting_node):
    # Construct the path from source to destination via the intersecting node
    path = []

    # From source to intersection
    current = intersecting_node
    while current is not None:
        path.append(current)
        current = src_visited[current]
    path.reverse()

    # From intersection to destination
    current = dest_visited[intersecting_node]
    while current is not None:
        path.append(current)
        current = dest_visited[current]

    return path


def _format_visited(src_visited, dest_visited):
    # Format the visited dictionaries into a single string
    res = src_visited | dest_visited
    return res
