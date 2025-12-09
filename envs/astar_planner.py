"""
A* pathfinding algorithm for 3D drone navigation.
"""
import numpy as np
from heapq import heappush, heappop


def astar_3d(start, goal, obstacles, grid_limit, obstacle_radius=0.5, resolution=1.0):
    """
    3D A* pathfinding with obstacle avoidance.
    
    Args:
        start: Starting position [x, y, z]
        goal: Goal position [x, y, z]
        obstacles: List of obstacle positions
        grid_limit: Maximum coordinate value
        obstacle_radius: Collision radius
        resolution: Grid resolution
        
    Returns:
        List of waypoints or None
    """
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    
    def is_collision(pos):
        for obs in obstacles:
            if np.linalg.norm(np.array(pos) - obs) < obstacle_radius:
                return True
        return False
    
    def is_valid(pos):
        return all(0 <= p <= grid_limit for p in pos)
    
    # Discretize start and goal
    start = tuple(np.clip(np.round(np.array(start) / resolution), 0, grid_limit).astype(int))
    goal = tuple(np.clip(np.round(np.array(goal) / resolution), 0, grid_limit).astype(int))
    
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    
    # 26-connected neighborhood
    directions = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0:
                    continue
                directions.append((dx, dy, dz))
    
    explored = 0
    max_iterations = 10000
    
    while open_set and explored < max_iterations:
        explored += 1
        _, current = heappop(open_set)
        
        if heuristic(current, goal) < 1.5:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(np.array(current) * resolution)
                current = came_from[current]
            path.append(np.array(start) * resolution)
            return path[::-1]
        
        for dx, dy, dz in directions:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            
            if not is_valid(neighbor):
                continue
            
            neighbor_pos = np.array(neighbor) * resolution
            if is_collision(neighbor_pos):
                continue
            
            move_cost = np.sqrt(dx**2 + dy**2 + dz**2)
            tentative_g = g_score[current] + move_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heappush(open_set, (f_score, neighbor))
    
    return None


class AStarController:
    """Controller that follows A* path."""
    
    def __init__(self, path, speed=1.0):
        self.path = path
        self.current_waypoint = 0
        self.speed = speed
    
    def get_action(self, current_pos):
        """Get action to follow path."""
        if self.path is None or self.current_waypoint >= len(self.path):
            return np.zeros(4)
        
        target = self.path[self.current_waypoint]
        direction = target - current_pos
        distance = np.linalg.norm(direction)
        
        if distance < 0.5:
            self.current_waypoint += 1
            if self.current_waypoint >= len(self.path):
                return np.zeros(4)
            target = self.path[self.current_waypoint]
            direction = target - current_pos
            distance = np.linalg.norm(direction)
        
        if distance > 0.01:
            velocity = (direction / distance) * self.speed
        else:
            velocity = np.zeros(3)
        
        return np.append(velocity, 0.0)