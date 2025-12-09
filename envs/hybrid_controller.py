"""
Hybrid A* + RL Controller for Real-World Drone Delivery.
Combines global planning (A*) with local control (RL).
"""
import numpy as np
from typing import List, Tuple, Optional
from envs.astar_planner import astar_3d


class HybridController:
    """
    Hybrid A* planning + RL control.
    
    Use case:
    - A* plans global path around static obstacles (buildings)
    - RL handles local control and dynamic obstacles (birds, wind)
    """
    
    def __init__(self, rl_model, grid_limit=10, waypoint_threshold=0.8):
        """
        Args:
            rl_model: Trained RL model (PPO/SAC/etc)
            grid_limit: Environment bounds
            waypoint_threshold: Distance to consider waypoint reached
        """
        self.rl_model = rl_model
        self.grid_limit = grid_limit
        self.waypoint_threshold = waypoint_threshold
        
        self.global_path = None
        self.current_waypoint_idx = 0
        self.replan_threshold = 2.0
        
    def plan_global_path(self, start: np.ndarray, goal: np.ndarray, 
                        static_obstacles: List[np.ndarray]) -> Optional[List[np.ndarray]]:
        """Use A* to plan global path around static obstacles."""
        print(f" Planning global path from {start} to {goal}...")
        
        path = astar_3d(
            start, goal, static_obstacles, self.grid_limit,
            obstacle_radius=0.7, resolution=0.5
        )
        
        if path is None:
            print(" No feasible path found!")
            return None
        
        self.global_path = path
        self.current_waypoint_idx = 0
        print(f" Global path planned: {len(path)} waypoints")
        return path
    
    def get_current_waypoint(self) -> Optional[np.ndarray]:
        """Get current target waypoint."""
        if self.global_path is None or self.current_waypoint_idx >= len(self.global_path):
            return None
        return self.global_path[self.current_waypoint_idx]
    
    def advance_waypoint(self, current_pos: np.ndarray) -> bool:
        """Check if waypoint reached and advance."""
        waypoint = self.get_current_waypoint()
        if waypoint is None:
            return True
        
        distance = np.linalg.norm(current_pos - waypoint)
        
        if distance < self.waypoint_threshold:
            self.current_waypoint_idx += 1
            
            if self.current_waypoint_idx >= len(self.global_path):
                print(" Final goal reached!")
                return True
            else:
                print(f"✓ Waypoint {self.current_waypoint_idx}/{len(self.global_path)} reached")
                return False
        
        return False
    
    def get_rl_action(self, observation: np.ndarray, 
                     current_waypoint: np.ndarray) -> np.ndarray:
        """Use RL to generate control action toward current waypoint."""
        # Modify observation to target current waypoint instead of final goal
        modified_obs = observation.copy()
        
        # For RealisticDroneEnv: [x,y,z,yaw,vx,vy,vz,target_x,target_y,target_z,closest_obs,obs_count]
        if len(modified_obs) == 12:
            modified_obs[7:10] = current_waypoint
        # For DroneEnvContinuous: [x,y,z,yaw,target_x,target_y,target_z]
        elif len(modified_obs) == 7:
            modified_obs[4:7] = current_waypoint
        
        action, _ = self.rl_model.predict(modified_obs, deterministic=True)
        return action
    
    def should_replan(self, current_pos: np.ndarray) -> bool:
        """Check if replanning needed due to deviation."""
        if self.global_path is None:
            return False
        
        waypoint = self.get_current_waypoint()
        if waypoint is None:
            return False
        
        distance_to_waypoint = np.linalg.norm(current_pos - waypoint)
        
        if distance_to_waypoint > self.replan_threshold:
            print("  Significant deviation detected, replanning...")
            return True
        
        return False
    
    def step(self, observation: np.ndarray, static_obstacles: List[np.ndarray],
            goal: np.ndarray) -> Tuple[np.ndarray, bool, dict]:
        """
        Execute one step of hybrid control.
        
        Returns:
            action: Control action [vx, vy, vz, yaw_rate]
            done: Whether goal is reached
            info: Additional information
        """
        current_pos = observation[:3]
        
        # Initial planning if no path exists
        if self.global_path is None:
            success = self.plan_global_path(current_pos, goal, static_obstacles)
            if not success:
                return np.zeros(4), True, {"status": "no_path"}
        
        # Check if replanning needed
        if self.should_replan(current_pos):
            self.plan_global_path(current_pos, goal, static_obstacles)
        
        # Get current waypoint
        current_waypoint = self.get_current_waypoint()
        
        if current_waypoint is None:
            return np.zeros(4), True, {"status": "goal_reached"}
        
        # Use RL to generate action toward waypoint
        action = self.get_rl_action(observation, current_waypoint)
        
        # Check if waypoint reached
        goal_reached = self.advance_waypoint(current_pos)
        
        info = {
            "status": "navigating",
            "current_waypoint": current_waypoint,
            "waypoint_idx": self.current_waypoint_idx,
            "total_waypoints": len(self.global_path) if self.global_path else 0,
            "distance_to_waypoint": np.linalg.norm(current_pos - current_waypoint)
        }
        
        return action, goal_reached, info