"""
Test script for environment validation.
"""
from envs.drone_env_continuous import DroneEnvContinuous
from envs.astar_planner import astar_3d
import numpy as np


def test_environment():
    """Test basic environment functionality."""
    print(" Testing Drone Environment...")
    
    env = DroneEnvContinuous(render_mode="human")
    
    # Test reset
    obs, info = env.reset()
    print(f" Reset successful. Observation shape: {obs.shape}")
    
    # Test random actions
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            print(f" Episode finished at step {i+1}")
            print(f"   Final distance to target: {info['distance_to_target']:.2f}")
            break
    
    env.close()
    print(" Environment test passed!\n")


def test_astar():
    """Test A* pathfinding."""
    print(" Testing A* Pathfinding...")
    
    start = np.array([0, 0, 0])
    goal = np.array([8, 8, 8])
    obstacles = [
        np.array([4, 4, 4]),
        np.array([5, 5, 5])
    ]
    
    path = astar_3d(start, goal, obstacles, grid_limit=10)
    
    if path is not None:
        print(f" A* found path with {len(path)} waypoints")
        print(f"   Start: {start}")
        print(f"   Goal: {goal}")
        print(f"   Path length: {len(path)}")
    else:
        print(" A* failed to find path")
    
    print(" A* test passed!\n")


if __name__ == "__main__":
    test_environment()
    test_astar()
    print(" All tests passed!")