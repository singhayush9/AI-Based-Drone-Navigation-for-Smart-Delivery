"""
Realistic Drone Environment for Real-World Delivery Training.
Features: Variable obstacles (2-12), curriculum learning, realistic rewards.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class RealisticDroneEnv(gym.Env):
    """
    Realistic drone environment for smart delivery.
    
    Features:
    - Variable obstacles: 2-12 (residential to dense urban)
    - Variable obstacle sizes: Birds (0.3m) to buildings (0.8m)
    - Curriculum learning: Adaptive difficulty
    - Realistic rewards: Delivery success, energy efficiency
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        grid_limit=10,
        min_obstacles=2,
        max_obstacles=12,
        max_steps=400,
        step_scale=0.5,
        max_lin_vel=3.0,
        max_yaw_rate=1.5,
        curriculum_learning=True,
        render_mode=None
    ):
        super().__init__()
        
        self.grid_limit = grid_limit
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.max_steps = max_steps
        self.step_scale = step_scale
        self.max_lin_vel = max_lin_vel
        self.max_yaw_rate = max_yaw_rate
        self.curriculum_learning = curriculum_learning
        self.render_mode = render_mode
        
        # Curriculum tracking
        self.episode_count = 0
        self.success_count = 0
        self.current_difficulty = 0.0
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # Enhanced observation: [x,y,z,yaw,vx,vy,vz,target_x,target_y,target_z,closest_obs_dist,obs_count]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -np.pi, -5, -5, -5, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([grid_limit, grid_limit, grid_limit, np.pi, 
                          5, 5, 5, grid_limit, grid_limit, grid_limit, 
                          grid_limit, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.reset()
    
    def _get_obstacle_count(self) -> int:
        """Determine obstacle count based on curriculum."""
        if self.curriculum_learning:
            difficulty = min(self.current_difficulty, 1.0)
            min_obs = self.min_obstacles
            max_obs = int(self.min_obstacles + difficulty * (self.max_obstacles - self.min_obstacles))
        else:
            min_obs = self.min_obstacles
            max_obs = self.max_obstacles
        
        return np.random.randint(min_obs, max_obs + 1)
    
    def _generate_obstacles(self):
        """Generate variable obstacles with different sizes."""
        n_obstacles = self._get_obstacle_count()
        self.obstacles = []
        self.obstacle_radii = []
        
        for i in range(n_obstacles):
            obs = np.random.uniform(2, self.grid_limit - 2, size=3)
            
            # Variable sizes: birds (0.3), poles (0.5), buildings (0.8)
            rand = np.random.random()
            if rand < 0.3:
                radius = 0.3  # Small obstacles
            elif rand < 0.7:
                radius = 0.5  # Medium obstacles
            else:
                radius = 0.8  # Large obstacles
            
            self.obstacles.append(obs)
            self.obstacle_radii.append(radius)
    
    def _update_curriculum(self, success: bool):
        """Update difficulty based on performance."""
        if not self.curriculum_learning:
            return
        
        self.episode_count += 1
        if success:
            self.success_count += 1
        
        if self.episode_count >= 10:
            recent_success_rate = self.success_count / min(self.episode_count, 100)
            
            # Increase difficulty if success rate > 70%
            if recent_success_rate > 0.7:
                self.current_difficulty = min(1.0, self.current_difficulty + 0.05)
            # Decrease if success rate < 30%
            elif recent_success_rate < 0.3:
                self.current_difficulty = max(0.0, self.current_difficulty - 0.05)
            
            # Log progress every 100 episodes
            if self.episode_count % 100 == 0:
                avg_obstacles = self.min_obstacles + self.current_difficulty * (self.max_obstacles - self.min_obstacles)
                print(f" Curriculum Update (Episode {self.episode_count}):")
                print(f"   Success Rate: {recent_success_rate*100:.1f}%")
                print(f"   Difficulty: {self.current_difficulty*100:.1f}%")
                print(f"   Avg Obstacles: {avg_obstacles:.1f}")
                self.success_count = 0
    
    def reset(self, seed=None, options=None):
        """Reset environment with variable obstacles."""
        super().reset(seed=seed)
        
        self.state = np.array([
            np.random.uniform(0, 2),
            np.random.uniform(0, 2),
            np.random.uniform(1, 3),  # Start at safe altitude
            0.0  # yaw
        ], dtype=np.float32)
        
        self.target = np.random.uniform(
            self.grid_limit - 3,
            self.grid_limit - 1,
            size=3
        ).astype(np.float32)
        
        self._generate_obstacles()
        self.velocity = np.zeros(3, dtype=np.float32)
        self.steps = 0
        self.trajectory = [self.state[:3].copy()]
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Enhanced observation with obstacle info."""
        if self.obstacles:
            distances = [np.linalg.norm(self.state[:3] - obs) for obs in self.obstacles]
            closest_obstacle_dist = min(distances)
        else:
            closest_obstacle_dist = self.grid_limit
        
        obstacle_count_normalized = len(self.obstacles) / self.max_obstacles
        
        obs = np.concatenate([
            self.state,  # [x, y, z, yaw]
            self.velocity,  # [vx, vy, vz]
            self.target,  # [target_x, target_y, target_z]
            [closest_obstacle_dist, obstacle_count_normalized]
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action):
        """Execute step with realistic dynamics."""
        vx, vy, vz, yaw_rate = np.clip(action, -1, 1)
        vx *= self.max_lin_vel
        vy *= self.max_lin_vel
        vz *= self.max_lin_vel
        yaw_rate *= self.max_yaw_rate
        
        yaw = self.state[3] + yaw_rate * 0.1
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
        
        dx = vx * np.cos(yaw) - vy * np.sin(yaw)
        dy = vx * np.sin(yaw) + vy * np.cos(yaw)
        dz = vz
        
        move = np.array([dx, dy, dz]) * self.step_scale
        self.velocity = 0.9 * self.velocity + 0.1 * move
        next_pos = np.clip(self.state[:3] + self.velocity, 0, self.grid_limit)
        
        d_old = np.linalg.norm(self.state[:3] - self.target)
        d_new = np.linalg.norm(next_pos - self.target)
        
        # Realistic delivery rewards
        reward = 0.0
        reward += (d_old - d_new) * 10.0  # Progress reward
        reward -= 0.05 * np.linalg.norm(action)  # Energy efficiency
        reward -= 0.02  # Time penalty
        
        # Safety: Altitude penalties
        if next_pos[2] < 2.0:
            reward -= 0.5  # Too low (unsafe)
        elif next_pos[2] > 8.0:
            reward -= 0.3  # Too high (energy waste)
        
        done = False
        terminated = False
        truncated = False
        collision = False
        
        # Check collisions with variable-sized obstacles
        for obs, radius in zip(self.obstacles, self.obstacle_radii):
            if np.linalg.norm(next_pos - obs) < radius:
                reward -= 100.0
                done = True
                terminated = True
                collision = True
                break
        
        # Goal reached
        if np.linalg.norm(next_pos - self.target) < 0.5:
            reward += 200.0  # Delivery success!
            done = True
            terminated = True
        
        self.state[:3] = next_pos
        self.state[3] = yaw
        self.steps += 1
        self.trajectory.append(next_pos.copy())
        
        if self.steps >= self.max_steps:
            done = True
            truncated = True
        
        # Update curriculum
        if done:
            success = terminated and not collision and reward > 0
            self._update_curriculum(success)
        
        info = {
            "distance_to_target": np.linalg.norm(next_pos - self.target),
            "steps": self.steps,
            "collision": collision,
            "num_obstacles": len(self.obstacles),
            "difficulty": self.current_difficulty
        }
        
        return self._get_obs(), float(reward), terminated, truncated, info
    
    def render(self):
        """Render environment."""
        if self.render_mode is None:
            return
        
        if not hasattr(self, 'fig') or self.fig is None:
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.ion()
        
        self.ax.clear()
        
        # Plot obstacles with size-based colors
        for obs, radius in zip(self.obstacles, self.obstacle_radii):
            if radius > 0.6:
                color = 'red'  # Buildings
            elif radius > 0.4:
                color = 'orange'  # Poles
            else:
                color = 'yellow'  # Birds
            self.ax.scatter(*obs, c=color, s=radius*500, marker='o', alpha=0.6)
        
        self.ax.scatter(*self.target, c='green', s=400, marker='*', label='Delivery Target')
        self.ax.scatter(*self.state[:3], c='blue', s=200, marker='^', label='Drone')
        
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b--', linewidth=1.5, alpha=0.7)
        
        self.ax.set_xlim(0, self.grid_limit)
        self.ax.set_ylim(0, self.grid_limit)
        self.ax.set_zlim(0, self.grid_limit)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        title = f'Realistic Delivery (Step {self.steps}/{self.max_steps})\n'
        title += f'Obstacles: {len(self.obstacles)}, Difficulty: {self.current_difficulty*100:.0f}%'
        self.ax.set_title(title)
        self.ax.legend()
        
        plt.draw()
        plt.pause(0.01)
    
    def close(self):
        """Close rendering."""
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)
            self.fig = None