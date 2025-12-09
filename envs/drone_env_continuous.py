"""
Basic Continuous 3D Drone Navigation Environment.
Use realistic_drone_env.py for production training.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DroneEnvContinuous(gym.Env):
    """Basic continuous 3D drone control."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        grid_limit=10,
        obstacle_count=4,
        max_steps=300,
        step_scale=0.5,
        max_lin_vel=3.0,
        max_yaw_rate=1.5,
        obstacle_radius=0.5,
        target_threshold=0.5,
        render_mode=None
    ):
        super().__init__()
        
        self.grid_limit = grid_limit
        self.obstacle_count = obstacle_count
        self.max_steps = max_steps
        self.step_scale = step_scale
        self.max_lin_vel = max_lin_vel
        self.max_yaw_rate = max_yaw_rate
        self.obstacle_radius = obstacle_radius
        self.target_threshold = target_threshold
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -np.pi, 0, 0, 0], dtype=np.float32),
            high=np.array([grid_limit, grid_limit, grid_limit, np.pi, 
                          grid_limit, grid_limit, grid_limit], dtype=np.float32),
            dtype=np.float32
        )

        self.state = None
        self.target = None
        self.obstacles = []
        self.velocity = None
        self.steps = 0
        self.trajectory = []
        self.fig = None
        self.ax = None

    def _generate_obstacles(self):
        """Generate random obstacles."""
        self.obstacles = []
        for _ in range(self.obstacle_count):
            obs = np.random.uniform(2, self.grid_limit - 2, size=3)
            self.obstacles.append(obs)

    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        self.state = np.array([
            np.random.uniform(0, 2),
            np.random.uniform(0, 2),
            np.random.uniform(0, 2),
            0.0
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
        """Get observation."""
        return np.concatenate((self.state, self.target)).astype(np.float32)

    def step(self, action):
        """Execute step."""
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
        
        reward = (d_old - d_new) * 10.0
        reward -= 0.1 * np.linalg.norm(action)
        reward -= 0.05

        done = False
        terminated = False
        truncated = False

        for obs in self.obstacles:
            if np.linalg.norm(next_pos - obs) < self.obstacle_radius:
                reward -= 50.0
                done = True
                terminated = True
                break

        if np.linalg.norm(next_pos - self.target) < self.target_threshold:
            reward += 100.0
            done = True
            terminated = True

        self.state[:3] = next_pos
        self.state[3] = yaw
        self.steps += 1
        self.trajectory.append(next_pos.copy())

        if self.steps >= self.max_steps:
            done = True
            truncated = True

        info = {
            "distance_to_target": np.linalg.norm(next_pos - self.target),
            "steps": self.steps,
            "collision": terminated and not truncated and reward < 0
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        """Render environment."""
        if self.render_mode is None:
            return

        if self.fig is None:
            self.fig = plt.figure(figsize=(8, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.ion()

        self.ax.clear()

        for obs in self.obstacles:
            self.ax.scatter(*obs, c='red', s=200, marker='o', alpha=0.6)

        self.ax.scatter(*self.target, c='green', s=300, marker='*', label='Target')
        self.ax.scatter(*self.state[:3], c='blue', s=200, marker='^', label='Drone')

        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b--', linewidth=1.5, alpha=0.7)

        self.ax.set_xlim(0, self.grid_limit)
        self.ax.set_ylim(0, self.grid_limit)
        self.ax.set_zlim(0, self.grid_limit)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'Drone Navigation (Step {self.steps}/{self.max_steps})')
        self.ax.legend()

        plt.draw()
        plt.pause(0.01)

    def close(self):
        """Close rendering."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None