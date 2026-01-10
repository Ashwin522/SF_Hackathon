import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BasketballEnv(gym.Env):
    """
    A custom basketball game environment for OpenAI Gym.
    
    The environment simulates a basketball game between two teams.
    Each team has 5 players and the goal is to score more points than the opponent.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        
        # Game state: [team1_score, team2_score, possession, time_remaining, ball_position]
        # possession: 0 = team1, 1 = team2
        # ball_position: 0-100 (0 = team1 basket, 100 = team2 basket, 50 = center)
        # time_remaining: 0-2400 (in seconds, 40 minutes game)
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([200, 200, 1, 2400, 100], dtype=np.float32),
            dtype=np.float32
        )
        
        # Actions: 0=pass, 1=dribble forward, 2=dribble backward, 3=shoot
        self.action_space = spaces.Discrete(4)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.team1_score = 0
        self.team2_score = 0
        self.possession = 0  # 0 for team1, 1 for team2
        self.time_remaining = 2400  # 40 minutes
        self.ball_position = 50  # Center court
        self.steps = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        return np.array([
            self.team1_score,
            self.team2_score,
            self.possession,
            self.time_remaining,
            self.ball_position
        ], dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        self.time_remaining = max(0, self.time_remaining - 5)  # Each action takes 5 seconds
        
        reward = 0
        shot_made = False
        turnover = False
        
        if self.possession == 0:  # Team 1 has possession
            if action == 0:  # Pass
                if np.random.random() < 0.1:  # 10% chance of turnover
                    self.possession = 1
                    turnover = True
                else:
                    self.ball_position = min(100, self.ball_position + np.random.uniform(5, 15))
            
            elif action == 1:  # Dribble forward
                self.ball_position = min(100, self.ball_position + np.random.uniform(8, 12))
            
            elif action == 2:  # Dribble backward
                self.ball_position = max(0, self.ball_position - np.random.uniform(5, 10))
            
            elif action == 3:  # Shoot
                if self.ball_position > 75:  # Close enough to basket
                    if np.random.random() < 0.4:  # 40% shot accuracy
                        self.team1_score += 2
                        reward = 10
                        shot_made = True
                    else:
                        reward = -5
                else:
                    reward = -5  # Penalty for bad shot
                self.possession = 1  # Turnover after shot
        
        else:  # Team 2 has possession
            if action == 0:  # Pass
                if np.random.random() < 0.1:
                    self.possession = 0
                    turnover = True
                else:
                    self.ball_position = max(0, self.ball_position - np.random.uniform(5, 15))
            
            elif action == 1:  # Dribble forward
                self.ball_position = max(0, self.ball_position - np.random.uniform(8, 12))
            
            elif action == 2:  # Dribble backward
                self.ball_position = min(100, self.ball_position + np.random.uniform(5, 10))
            
            elif action == 3:  # Shoot
                if self.ball_position < 25:  # Close enough to basket
                    if np.random.random() < 0.4:  # 40% shot accuracy
                        self.team2_score += 2
                        reward = 10
                        shot_made = True
                    else:
                        reward = -5
                else:
                    reward = -5
                self.possession = 0  # Turnover after shot
        
        # Penalize doing nothing (no meaningful action)
        if reward == 0 and not turnover and not shot_made:
            reward = -0.1
        
        terminated = self.time_remaining <= 0
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def render(self):
        if self.render_mode == "human":
            print(f"\n{'='*50}")
            print(f"Team 1: {self.team1_score} | Team 2: {self.team2_score}")
            print(f"Time Remaining: {self.time_remaining // 60}:{self.time_remaining % 60:02d}")
            print(f"Possession: {'Team 1' if self.possession == 0 else 'Team 2'}")
            print(f"Ball Position: {self.ball_position:.1f} (0=Team1 Basket, 100=Team2 Basket)")
            print(f"{'='*50}")
