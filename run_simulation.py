#!/usr/bin/env python3
"""
Basketball Game Simulation using OpenAI Gym
"""

import numpy as np
from basketball_env import BasketballEnv


def run_simulation(num_episodes=5, render=True):
    """
    Run basketball game simulations
    
    Args:
        num_episodes: Number of games to simulate
        render: Whether to display game state
    """
    
    env = BasketballEnv(render_mode="human" if render else None)
    
    for episode in range(num_episodes):
        print(f"\n\n{'*'*60}")
        print(f"GAME {episode + 1}")
        print(f"{'*'*60}")
        
        observation, info = env.reset()
        total_reward = 0
        
        while True:
            if render:
                env.render()
            
            # Random action for now (you can implement a strategy here)
            action = env.action_space.sample()
            
            # Convert action to readable format
            action_names = ["Pass", "Dribble Forward", "Dribble Backward", "Shoot"]
            print(f"Action: {action_names[action]}")
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated:
                break
        
        # Game summary
        print(f"\n{'='*50}")
        print(f"GAME {episode + 1} FINAL SCORE")
        print(f"Team 1: {int(observation[0])} | Team 2: {int(observation[1])}")
        if observation[0] > observation[1]:
            print("Team 1 WINS!")
        elif observation[1] > observation[0]:
            print("Team 2 WINS!")
        else:
            print("TIE GAME!")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"{'='*50}")
    
    env.close()


def run_trained_agent_simulation(num_episodes=3):
    """
    Example of running simulation with a simple rule-based strategy
    """
    
    env = BasketballEnv(render_mode="human")
    
    print("\n\n" + "="*60)
    print("RUNNING WITH SIMPLE STRATEGY")
    print("="*60)
    
    for episode in range(num_episodes):
        print(f"\n\n{'*'*60}")
        print(f"GAME {episode + 1} - STRATEGY-BASED")
        print(f"{'*'*60}")
        
        observation, info = env.reset()
        
        while True:
            env.render()
            
            # Simple strategy: move towards opponent's basket, then shoot when close
            team1_has_ball = observation[2] == 0
            ball_pos = observation[4]
            
            if team1_has_ball:
                if ball_pos > 75:  # Close to basket, shoot
                    action = 3  # Shoot
                elif ball_pos < 90:  # Move forward
                    action = 1  # Dribble forward
                else:
                    action = 0  # Pass
            else:
                if ball_pos < 25:  # Close to basket, shoot
                    action = 3  # Shoot
                elif ball_pos > 10:  # Move forward
                    action = 1  # Dribble forward
                else:
                    action = 0  # Pass
            
            action_names = ["Pass", "Dribble Forward", "Dribble Backward", "Shoot"]
            print(f"Action: {action_names[action]}")
            
            observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                break
        
        # Game summary
        print(f"\n{'='*50}")
        print(f"GAME {episode + 1} FINAL SCORE")
        print(f"Team 1: {int(observation[0])} | Team 2: {int(observation[1])}")
        if observation[0] > observation[1]:
            print("Team 1 WINS!")
        elif observation[1] > observation[0]:
            print("Team 2 WINS!")
        else:
            print("TIE GAME!")
        print(f"{'='*50}")
    
    env.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BASKETBALL GAME SIMULATION - RANDOM ACTIONS")
    print("="*60)
    
    # Run random simulations
    run_simulation(num_episodes=3, render=True)
    
    # Run with simple strategy
    run_trained_agent_simulation(num_episodes=2)
    
    print("\n\nSimulation complete!")
