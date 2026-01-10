#!/usr/bin/env python3
"""
Basketball Simulation with LLM Tactical Analysis using Hugging Face
Captures defensive coordinates and gets counter-positioning recommendations
"""

import numpy as np
import json
import os
from basketball_env import BasketballEnv
import requests


def generate_player_positions(ball_pos, possession, frame):
    """Generate player positions with defensive status"""
    np.random.seed(frame)
    
    team1_players = []
    team2_players = []
    
    if possession == 0:  # Team 1 attacking, Team 2 defending
        team1_players.append({'id': 1, 'x': ball_pos, 'y': 25, 'status': 'attacking'})
        team1_players.append({'id': 2, 'x': min(ball_pos + 15, 95), 'y': 15 + np.random.uniform(-3, 3), 'status': 'attacking'})
        team1_players.append({'id': 3, 'x': min(ball_pos + 10, 90), 'y': 35 + np.random.uniform(-3, 3), 'status': 'attacking'})
        team1_players.append({'id': 4, 'x': max(ball_pos - 15, 5), 'y': 10 + np.random.uniform(-3, 3), 'status': 'attacking'})
        team1_players.append({'id': 5, 'x': max(ball_pos - 20, 0), 'y': 40 + np.random.uniform(-3, 3), 'status': 'attacking'})
        
        team2_players.append({'id': 1, 'x': ball_pos + np.random.uniform(-8, 8), 'y': 25 + np.random.uniform(-5, 5), 'status': 'defending'})
        team2_players.append({'id': 2, 'x': min(ball_pos + 15, 95), 'y': 15 + np.random.uniform(-5, 5), 'status': 'defending'})
        team2_players.append({'id': 3, 'x': min(ball_pos + 10, 90), 'y': 35 + np.random.uniform(-5, 5), 'status': 'defending'})
        team2_players.append({'id': 4, 'x': max(ball_pos - 15, 5), 'y': 10 + np.random.uniform(-5, 5), 'status': 'defending'})
        team2_players.append({'id': 5, 'x': max(ball_pos - 20, 0), 'y': 40 + np.random.uniform(-5, 5), 'status': 'defending'})
    else:  # Team 2 attacking, Team 1 defending
        team2_players.append({'id': 1, 'x': ball_pos, 'y': 25, 'status': 'attacking'})
        team2_players.append({'id': 2, 'x': max(ball_pos - 15, 5), 'y': 15 + np.random.uniform(-3, 3), 'status': 'attacking'})
        team2_players.append({'id': 3, 'x': max(ball_pos - 10, 10), 'y': 35 + np.random.uniform(-3, 3), 'status': 'attacking'})
        team2_players.append({'id': 4, 'x': min(ball_pos + 15, 95), 'y': 10 + np.random.uniform(-3, 3), 'status': 'attacking'})
        team2_players.append({'id': 5, 'x': min(ball_pos + 20, 100), 'y': 40 + np.random.uniform(-3, 3), 'status': 'attacking'})
        
        team1_players.append({'id': 1, 'x': ball_pos + np.random.uniform(-8, 8), 'y': 25 + np.random.uniform(-5, 5), 'status': 'defending'})
        team1_players.append({'id': 2, 'x': max(ball_pos - 15, 5), 'y': 15 + np.random.uniform(-5, 5), 'status': 'defending'})
        team1_players.append({'id': 3, 'x': max(ball_pos - 10, 10), 'y': 35 + np.random.uniform(-5, 5), 'status': 'defending'})
        team1_players.append({'id': 4, 'x': min(ball_pos + 15, 95), 'y': 10 + np.random.uniform(-5, 5), 'status': 'defending'})
        team1_players.append({'id': 5, 'x': min(ball_pos + 20, 100), 'y': 40 + np.random.uniform(-5, 5), 'status': 'defending'})
    
    return team1_players, team2_players


def format_game_state_for_llm(attacking_team, defending_team, ball_pos, time_remaining, score1, score2, possession):
    """Format game state and positions for LLM analysis"""
    
    possession_team = "Team 1" if possession == 0 else "Team 2"
    defending_team_name = "Team 2" if possession == 0 else "Team 1"
    
    # Format attacking positions
    attacking_info = f"\n{possession_team} ATTACKING POSITIONS (moving toward opponent basket):\n"
    for player in attacking_team:
        attacking_info += f"  Player {player['id']}: X={player['x']:.1f}, Y={player['y']:.1f}\n"
    
    # Format defending positions
    defending_info = f"\n{defending_team_name} DEFENDING POSITIONS (current formation):\n"
    for player in defending_team:
        defending_info += f"  Player {player['id']}: X={player['x']:.1f}, Y={player['y']:.1f}\n"
    
    # Court info
    court_info = f"""
COURT DIMENSIONS & SETUP:
- Court Length: 0 to 100 (X-axis)
- Court Width: 0 to 50 (Y-axis)
- {possession_team} Attacking Toward: X=100
- {defending_team_name} Defending at: X=0-25
- Ball Position: X={ball_pos:.1f}, Y=25 (center)
- Time Remaining: {time_remaining // 60}:{time_remaining % 60:02d}
- Score: Team 1: {score1} | Team 2: {score2}
"""
    
    return attacking_info + defending_info + court_info


def call_huggingface_api(api_key, messages):
    """Call Hugging Face Inference API"""
    
    api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": messages[-1]["content"],
        "parameters": {
            "max_new_tokens": 1000,
            "temperature": 0.7
        }
    }
    
    try:
        print("  Calling Hugging Face API...")
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            if 'generated_text' in result[0]:
                return result[0]['generated_text']
        
        return str(result)
    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"


def run_tactical_analysis():
    """Run simulation and get LLM tactical analysis on selected frames"""
    
    # Get API key
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        print("ERROR: HUGGINGFACE_API_KEY environment variable not set")
        print("Please set your API key:")
        print("export HUGGINGFACE_API_KEY='your-token-here'")
        return
    
    env = BasketballEnv()
    
    print("="*80)
    print("BASKETBALL TACTICAL ANALYSIS - LLM WITH HUGGING FACE")
    print("="*80)
    print("\nRunning simulation and collecting game states...\n")
    
    observation, info = env.reset()
    
    game_snapshots = []
    frame = 0
    
    # Collect game data
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        frame += 1
        
        team1_score = int(observation[0])
        team2_score = int(observation[1])
        possession = int(observation[2])
        time_remaining = int(observation[3])
        ball_pos = float(observation[4])
        
        team1_players, team2_players = generate_player_positions(ball_pos, possession, frame)
        
        snapshot = {
            'frame': frame,
            'ball_pos': ball_pos,
            'possession': possession,
            'time_remaining': time_remaining,
            'score1': team1_score,
            'score2': team2_score,
            'team1_players': team1_players,
            'team2_players': team2_players
        }
        
        game_snapshots.append(snapshot)
        
        if terminated or frame >= 50:
            break
    
    print(f"Simulation complete! Captured {len(game_snapshots)} frames.\n")
    
    # Analyze selected frames with LLM
    selected_frames = [0, len(game_snapshots)//2]
    
    for frame_idx in selected_frames:
        if frame_idx >= len(game_snapshots):
            continue
            
        snapshot = game_snapshots[frame_idx]
        
        if snapshot['possession'] == 0:
            attacking_team = snapshot['team1_players']
            defending_team = snapshot['team2_players']
        else:
            attacking_team = snapshot['team2_players']
            defending_team = snapshot['team1_players']
        
        game_state = format_game_state_for_llm(
            attacking_team, defending_team, snapshot['ball_pos'],
            snapshot['time_remaining'], snapshot['score1'], snapshot['score2'],
            snapshot['possession']
        )
        
        print("="*80)
        print(f"FRAME {snapshot['frame']} ANALYSIS")
        print("="*80)
        print(game_state)
        
        # Send to LLM
        print("\nSending to Hugging Face for tactical analysis...\n")
        
        user_message = f"""Analyze this basketball game state and provide counter-positioning recommendations:

{game_state}

Please provide:
1. Current defensive formation assessment
2. Weaknesses in the defending team's positioning
3. Specific counter-position recommendations for each defender (with exact coordinates)
4. Expected outcome if this positioning continues"""
        
        messages = [{"role": "user", "content": user_message}]
        
        response_text = call_huggingface_api(api_key, messages)
        
        print("LLM TACTICAL ANALYSIS:")
        print("-" * 80)
        print(response_text)
        print("\n" + "="*80 + "\n")
    
    env.close()
    print("\nTactical analysis complete!")


if __name__ == "__main__":
    run_tactical_analysis()
