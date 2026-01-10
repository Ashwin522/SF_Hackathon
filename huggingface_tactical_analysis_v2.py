#!/usr/bin/env python3
"""
Basketball Simulation with LLM Tactical Analysis using Hugging Face
Captures defensive coordinates and gets counter-positioning recommendations
Falls back to local analysis if API is unavailable
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
        team2_players.append({'id': 2, 'x': min(ball_pos + 15, 95), 'y': 15 + np.random.uniform(-3, 3), 'status': 'attacking'})
        team2_players.append({'id': 3, 'x': min(ball_pos + 10, 90), 'y': 35 + np.random.uniform(-3, 3), 'status': 'attacking'})
        team2_players.append({'id': 4, 'x': max(ball_pos - 15, 5), 'y': 10 + np.random.uniform(-3, 3), 'status': 'attacking'})
        team2_players.append({'id': 5, 'x': max(ball_pos - 20, 0), 'y': 40 + np.random.uniform(-3, 3), 'status': 'attacking'})
        
        team1_players.append({'id': 1, 'x': ball_pos + np.random.uniform(-8, 8), 'y': 25 + np.random.uniform(-5, 5), 'status': 'defending'})
        team1_players.append({'id': 2, 'x': min(ball_pos + 15, 95), 'y': 15 + np.random.uniform(-5, 5), 'status': 'defending'})
        team1_players.append({'id': 3, 'x': min(ball_pos + 10, 90), 'y': 35 + np.random.uniform(-5, 5), 'status': 'defending'})
        team1_players.append({'id': 4, 'x': max(ball_pos - 15, 5), 'y': 10 + np.random.uniform(-5, 5), 'status': 'defending'})
        team1_players.append({'id': 5, 'x': max(ball_pos - 20, 0), 'y': 40 + np.random.uniform(-5, 5), 'status': 'defending'})
    
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


def generate_local_tactical_analysis(attacking_team, defending_team, ball_pos, possession):
    """Generate tactical analysis locally without LLM"""
    possession_team = "Team 1" if possession == 0 else "Team 2"
    defending_team_name = "Team 2" if possession == 0 else "Team 1"
    
    # Find defensive gaps (Y-axis spacing between defenders)
    defender_y_positions = sorted([p['y'] for p in defending_team])
    gaps = []
    for i in range(len(defender_y_positions) - 1):
        gap_size = defender_y_positions[i+1] - defender_y_positions[i]
        gap_center = (defender_y_positions[i] + defender_y_positions[i+1]) / 2
        if gap_size > 8:
            gaps.append((gap_center, gap_size))
    
    # Find isolated defenders
    isolation_targets = []
    for defender in defending_team:
        nearby_count = sum(1 for d in defending_team 
                         if abs(d['x'] - defender['x']) < 15 and abs(d['y'] - defender['y']) < 8)
        if nearby_count <= 1:
            isolation_targets.append(defender)
    
    analysis = f"""
=== LOCAL TACTICAL ANALYSIS FOR {possession_team} ===

DEFENSIVE VULNERABILITIES DETECTED:
"""
    
    if gaps:
        analysis += f"\n1. Vertical Spacing Gaps: {len(gaps)} gaps detected\n"
        for i, (gap_y, gap_size) in enumerate(gaps):
            analysis += f"   Gap {i+1}: Y-coordinate {gap_y:.1f}, Size: {gap_size:.1f}\n"
            analysis += f"   -> RECOMMENDED DRIVE: Send player to X=50-60, Y={gap_y:.1f}\n"
    
    if isolation_targets:
        analysis += f"\n2. Isolated Defenders: {len(isolation_targets)} players out of position\n"
        for target in isolation_targets:
            analysis += f"   Player at X={target['x']:.1f}, Y={target['y']:.1f} (isolated)\n"
            analysis += f"   -> RECOMMENDED ATTACK: Pick and roll toward X={target['x']:.1f}\n"
    
    # Calculate defensive density
    left_half = sum(1 for p in defending_team if p['x'] < 50)
    right_half = len(defending_team) - left_half
    
    if left_half < right_half:
        analysis += f"\n3. Defensive Imbalance: More defenders on right side ({right_half}) vs left ({left_half})\n"
        analysis += f"   -> RECOMMENDED PLAY: Run offense to LEFT side (X: 20-40, Y: varied)\n"
    else:
        analysis += f"\n3. Defensive Imbalance: More defenders on left side ({left_half}) vs right ({right_half})\n"
        analysis += f"   -> RECOMMENDED PLAY: Run offense to RIGHT side (X: 60-80, Y: varied)\n"
    
    # Suggest counter-positioning
    analysis += f"\n4. COUNTER-POSITIONING FOR {possession_team}:\n"
    player_recommendations = [
        f"   Player 1 (Ball Handler): Move to X=45, Y=25",
        f"   Player 2 (Wing): Cut to X=70, Y=20",
        f"   Player 3 (Forward): Post at X=75, Y=35",
        f"   Player 4 (Guard): Screen at X=50, Y=35",
        f"   Player 5 (Center): Back screen at X=40, Y=15"
    ]
    for rec in player_recommendations:
        analysis += f"\n   {rec}\n"
    
    return analysis


def query_huggingface(formatted_positions, team_analyzing, defending_team, attacking_team, ball_pos, possession):
    """Query Hugging Face LLM for tactical recommendations, fallback to local analysis"""
    api_key = os.getenv('HUGGINGFACE_API_KEY')
    if not api_key:
        print("  Note: HUGGINGFACE_API_KEY not set. Using local tactical analysis...\n")
        return generate_local_tactical_analysis(attacking_team, defending_team, ball_pos, possession)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    user_prompt = f"""You are a professional basketball tactical analyst. Analyze this game state:

{formatted_positions}

Provide specific counter-positioning recommendations with:
1. Key defensive weaknesses (gaps, isolated players)
2. Recommended attack vector (specific X,Y coordinates)
3. Player positioning (e.g., "Player 2 to X=70, Y=20")
4. Expected outcome if executed

Be concise and actionable."""

    data = {
        "inputs": user_prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7
        }
    }
    
    try:
        print("  Calling Hugging Face API (Mistral model)...")
        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', 'No analysis generated')
        return str(result)
    
    except requests.exceptions.RequestException as e:
        print(f"  API Error: {e}")
        print("  Falling back to local tactical analysis...\n")
        return generate_local_tactical_analysis(attacking_team, defending_team, ball_pos, possession)


def run_tactical_analysis():
    """Run basketball simulation and collect tactical analysis"""
    print("=" * 60)
    print("BASKETBALL TACTICAL ANALYSIS - WITH HUGGING FACE LLM")
    print("=" * 60)
    print()
    
    env = BasketballEnv()
    obs, _ = env.reset()
    
    print("Running simulation and collecting game states...")
    game_states = []
    frame_count = 0
    
    while frame_count < 100:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        game_states.append({
            'frame': frame_count,
            'obs': obs,
            'info': info
        })
        
        frame_count += 1
        if done or truncated:
            break
    
    print(f"Simulation complete! Captured {len(game_states)} frames.\n")
    
    # Analyze every 25 frames (max 4 analyses)
    analysis_frames = [0, 25, 50, 75]
    analysis_frames = [f for f in analysis_frames if f < len(game_states)]
    
    for frame_idx in analysis_frames:
        state = game_states[frame_idx]
        obs = state['obs']
        
        score1, score2, possession, time_remaining, ball_pos = obs
        time_remaining = int(time_remaining)
        
        attacking_team, defending_team = generate_player_positions(
            ball_pos, possession, state['frame']
        )
        
        formatted_positions = format_game_state_for_llm(
            attacking_team, defending_team, ball_pos, 
            time_remaining, int(score1), int(score2), possession
        )
        
        print("=" * 60)
        print(f"FRAME {frame_idx} ANALYSIS")
        print("=" * 60)
        print(formatted_positions)
        print("\nSending to LLM for tactical analysis...")
        
        possession_team = "Team 1" if possession == 0 else "Team 2"
        analysis = query_huggingface(
            formatted_positions, 
            possession_team,
            defending_team,
            attacking_team,
            ball_pos,
            possession
        )
        
        print("LLM TACTICAL ANALYSIS:")
        print("-" * 60)
        print(analysis)
        print("=" * 60)
        print()
    
    print("Tactical analysis complete!")


if __name__ == "__main__":
    run_tactical_analysis()
