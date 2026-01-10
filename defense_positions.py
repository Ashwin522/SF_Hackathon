#!/usr/bin/env python3
"""
Basketball Game Simulation - Defensive Player Positions Tracker
"""

import numpy as np
import json
from basketball_env import BasketballEnv


def generate_defensive_positions(ball_pos, defending_team, frame):
    """
    Generate defensive player positions for a team
    
    defending_team: 0 = Team 1 defending, 1 = Team 2 defending
    """
    np.random.seed(frame)  # Consistent positions per frame
    
    defenders = []
    
    if defending_team == 0:  # Team 1 defending (their basket at x=0, defending against Team 2)
        # Defensive formation: zone defense around their basket
        # Center defender (guards the basket)
        defenders.append({
            'player_id': 1,
            'x': 10,
            'y': 25,
            'role': 'Center/Paint'
        })
        
        # Two wing defenders
        defenders.append({
            'player_id': 2,
            'x': 20,
            'y': 12,
            'role': 'Wing Left'
        })
        
        defenders.append({
            'player_id': 3,
            'x': 20,
            'y': 38,
            'role': 'Wing Right'
        })
        
        # Two perimeter defenders (guard the three-point line)
        defenders.append({
            'player_id': 4,
            'x': 30,
            'y': 10,
            'role': 'Perimeter Left'
        })
        
        defenders.append({
            'player_id': 5,
            'x': 30,
            'y': 40,
            'role': 'Perimeter Right'
        })
        
        # Adjust positions based on ball position (defenders follow the ball)
        if ball_pos > 50:  # Ball on opponent's side, shift defensive formation forward
            for defender in defenders:
                defender['x'] = min(45, defender['x'] + 10)
        elif ball_pos < 50:  # Ball on Team 1's side, compress defense
            for defender in defenders:
                defender['x'] = max(5, defender['x'] - 5)
    
    else:  # Team 2 defending (their basket at x=100, defending against Team 1)
        # Defensive formation: zone defense around their basket
        # Center defender
        defenders.append({
            'player_id': 1,
            'x': 90,
            'y': 25,
            'role': 'Center/Paint'
        })
        
        # Two wing defenders
        defenders.append({
            'player_id': 2,
            'x': 80,
            'y': 12,
            'role': 'Wing Left'
        })
        
        defenders.append({
            'player_id': 3,
            'x': 80,
            'y': 38,
            'role': 'Wing Right'
        })
        
        # Two perimeter defenders
        defenders.append({
            'player_id': 4,
            'x': 70,
            'y': 10,
            'role': 'Perimeter Left'
        })
        
        defenders.append({
            'player_id': 5,
            'x': 70,
            'y': 40,
            'role': 'Perimeter Right'
        })
        
        # Adjust positions based on ball position
        if ball_pos < 50:  # Ball on opponent's side, shift defensive formation forward
            for defender in defenders:
                defender['x'] = max(55, defender['x'] - 10)
        elif ball_pos > 50:  # Ball on Team 2's side, compress defense
            for defender in defenders:
                defender['x'] = min(95, defender['x'] + 5)
    
    return defenders


def run_defense_position_simulation(num_games=2):
    """
    Run basketball simulation and track defensive positions
    """
    env = BasketballEnv()
    all_game_data = []
    
    for game_num in range(num_games):
        print(f"\n{'='*70}")
        print(f"GAME {game_num + 1} - TRACKING DEFENSIVE POSITIONS")
        print(f"{'='*70}")
        
        observation, info = env.reset()
        
        game_defense_data = {
            'game_number': game_num + 1,
            'snapshots': []
        }
        
        frame = 0
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            team1_score = int(observation[0])
            team2_score = int(observation[1])
            possession = int(observation[2])
            time_remaining = int(observation[3])
            ball_pos = float(observation[4])
            
            # Get defensive positions
            if possession == 0:  # Team 1 attacking, Team 2 defending
                defending_team = 1
                defending_team_name = "Team 2"
            else:  # Team 2 attacking, Team 1 defending
                defending_team = 0
                defending_team_name = "Team 1"
            
            defenders = generate_defensive_positions(ball_pos, defending_team, frame)
            
            # Create snapshot
            snapshot = {
                'frame': frame,
                'time_remaining': time_remaining,
                'possession': 'Team 1' if possession == 0 else 'Team 2',
                'defending_team': defending_team_name,
                'ball_position': round(ball_pos, 2),
                'team1_score': team1_score,
                'team2_score': team2_score,
                'defensive_formation': defenders
            }
            
            game_defense_data['snapshots'].append(snapshot)
            
            frame += 1
            if terminated:
                break
        
        all_game_data.append(game_defense_data)
        
        print(f"\nGame {game_num + 1} Summary:")
        print(f"Final Score: Team 1: {observation[0]:.0f} | Team 2: {observation[1]:.0f}")
        print(f"Total frames: {frame}")
        print(f"Defensive snapshots captured: {len(game_defense_data['snapshots'])}")
    
    env.close()
    return all_game_data


def print_defensive_positions(all_game_data):
    """
    Print defensive positions in a readable format
    """
    for game in all_game_data:
        print(f"\n\n{'#'*70}")
        print(f"GAME {game['game_number']} - DEFENSIVE POSITIONS")
        print(f"{'#'*70}")
        
        # Sample every 10 frames to reduce output
        for snapshot in game['snapshots'][::10]:
            print(f"\n{'-'*70}")
            print(f"Frame: {snapshot['frame']} | Time: {snapshot['time_remaining']//60}:{snapshot['time_remaining']%60:02d}")
            print(f"Possession: {snapshot['possession']} | Defending: {snapshot['defending_team']}")
            print(f"Ball Position: {snapshot['ball_position']} | Score: {snapshot['team1_score']}-{snapshot['team2_score']}")
            print(f"{'-'*70}")
            
            print(f"\n{snapshot['defending_team']} DEFENSIVE FORMATION:")
            print(f"{'Player':<10} {'Position X':<12} {'Position Y':<12} {'Role':<20}")
            print(f"{'-'*60}")
            
            for defender in snapshot['defensive_formation']:
                print(f"{defender['player_id']:<10} {defender['x']:<12} {defender['y']:<12} {defender['role']:<20}")


def export_defensive_data_json(all_game_data, filename='defense_data.json'):
    """
    Export defensive positions to JSON file
    """
    with open(filename, 'w') as f:
        json.dump(all_game_data, f, indent=2)
    print(f"\nDefensive data exported to {filename}")


def export_defensive_data_csv(all_game_data, filename='defense_data.csv'):
    """
    Export defensive positions to CSV file
    """
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Game', 'Frame', 'Time_Remaining', 'Possession', 'Defending_Team',
            'Ball_Position', 'Team1_Score', 'Team2_Score',
            'Player_ID', 'Player_X', 'Player_Y', 'Player_Role'
        ])
        
        for game in all_game_data:
            for snapshot in game['snapshots']:
                for defender in snapshot['defensive_formation']:
                    writer.writerow([
                        game['game_number'],
                        snapshot['frame'],
                        snapshot['time_remaining'],
                        snapshot['possession'],
                        snapshot['defending_team'],
                        snapshot['ball_position'],
                        snapshot['team1_score'],
                        snapshot['team2_score'],
                        defender['player_id'],
                        defender['x'],
                        defender['y'],
                        defender['role']
                    ])
    print(f"Defensive data exported to {filename}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("BASKETBALL GAME - DEFENSIVE POSITIONS TRACKER")
    print("="*70)
    
    # Run simulation and collect defensive positions
    game_data = run_defense_position_simulation(num_games=2)
    
    # Print defensive positions to console
    print_defensive_positions(game_data)
    
    # Export to files
    export_defensive_data_json(game_data)
    export_defensive_data_csv(game_data)
    
    print("\n" + "="*70)
    print("Simulation complete! Data exported to defense_data.json and defense_data.csv")
    print("="*70)
