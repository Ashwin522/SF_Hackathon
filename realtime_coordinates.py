#!/usr/bin/env python3
"""
Real-time Basketball Game Simulation with Live Player Coordinates
"""

import numpy as np
from basketball_env import BasketballEnv
import time


def generate_player_positions(ball_pos, possession, frame, action):
    """Generate realistic player positions based on game state"""
    np.random.seed(frame)
    
    team1_players = []
    team2_players = []
    
    if possession == 0:  # Team 1 attacking
        # Team 1 (attacking players moving toward x=100)
        team1_players.append({'id': 1, 'x': ball_pos, 'y': 25, 'role': 'Ball Handler'})
        team1_players.append({'id': 2, 'x': min(ball_pos + 12, 95), 'y': 15, 'role': 'Forward'})
        team1_players.append({'id': 3, 'x': min(ball_pos + 15, 98), 'y': 35, 'role': 'Wing'})
        team1_players.append({'id': 4, 'x': max(ball_pos - 15, 5), 'y': 10, 'role': 'Guard'})
        team1_players.append({'id': 5, 'x': max(ball_pos - 18, 0), 'y': 40, 'role': 'Back Guard'})
        
        # Team 2 (defending players)
        team2_players.append({'id': 1, 'x': min(ball_pos + 8, 98), 'y': 25, 'role': 'Defender on Ball'})
        team2_players.append({'id': 2, 'x': min(ball_pos + 12, 95), 'y': 15, 'role': 'Wing Defender'})
        team2_players.append({'id': 3, 'x': min(ball_pos + 15, 98), 'y': 35, 'role': 'Wing Defender'})
        team2_players.append({'id': 4, 'x': max(ball_pos - 8, 10), 'y': 10, 'role': 'Perimeter Def'})
        team2_players.append({'id': 5, 'x': max(ball_pos - 12, 5), 'y': 40, 'role': 'Perimeter Def'})
    
    else:  # Team 2 attacking
        # Team 2 (attacking players moving toward x=0)
        team2_players.append({'id': 1, 'x': ball_pos, 'y': 25, 'role': 'Ball Handler'})
        team2_players.append({'id': 2, 'x': max(ball_pos - 12, 5), 'y': 15, 'role': 'Forward'})
        team2_players.append({'id': 3, 'x': max(ball_pos - 15, 2), 'y': 35, 'role': 'Wing'})
        team2_players.append({'id': 4, 'x': min(ball_pos + 15, 95), 'y': 10, 'role': 'Guard'})
        team2_players.append({'id': 5, 'x': min(ball_pos + 18, 100), 'y': 40, 'role': 'Back Guard'})
        
        # Team 1 (defending players)
        team1_players.append({'id': 1, 'x': max(ball_pos - 8, 2), 'y': 25, 'role': 'Defender on Ball'})
        team1_players.append({'id': 2, 'x': max(ball_pos - 12, 5), 'y': 15, 'role': 'Wing Defender'})
        team1_players.append({'id': 3, 'x': max(ball_pos - 15, 2), 'y': 35, 'role': 'Wing Defender'})
        team1_players.append({'id': 4, 'x': min(ball_pos + 8, 90), 'y': 10, 'role': 'Perimeter Def'})
        team1_players.append({'id': 5, 'x': min(ball_pos + 12, 95), 'y': 40, 'role': 'Perimeter Def'})
    
    return team1_players, team2_players


def run_realtime_simulation(num_games=1):
    """Run simulation and display player coordinates in real-time"""
    env = BasketballEnv()
    
    action_names = ["Pass", "Dribble Fwd", "Dribble Back", "Shoot"]
    
    for game_num in range(num_games):
        print(f"\n{'='*120}")
        print(f"GAME {game_num + 1} - REAL-TIME PLAYER COORDINATES")
        print(f"{'='*120}\n")
        
        observation, info = env.reset()
        frame = 0
        
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            team1_score = int(observation[0])
            team2_score = int(observation[1])
            possession = int(observation[2])
            time_remaining = int(observation[3])
            ball_pos = float(observation[4])
            
            # Get player positions
            team1_players, team2_players = generate_player_positions(ball_pos, possession, frame, action)
            
            # Clear screen (works on macOS/Linux)
            print("\033[2J\033[H", end="")
            
            # Header
            minutes = time_remaining // 60
            seconds = time_remaining % 60
            
            print(f"\n{'='*120}")
            print(f"GAME {game_num + 1} | FRAME {frame} | TIME: {minutes}:{seconds:02d} | SCORE: Team 1 {team1_score} - {team2_score} Team 2")
            print(f"{'='*120}")
            print(f"Possession: {'TEAM 1' if possession == 0 else 'TEAM 2':<20} | Action: {action_names[action]:<15} | Ball Position: {ball_pos:.2f}")
            print(f"{'='*120}\n")
            
            # Team 1 Coordinates
            print(f"{'TEAM 1 - PLAYERS':<60} {'TEAM 2 - PLAYERS':<60}")
            print(f"{'-'*60} {'-'*60}")
            print(f"{'ID':<5} {'X':<8} {'Y':<8} {'Role':<42} {'ID':<5} {'X':<8} {'Y':<8} {'Role':<42}")
            print(f"{'-'*60} {'-'*60}")
            
            for i in range(5):
                p1 = team1_players[i]
                p2 = team2_players[i]
                
                print(f"{p1['id']:<5} {p1['x']:<8.2f} {p1['y']:<8.2f} {p1['role']:<42} {p2['id']:<5} {p2['x']:<8.2f} {p2['y']:<8.2f} {p2['role']:<42}")
            
            print(f"\n{'-'*120}")
            
            # Court visualization (simple ASCII)
            print("\nCOURT VIEW (Top-Down):")
            print("X: 0 (Team 1 Basket) ←→ 100 (Team 2 Basket)")
            print("Y: 0 (Bottom) ←→ 50 (Top)\n")
            
            # Create a simple visualization
            court = [['.' for _ in range(100)] for _ in range(50)]
            
            # Mark baskets
            for y in range(20, 31):
                court[y][0] = '1'  # Team 1 basket
                court[y][99] = '2'  # Team 2 basket
            
            # Mark ball
            ball_y = int(25)
            ball_x = int(ball_pos)
            if 0 <= ball_x < 100 and 0 <= ball_y < 50:
                court[ball_y][ball_x] = 'O'
            
            # Mark Team 1 players
            for p in team1_players:
                x, y = int(p['x']), int(p['y'])
                if 0 <= x < 100 and 0 <= y < 50:
                    court[y][x] = 'A'
            
            # Mark Team 2 players
            for p in team2_players:
                x, y = int(p['x']), int(p['y'])
                if 0 <= x < 100 and 0 <= y < 50:
                    court[y][x] = 'B'
            
            # Print court
            for y in range(50):
                print(''.join(court[y]))
            
            print("\nLegend: 1=T1 Basket | 2=T2 Basket | O=Ball | A=Team1 | B=Team2")
            print(f"{'-'*120}\n")
            
            frame += 1
            
            if terminated:
                break
            
            # Pause for readability (optional)
            time.sleep(0.1)
        
        # Game summary
        print(f"\n{'='*120}")
        print(f"GAME {game_num + 1} FINAL SCORE")
        print(f"Team 1: {int(observation[0])} | Team 2: {int(observation[1])}")
        if observation[0] > observation[1]:
            print("TEAM 1 WINS!")
        elif observation[1] > observation[0]:
            print("TEAM 2 WINS!")
        else:
            print("TIE GAME!")
        print(f"Total Frames: {frame}")
        print(f"{'='*120}\n")
    
    env.close()


if __name__ == "__main__":
    print("\n" + "="*120)
    print("BASKETBALL GAME - REAL-TIME PLAYER COORDINATES")
    print("="*120)
    print("\nShowing live player positions as the game progresses...")
    print("Coordinates are updated in real-time\n")
    
    run_realtime_simulation(num_games=1)
