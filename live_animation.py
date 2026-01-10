#!/usr/bin/env python3
"""
Real-time Basketball Game Visualization with Live Player Coordinates
Shows animated court with player positions updating in real-time
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from basketball_env import BasketballEnv
import matplotlib.patheffects as path_effects


def generate_player_positions(ball_pos, possession, frame):
    """Generate realistic player positions based on game state"""
    np.random.seed(frame)
    
    team1_players = []
    team2_players = []
    
    if possession == 0:  # Team 1 attacking
        team1_players.append({'id': 1, 'x': ball_pos, 'y': 25, 'role': 'Ball Handler'})
        team1_players.append({'id': 2, 'x': min(ball_pos + 12, 95), 'y': 15, 'role': 'Forward'})
        team1_players.append({'id': 3, 'x': min(ball_pos + 15, 98), 'y': 35, 'role': 'Wing'})
        team1_players.append({'id': 4, 'x': max(ball_pos - 15, 5), 'y': 10, 'role': 'Guard'})
        team1_players.append({'id': 5, 'x': max(ball_pos - 18, 0), 'y': 40, 'role': 'Back Guard'})
        
        team2_players.append({'id': 1, 'x': min(ball_pos + 8, 98), 'y': 25, 'role': 'Defender'})
        team2_players.append({'id': 2, 'x': min(ball_pos + 12, 95), 'y': 15, 'role': 'Defender'})
        team2_players.append({'id': 3, 'x': min(ball_pos + 15, 98), 'y': 35, 'role': 'Defender'})
        team2_players.append({'id': 4, 'x': max(ball_pos - 8, 10), 'y': 10, 'role': 'Defender'})
        team2_players.append({'id': 5, 'x': max(ball_pos - 12, 5), 'y': 40, 'role': 'Defender'})
    else:  # Team 2 attacking
        team2_players.append({'id': 1, 'x': ball_pos, 'y': 25, 'role': 'Ball Handler'})
        team2_players.append({'id': 2, 'x': max(ball_pos - 12, 5), 'y': 15, 'role': 'Forward'})
        team2_players.append({'id': 3, 'x': max(ball_pos - 15, 2), 'y': 35, 'role': 'Wing'})
        team2_players.append({'id': 4, 'x': min(ball_pos + 15, 95), 'y': 10, 'role': 'Guard'})
        team2_players.append({'id': 5, 'x': min(ball_pos + 18, 100), 'y': 40, 'role': 'Back Guard'})
        
        team1_players.append({'id': 1, 'x': max(ball_pos - 8, 2), 'y': 25, 'role': 'Defender'})
        team1_players.append({'id': 2, 'x': max(ball_pos - 12, 5), 'y': 15, 'role': 'Defender'})
        team1_players.append({'id': 3, 'x': max(ball_pos - 15, 2), 'y': 35, 'role': 'Defender'})
        team1_players.append({'id': 4, 'x': min(ball_pos + 8, 90), 'y': 10, 'role': 'Defender'})
        team1_players.append({'id': 5, 'x': min(ball_pos + 12, 95), 'y': 40, 'role': 'Defender'})
    
    return team1_players, team2_players


def draw_court(ax):
    """Draw basketball court"""
    # Court rectangle
    court = patches.Rectangle((0, 0), 100, 50, linewidth=2, edgecolor='white', facecolor='#d4a574')
    ax.add_patch(court)
    
    # Center line
    ax.plot([50, 50], [0, 50], 'w--', linewidth=1, alpha=0.5)
    
    # Center circle
    circle = patches.Circle((50, 25), 6, linewidth=1, edgecolor='white', facecolor='none', alpha=0.5)
    ax.add_patch(circle)
    
    # Baskets
    ax.plot([0, 0], [20, 30], 'w-', linewidth=3)
    ax.plot([100, 100], [20, 30], 'w-', linewidth=3)
    
    # Basket circles
    basket1 = patches.Circle((0, 25), 0.8, linewidth=1.5, edgecolor='yellow', facecolor='none')
    basket2 = patches.Circle((100, 25), 0.8, linewidth=1.5, edgecolor='yellow', facecolor='none')
    ax.add_patch(basket1)
    ax.add_patch(basket2)
    
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 55)
    ax.set_aspect('equal')
    ax.set_facecolor('#0a0a0a')
    ax.axis('off')


def run_animated_simulation():
    """Run simulation with live animation"""
    env = BasketballEnv()
    
    # Collect game data first
    observation, info = env.reset()
    game_data = {
        'team1_players': [],
        'team2_players': [],
        'ball_pos': [],
        'possession': [],
        'score1': [],
        'score2': [],
        'time': [],
        'action': []
    }
    
    action_names = ["Pass", "Dribble Fwd", "Dribble Back", "Shoot"]
    frame = 0
    
    print("Simulating game...")
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        team1_score = int(observation[0])
        team2_score = int(observation[1])
        possession = int(observation[2])
        time_remaining = int(observation[3])
        ball_pos = float(observation[4])
        
        team1_players, team2_players = generate_player_positions(ball_pos, possession, frame)
        
        game_data['team1_players'].append(team1_players)
        game_data['team2_players'].append(team2_players)
        game_data['ball_pos'].append(ball_pos)
        game_data['possession'].append(possession)
        game_data['score1'].append(team1_score)
        game_data['score2'].append(team2_score)
        game_data['time'].append(time_remaining)
        game_data['action'].append(action_names[action])
        
        frame += 1
        if terminated or frame >= 100:  # Limit to 100 frames for demo
            break
    
    print(f"Game simulated. Total frames: {frame}")
    
    # Create animation
    fig, (ax_court, ax_coords) = plt.subplots(1, 2, figsize=(18, 8))
    
    def animate(f):
        ax_court.clear()
        ax_coords.clear()
        
        draw_court(ax_court)
        
        # Get current game state
        team1_players = game_data['team1_players'][f]
        team2_players = game_data['team2_players'][f]
        ball_pos = game_data['ball_pos'][f]
        possession = game_data['possession'][f]
        score1 = game_data['score1'][f]
        score2 = game_data['score2'][f]
        time_remaining = game_data['time'][f]
        action = game_data['action'][f]
        
        # Draw Team 1 players (Cyan)
        for player in team1_players:
            circle = patches.Circle((player['x'], player['y']), 2, linewidth=2, 
                                   edgecolor='cyan', facecolor='darkblue', alpha=0.8)
            ax_court.add_patch(circle)
            text = ax_court.text(player['x'], player['y'], str(player['id']), 
                               fontsize=10, ha='center', va='center', color='white', weight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), 
                                 path_effects.Normal()])
        
        # Draw Team 2 players (Red)
        for player in team2_players:
            circle = patches.Circle((player['x'], player['y']), 2, linewidth=2, 
                                   edgecolor='red', facecolor='darkred', alpha=0.8)
            ax_court.add_patch(circle)
            text = ax_court.text(player['x'], player['y'], str(player['id']), 
                               fontsize=10, ha='center', va='center', color='white', weight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), 
                                 path_effects.Normal()])
        
        # Draw ball
        ball = patches.Circle((ball_pos, 25), 1.5, linewidth=2, edgecolor='gold', facecolor='orange', alpha=0.9)
        ax_court.add_patch(ball)
        
        # Possession indicator
        if possession == 0:
            ax_court.text(10, 48, "Team 1", fontsize=14, color='cyan', weight='bold')
        else:
            ax_court.text(85, 48, "Team 2", fontsize=14, color='red', weight='bold')
        
        # Score display
        ax_court.text(50, 48, f"{score1} - {score2}", fontsize=16, ha='center', 
                     color='white', weight='bold',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Coordinates panel
        ax_coords.axis('off')
        ax_coords.set_facecolor('#0a0a0a')
        
        minutes = time_remaining // 60
        seconds = time_remaining % 60
        
        # Title
        ax_coords.text(0.5, 0.95, f"Frame {f} | Time: {minutes}:{seconds:02d}", 
                      fontsize=14, weight='bold', color='white', ha='center', 
                      transform=ax_coords.transAxes)
        
        ax_coords.text(0.5, 0.90, f"Score: Team 1 {score1} - {score2} Team 2", 
                      fontsize=12, color='white', ha='center', transform=ax_coords.transAxes)
        
        ax_coords.text(0.5, 0.85, f"Action: {action}", 
                      fontsize=11, color='yellow', ha='center', transform=ax_coords.transAxes)
        
        ax_coords.text(0.5, 0.80, f"Ball Position: {ball_pos:.2f}", 
                      fontsize=11, color='orange', ha='center', transform=ax_coords.transAxes)
        
        # Team 1 coordinates
        y_pos = 0.72
        ax_coords.text(0.05, y_pos, "TEAM 1 (Cyan)", fontsize=11, weight='bold', 
                      color='cyan', transform=ax_coords.transAxes)
        y_pos -= 0.05
        for player in team1_players:
            ax_coords.text(0.05, y_pos, f"P{player['id']}: ({player['x']:.1f}, {player['y']:.1f})", 
                          fontsize=9, color='cyan', transform=ax_coords.transAxes, family='monospace')
            y_pos -= 0.04
        
        # Team 2 coordinates
        y_pos -= 0.02
        ax_coords.text(0.05, y_pos, "TEAM 2 (Red)", fontsize=11, weight='bold', 
                      color='red', transform=ax_coords.transAxes)
        y_pos -= 0.05
        for player in team2_players:
            ax_coords.text(0.05, y_pos, f"P{player['id']}: ({player['x']:.1f}, {player['y']:.1f})", 
                          fontsize=9, color='red', transform=ax_coords.transAxes, family='monospace')
            y_pos -= 0.04
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(game_data['ball_pos']), 
                        interval=100, repeat=True, repeat_delay=2000)
    
    plt.tight_layout()
    plt.show()
    
    env.close()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("BASKETBALL GAME - LIVE PLAYER COORDINATES ANIMATION")
    print("="*80)
    print("\nLeft: Basketball Court with players")
    print("Right: Live player coordinates\n")
    
    run_animated_simulation()
