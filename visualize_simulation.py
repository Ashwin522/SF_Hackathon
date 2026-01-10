#!/usr/bin/env python3
"""
Basketball Game Visualization using Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from basketball_env import BasketballEnv


def draw_basketball_court(ax):
    """Draw a basketball court"""
    # Court dimensions
    court_length = 100
    court_width = 50
    
    # Draw court boundaries
    court = patches.Rectangle((0, 0), court_length, court_width, 
                             linewidth=2, edgecolor='white', facecolor='#d4a574')
    ax.add_patch(court)
    
    # Draw center line
    ax.plot([50, 50], [0, court_width], 'w--', linewidth=1)
    
    # Draw center circle
    circle = patches.Circle((50, court_width/2), 6, linewidth=1, 
                           edgecolor='white', facecolor='none')
    ax.add_patch(circle)
    
    # Draw baskets (3-point line and paint)
    # Team 1 basket (left side)
    ax.plot([0, 0], [court_width/2 - 8, court_width/2 + 8], 'w-', linewidth=3)  # Baseline
    basket1 = patches.Circle((5, court_width/2), 0.9, linewidth=1, 
                            edgecolor='yellow', facecolor='none')
    ax.add_patch(basket1)
    
    # Team 2 basket (right side)
    ax.plot([100, 100], [court_width/2 - 8, court_width/2 + 8], 'w-', linewidth=3)  # Baseline
    basket2 = patches.Circle((95, court_width/2), 0.9, linewidth=1, 
                            edgecolor='yellow', facecolor='none')
    ax.add_patch(basket2)
    
    # Three-point lines
    three_point1 = patches.Arc((0, court_width/2), 16, 16, angle=0, 
                              theta1=270, theta2=90, linewidth=1, color='white')
    ax.add_patch(three_point1)
    
    three_point2 = patches.Arc((100, court_width/2), 16, 16, angle=0, 
                              theta1=90, theta2=270, linewidth=1, color='white')
    ax.add_patch(three_point2)
    
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 55)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')


def generate_player_positions(ball_pos, possession, action, frame):
    """Generate player positions based on game state with defense info"""
    np.random.seed(frame)  # Consistent random positions per frame
    
    team1_players = []
    team2_players = []
    
    if possession == 0:  # Team 1 has ball, Team 2 defending
        # Team 1 players (attacking towards right)
        team1_players.append({'id': 1, 'x': ball_pos, 'y': 25, 'status': 'attacking'})
        team1_players.append({'id': 2, 'x': min(ball_pos + 15, 95), 'y': 15 + np.random.uniform(-3, 3), 'status': 'attacking'})
        team1_players.append({'id': 3, 'x': min(ball_pos + 10, 90), 'y': 35 + np.random.uniform(-3, 3), 'status': 'attacking'})
        team1_players.append({'id': 4, 'x': max(ball_pos - 15, 5), 'y': 10 + np.random.uniform(-3, 3), 'status': 'attacking'})
        team1_players.append({'id': 5, 'x': max(ball_pos - 20, 0), 'y': 40 + np.random.uniform(-3, 3), 'status': 'attacking'})
        
        # Team 2 players (defending)
        team2_players.append({'id': 1, 'x': ball_pos + np.random.uniform(-8, 8), 'y': 25 + np.random.uniform(-5, 5), 'status': 'defending'})
        team2_players.append({'id': 2, 'x': min(ball_pos + 15, 95), 'y': 15 + np.random.uniform(-5, 5), 'status': 'defending'})
        team2_players.append({'id': 3, 'x': min(ball_pos + 10, 90), 'y': 35 + np.random.uniform(-5, 5), 'status': 'defending'})
        team2_players.append({'id': 4, 'x': max(ball_pos - 15, 5), 'y': 10 + np.random.uniform(-5, 5), 'status': 'defending'})
        team2_players.append({'id': 5, 'x': max(ball_pos - 20, 0), 'y': 40 + np.random.uniform(-5, 5), 'status': 'defending'})
    else:  # Team 2 has ball, Team 1 defending
        # Team 2 players (attacking towards left)
        team2_players.append({'id': 1, 'x': ball_pos, 'y': 25, 'status': 'attacking'})
        team2_players.append({'id': 2, 'x': max(ball_pos - 15, 5), 'y': 15 + np.random.uniform(-3, 3), 'status': 'attacking'})
        team2_players.append({'id': 3, 'x': max(ball_pos - 10, 10), 'y': 35 + np.random.uniform(-3, 3), 'status': 'attacking'})
        team2_players.append({'id': 4, 'x': min(ball_pos + 15, 95), 'y': 10 + np.random.uniform(-3, 3), 'status': 'attacking'})
        team2_players.append({'id': 5, 'x': min(ball_pos + 20, 100), 'y': 40 + np.random.uniform(-3, 3), 'status': 'attacking'})
        
        # Team 1 players (defending)
        team1_players.append({'id': 1, 'x': ball_pos + np.random.uniform(-8, 8), 'y': 25 + np.random.uniform(-5, 5), 'status': 'defending'})
        team1_players.append({'id': 2, 'x': max(ball_pos - 15, 5), 'y': 15 + np.random.uniform(-5, 5), 'status': 'defending'})
        team1_players.append({'id': 3, 'x': max(ball_pos - 10, 10), 'y': 35 + np.random.uniform(-5, 5), 'status': 'defending'})
        team1_players.append({'id': 4, 'x': min(ball_pos + 15, 95), 'y': 10 + np.random.uniform(-5, 5), 'status': 'defending'})
        team1_players.append({'id': 5, 'x': min(ball_pos + 20, 100), 'y': 40 + np.random.uniform(-5, 5), 'status': 'defending'})
    
    return team1_players, team2_players


def run_visual_simulation(num_games=2):
    """Run simulation with visualization"""
    env = BasketballEnv()
    
    for game_num in range(num_games):
        print(f"\n{'='*60}")
        print(f"VISUALIZING GAME {game_num + 1}")
        print(f"{'='*60}")
        
        observation, info = env.reset()
        
        game_history = {
            'team1_scores': [0],
            'team2_scores': [0],
            'ball_positions': [50],
            'possessions': [0],
            'times': [2400],
            'actions': []
        }
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Court
        court_ax = axes[0]
        draw_basketball_court(court_ax)
        
        # Right plot: Stats
        stats_ax = axes[1]
        stats_ax.set_facecolor('#1a1a1a')
        
        # Run game and collect history
        step_count = 0
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            game_history['team1_scores'].append(int(observation[0]))
            game_history['team2_scores'].append(int(observation[1]))
            game_history['ball_positions'].append(float(observation[4]))
            game_history['possessions'].append(int(observation[2]))
            game_history['times'].append(int(observation[3]))
            
            action_names = ["Pass", "Dribble Fwd", "Dribble Back", "Shoot"]
            game_history['actions'].append(action_names[action])
            
            if terminated:
                break
        
        # Create animation frames
        num_frames = len(game_history['team1_scores'])
        
        def update_frame(frame):
            court_ax.clear()
            stats_ax.clear()
            
            draw_basketball_court(court_ax)
            
            # Get player positions
            ball_pos = game_history['ball_positions'][frame]
            possession = game_history['possessions'][frame]
            action = env.action_space.sample()
            team1_players, team2_players = generate_player_positions(ball_pos, possession, action, frame)
            
            # Draw Team 1 players (cyan)
            for i, (x, y) in enumerate(team1_players):
                player = patches.Circle((x, y), 1.5, linewidth=2, 
                                       edgecolor='cyan', facecolor='darkblue', alpha=0.7)
                court_ax.add_patch(player)
                court_ax.text(x, y, str(i+1), fontsize=8, ha='center', va='center', 
                            color='white', weight='bold')
            
            # Draw Team 2 players (red)
            for i, (x, y) in enumerate(team2_players):
                player = patches.Circle((x, y), 1.5, linewidth=2, 
                                       edgecolor='red', facecolor='darkred', alpha=0.7)
                court_ax.add_patch(player)
                court_ax.text(x, y, str(i+1), fontsize=8, ha='center', va='center', 
                            color='white', weight='bold')
            
            # Draw ball (larger, more visible)
            ball = patches.Circle((ball_pos, 25), 1.2, linewidth=2, 
                                 edgecolor='gold', facecolor='darkorange', alpha=0.9)
            court_ax.add_patch(ball)
            court_ax.text(ball_pos, 25, '●', fontsize=20, ha='center', va='center', 
                         color='yellow', alpha=0.8)
            
            # Draw possession indicator
            if possession == 0:
                court_ax.text(10, 48, "Team 1 ●", fontsize=14, color='cyan', weight='bold')
                court_ax.text(75, 48, "Team 2", fontsize=14, color='white', weight='bold', alpha=0.5)
            else:
                court_ax.text(10, 48, "Team 1", fontsize=14, color='white', weight='bold', alpha=0.5)
                court_ax.text(75, 48, "Team 2 ●", fontsize=14, color='red', weight='bold')
            
            # Draw legends
            court_ax.text(2, 2, "1-5: Team 1 (Cyan)", fontsize=9, color='cyan', 
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            court_ax.text(65, 2, "1-5: Team 2 (Red)", fontsize=9, color='red',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            
            # Stats panel
            time_remaining = game_history['times'][frame]
            minutes = time_remaining // 60
            seconds = time_remaining % 60
            
            stats_ax.text(0.5, 0.92, "GAME STATS", fontsize=18, weight='bold', 
                         color='white', ha='center', transform=stats_ax.transAxes)
            
            stats_ax.text(0.5, 0.83, f"Time: {minutes}:{seconds:02d}", fontsize=14, 
                         color='white', ha='center', transform=stats_ax.transAxes)
            
            stats_ax.text(0.5, 0.73, f"Team 1: {game_history['team1_scores'][frame]}", 
                         fontsize=16, weight='bold', color='cyan', ha='center', 
                         transform=stats_ax.transAxes)
            
            stats_ax.text(0.5, 0.63, f"Team 2: {game_history['team2_scores'][frame]}", 
                         fontsize=16, weight='bold', color='red', ha='center', 
                         transform=stats_ax.transAxes)
            
            stats_ax.text(0.5, 0.50, f"Action: {game_history['actions'][frame]}", 
                         fontsize=12, color='yellow', ha='center', 
                         transform=stats_ax.transAxes)
            
            stats_ax.text(0.5, 0.40, f"Ball: {game_history['ball_positions'][frame]:.1f}", 
                         fontsize=11, color='orange', ha='center', 
                         transform=stats_ax.transAxes)
            
            # Score trend
            trend_len = min(20, frame+1)
            trend_data1 = game_history['team1_scores'][max(0, frame-19):frame+1]
            trend_data2 = game_history['team2_scores'][max(0, frame-19):frame+1]
            
            stats_ax.plot(range(len(trend_data1)), trend_data1, 
                         'c-', linewidth=2, label='Team 1', marker='o', markersize=4)
            stats_ax.plot(range(len(trend_data2)), trend_data2, 
                         'r-', linewidth=2, label='Team 2', marker='s', markersize=4)
            
            max_score = max(max(game_history['team1_scores'][:frame+1]), 
                           max(game_history['team2_scores'][:frame+1])) + 5
            stats_ax.set_xlim(-1, 21)
            stats_ax.set_ylim(0, max_score)
            stats_ax.set_xlabel('Actions Ago', fontsize=10, color='white')
            stats_ax.set_ylabel('Score', fontsize=10, color='white')
            stats_ax.legend(loc='upper left', fontsize=10, facecolor='#1a1a1a', edgecolor='white')
            stats_ax.grid(True, alpha=0.2, color='white')
            stats_ax.tick_params(colors='white')
            stats_ax.set_facecolor('#222222')
            
            # Frame counter
            fig.suptitle(f'Basketball Game {game_num + 1} - Action {frame}/{num_frames-1} | Players Visible', 
                        fontsize=14, weight='bold', color='white')
        
        # Create animation
        anim = FuncAnimation(fig, update_frame, frames=num_frames, 
                           interval=200, repeat=True, repeat_delay=2000)
        
        plt.tight_layout()
        plt.show()
        
        # Print final stats
        print(f"\nFinal Score:")
        print(f"Team 1: {game_history['team1_scores'][-1]} | Team 2: {game_history['team2_scores'][-1]}")
        if game_history['team1_scores'][-1] > game_history['team2_scores'][-1]:
            print("Team 1 WINS!")
        elif game_history['team2_scores'][-1] > game_history['team1_scores'][-1]:
            print("Team 2 WINS!")
        else:
            print("TIE GAME!")
        print(f"Total Actions: {step_count}")
    
    env.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BASKETBALL GAME VISUAL SIMULATION")
    print("="*60)
    print("\nWatch the game unfold on the court!")
    print("Left: Basketball Court with ball position")
    print("Right: Live stats and score trend\n")
    
    run_visual_simulation(num_games=2)
