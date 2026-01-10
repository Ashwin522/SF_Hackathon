#!/usr/bin/env python3
"""
Basketball Game Visualization with Real-Time Defensive Coordinates
Shows court animation plus live defensive player positions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from basketball_env import BasketballEnv


def draw_basketball_court(ax):
    """Draw a basketball court"""
    court = patches.Rectangle((0, 0), 100, 50, 
                             linewidth=2, edgecolor='white', facecolor='#d4a574')
    ax.add_patch(court)
    
    ax.plot([50, 50], [0, 50], 'w--', linewidth=1)
    circle = patches.Circle((50, 25), 6, linewidth=1, edgecolor='white', facecolor='none')
    ax.add_patch(circle)
    
    ax.plot([0, 0], [20, 30], 'w-', linewidth=3)
    ax.plot([100, 100], [20, 30], 'w-', linewidth=3)
    
    basket1 = patches.Circle((0, 25), 0.8, linewidth=1.5, edgecolor='yellow', facecolor='none')
    basket2 = patches.Circle((100, 25), 0.8, linewidth=1.5, edgecolor='yellow', facecolor='none')
    ax.add_patch(basket1)
    ax.add_patch(basket2)
    
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 55)
    ax.set_aspect('equal')
    ax.set_facecolor('#0a0a0a')
    ax.axis('off')


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


def run_visual_simulation(num_games=2):
    """Run basketball simulation with real-time defensive coordinates display"""
    env = BasketballEnv()
    
    for game_num in range(num_games):
        print(f"\n{'='*60}")
        print(f"VISUALIZING GAME {game_num + 1}")
        print(f"{'='*60}\n")
        
        observation, info = env.reset()
        
        game_history = {
            'team1_scores': [0],
            'team2_scores': [0],
            'ball_positions': [50],
            'possessions': [0],
            'times': [2400],
            'actions': [],
            'team1_players': [],
            'team2_players': []
        }
        
        fig, (court_ax, stats_ax) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Collect game data
        step_count = 0
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            team1_score = int(observation[0])
            team2_score = int(observation[1])
            possession = int(observation[2])
            time_remaining = int(observation[3])
            ball_pos = float(observation[4])
            
            game_history['team1_scores'].append(team1_score)
            game_history['team2_scores'].append(team2_score)
            game_history['ball_positions'].append(ball_pos)
            game_history['possessions'].append(possession)
            game_history['times'].append(time_remaining)
            
            action_names = ["Pass", "Dribble Fwd", "Dribble Back", "Shoot"]
            game_history['actions'].append(action_names[action])
            
            team1_players, team2_players = generate_player_positions(ball_pos, possession, step_count)
            game_history['team1_players'].append(team1_players)
            game_history['team2_players'].append(team2_players)
            
            if terminated:
                break
        
        num_frames = len(game_history['team1_scores'])
        
        def update_frame(frame):
            court_ax.clear()
            stats_ax.clear()
            
            draw_basketball_court(court_ax)
            
            team1_players = game_history['team1_players'][frame]
            team2_players = game_history['team2_players'][frame]
            ball_pos = game_history['ball_positions'][frame]
            possession = game_history['possessions'][frame]
            
            # Draw Team 1 players
            for player in team1_players:
                circle = patches.Circle((player['x'], player['y']), 1.5, linewidth=2, 
                                       edgecolor='cyan', facecolor='darkblue', alpha=0.7)
                court_ax.add_patch(circle)
                court_ax.text(player['x'], player['y'], str(player['id']), fontsize=8, 
                            ha='center', va='center', color='white', weight='bold')
            
            # Draw Team 2 players
            for player in team2_players:
                circle = patches.Circle((player['x'], player['y']), 1.5, linewidth=2, 
                                       edgecolor='red', facecolor='darkred', alpha=0.7)
                court_ax.add_patch(circle)
                court_ax.text(player['x'], player['y'], str(player['id']), fontsize=8, 
                            ha='center', va='center', color='white', weight='bold')
            
            # Draw ball
            ball = patches.Circle((ball_pos, 25), 1.2, linewidth=2, 
                                 edgecolor='gold', facecolor='darkorange', alpha=0.9)
            court_ax.add_patch(ball)
            
            # Possession indicator
            if possession == 0:
                court_ax.text(10, 48, "Team 1 ●", fontsize=14, color='cyan', weight='bold')
                court_ax.text(75, 48, "Team 2", fontsize=14, color='white', weight='bold', alpha=0.5)
                defending_team = team2_players
                defending_color = 'red'
                defending_name = 'Team 2'
            else:
                court_ax.text(10, 48, "Team 1", fontsize=14, color='white', weight='bold', alpha=0.5)
                court_ax.text(75, 48, "Team 2 ●", fontsize=14, color='red', weight='bold')
                defending_team = team1_players
                defending_color = 'cyan'
                defending_name = 'Team 1'
            
            # Legends
            court_ax.text(2, 2, "1-5: Team 1 (Cyan)", fontsize=9, color='cyan', 
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            court_ax.text(65, 2, "1-5: Team 2 (Red)", fontsize=9, color='red',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            
            # Stats panel
            time_remaining = game_history['times'][frame]
            minutes = time_remaining // 60
            seconds = time_remaining % 60
            
            stats_ax.text(0.5, 0.98, "GAME STATS & DEFENSIVE POSITIONS", fontsize=13, weight='bold', 
                         color='white', ha='center', transform=stats_ax.transAxes)
            
            stats_ax.text(0.5, 0.93, f"Time: {minutes}:{seconds:02d}", fontsize=11, 
                         color='white', ha='center', transform=stats_ax.transAxes)
            
            stats_ax.text(0.5, 0.88, f"Team 1: {game_history['team1_scores'][frame]} | Team 2: {game_history['team2_scores'][frame]}", 
                         fontsize=12, weight='bold', color='white', ha='center', 
                         transform=stats_ax.transAxes)
            
            stats_ax.text(0.5, 0.82, f"Action: {game_history['actions'][frame]}", 
                         fontsize=10, color='yellow', ha='center', 
                         transform=stats_ax.transAxes)
            
            stats_ax.text(0.5, 0.77, f"Ball Position: {ball_pos:.1f}", 
                         fontsize=10, color='orange', ha='center', 
                         transform=stats_ax.transAxes)
            
            # Defensive coordinates
            stats_ax.text(0.05, 0.68, f"{defending_name} DEFENDING POSITIONS:", fontsize=10, 
                         weight='bold', color=defending_color, transform=stats_ax.transAxes)
            
            y_pos = 0.63
            for player in defending_team:
                if player['status'] == 'defending':
                    stats_ax.text(0.05, y_pos, 
                                f"P{player['id']}: X={player['x']:.1f} | Y={player['y']:.1f}", 
                                fontsize=9, color=defending_color, ha='left', 
                                transform=stats_ax.transAxes, family='monospace',
                                bbox=dict(boxstyle='round', facecolor='black', alpha=0.3))
                    y_pos -= 0.055
            
            # Score trend
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
            stats_ax.set_xlabel('Actions', fontsize=8, color='white')
            stats_ax.set_ylabel('Score', fontsize=8, color='white')
            stats_ax.legend(loc='lower right', fontsize=9, facecolor='#1a1a1a', edgecolor='white')
            stats_ax.grid(True, alpha=0.2, color='white')
            stats_ax.tick_params(colors='white', labelsize=8)
            stats_ax.set_facecolor('#222222')
            
            fig.suptitle(f'Basketball Game {game_num + 1} - Frame {frame}/{num_frames-1}', 
                        fontsize=14, weight='bold', color='white')
        
        anim = FuncAnimation(fig, update_frame, frames=num_frames, 
                           interval=500, repeat=True, repeat_delay=2000)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Final Score: Team 1: {game_history['team1_scores'][-1]} | Team 2: {game_history['team2_scores'][-1]}")
        if game_history['team1_scores'][-1] > game_history['team2_scores'][-1]:
            print("Team 1 WINS!")
        elif game_history['team2_scores'][-1] > game_history['team1_scores'][-1]:
            print("Team 2 WINS!")
        else:
            print("TIE GAME!")
        print(f"Total Frames: {step_count}\n")
    
    env.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BASKETBALL GAME SIMULATION")
    print("WITH REAL-TIME DEFENSIVE COORDINATES")
    print("="*60)
    print("\nLeft: Court with players (1-5 numbered)")
    print("Right: Live stats + Defensive player coordinates\n")
    
    run_visual_simulation(num_games=2)
