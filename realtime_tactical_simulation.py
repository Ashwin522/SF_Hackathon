#!/usr/bin/env python3
"""
Real-time Basketball Simulation with Live Tactical Analysis
Runs visualization and LLM tactical analysis in parallel on the same frames
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from basketball_env import BasketballEnv
import requests
from threading import Thread
import queue
import time
from opensource_llm_integration import OpenSourceLLM


def generate_player_positions(ball_pos, possession, frame):
    """Generate player positions with defensive status - truly dynamic each frame"""
    # Remove seeding - allow true randomness each frame for defensive formations
    
    team1_players = []
    team2_players = []
    
    # Ball moves in both X and Y directions
    ball_y = 25 + np.random.uniform(-10, 10)  # Ball can move vertically too
    
    # Add more natural variation to defensive positioning
    def_variation_y = 8  # Larger Y variation for defenders
    def_variation_x = 5  # X variation as defenders respond to ball
    
    if possession == 0:  # Team 1 attacking, Team 2 defending
        team1_players.append({'id': 1, 'x': ball_pos, 'y': ball_y, 'status': 'attacking'})
        team1_players.append({'id': 2, 'x': min(ball_pos + 15, 95), 'y': ball_y + np.random.uniform(-10, 10), 'status': 'attacking'})
        team1_players.append({'id': 3, 'x': min(ball_pos + 10, 90), 'y': ball_y + np.random.uniform(-15, 15), 'status': 'attacking'})
        team1_players.append({'id': 4, 'x': max(ball_pos - 15, 5), 'y': ball_y + np.random.uniform(-15, 15), 'status': 'attacking'})
        team1_players.append({'id': 5, 'x': max(ball_pos - 20, 0), 'y': ball_y + np.random.uniform(-10, 10), 'status': 'attacking'})
        
        # Defensive players respond to ball position - truly random each frame
        team2_players.append({'id': 1, 'x': ball_pos + np.random.uniform(-12, 12), 'y': ball_y + np.random.uniform(-def_variation_y, def_variation_y), 'status': 'defending'})
        team2_players.append({'id': 2, 'x': min(ball_pos + 20, 98), 'y': 10 + np.random.uniform(-def_variation_y, def_variation_y), 'status': 'defending'})
        team2_players.append({'id': 3, 'x': min(ball_pos + 18, 96), 'y': 40 + np.random.uniform(-def_variation_y, def_variation_y), 'status': 'defending'})
        team2_players.append({'id': 4, 'x': max(ball_pos - 20, 2), 'y': 15 + np.random.uniform(-def_variation_y, def_variation_y), 'status': 'defending'})
        team2_players.append({'id': 5, 'x': max(ball_pos - 25, 0), 'y': 35 + np.random.uniform(-def_variation_y, def_variation_y), 'status': 'defending'})
    else:  # Team 2 attacking, Team 1 defending
        team2_players.append({'id': 1, 'x': ball_pos, 'y': ball_y, 'status': 'attacking'})
        team2_players.append({'id': 2, 'x': min(ball_pos + 15, 95), 'y': ball_y + np.random.uniform(-10, 10), 'status': 'attacking'})
        team2_players.append({'id': 3, 'x': min(ball_pos + 10, 90), 'y': ball_y + np.random.uniform(-15, 15), 'status': 'attacking'})
        team2_players.append({'id': 4, 'x': max(ball_pos - 15, 5), 'y': ball_y + np.random.uniform(-15, 15), 'status': 'attacking'})
        team2_players.append({'id': 5, 'x': max(ball_pos - 20, 0), 'y': ball_y + np.random.uniform(-10, 10), 'status': 'attacking'})
        
        # Defensive players respond to ball position
        team1_players.append({'id': 1, 'x': ball_pos + np.random.uniform(-12, 12), 'y': ball_y + np.random.uniform(-def_variation_y, def_variation_y), 'status': 'defending'})
        team1_players.append({'id': 2, 'x': min(ball_pos + 20, 98), 'y': 10 + np.random.uniform(-def_variation_y, def_variation_y), 'status': 'defending'})
        team1_players.append({'id': 3, 'x': min(ball_pos + 18, 96), 'y': 40 + np.random.uniform(-def_variation_y, def_variation_y), 'status': 'defending'})
        team1_players.append({'id': 4, 'x': max(ball_pos - 20, 2), 'y': 15 + np.random.uniform(-def_variation_y, def_variation_y), 'status': 'defending'})
        team1_players.append({'id': 5, 'x': max(ball_pos - 25, 0), 'y': 35 + np.random.uniform(-def_variation_y, def_variation_y), 'status': 'defending'})
    
    return team1_players, team2_players


def draw_basketball_court(ax):
    """Draw a basketball court"""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.set_aspect('equal')
    
    # Court outline
    ax.plot([0, 100, 100, 0, 0], [0, 0, 50, 50, 0], 'k-', lw=2)
    
    # Center line
    ax.plot([50, 50], [0, 50], 'k--', lw=1)
    
    # Center circle
    circle = plt.Circle((50, 25), 6, color='black', fill=False, lw=1)
    ax.add_patch(circle)
    
    # Baskets
    ax.plot([5], [25], 'co', markersize=8, label='Team 1 Basket (Cyan)')
    ax.plot([95], [25], 'o', color='red', markersize=8, label='Team 2 Basket (Red)')
    
    ax.set_xlabel('Court Length (X)', fontsize=10)
    ax.set_ylabel('Court Width (Y)', fontsize=10)
    ax.set_title('Basketball Court - Real-Time Tactical Analysis', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)


def generate_local_tactical_analysis(attacking_team, defending_team, ball_pos, possession):
    """Generate tactical analysis with detailed reasoning - OR use Groq LLM"""
    
    # Try to use Groq LLM first
    try:
        llm = OpenSourceLLM(provider="groq")
        if llm.api_key:
            # Convert team lists to position dicts
            attacking_positions = {p['id']: (p['x'], p['y']) for p in attacking_team}
            defending_positions = {p['id']: (p['x'], p['y']) for p in defending_team}
            
            llm_response = llm.generate_tactical_analysis(
                attacking_positions, defending_positions, ball_pos, possession
            )
            
            # Format LLM response
            possession_team = "Team 1" if possession == 0 else "Team 2"
            defending_team_name = "Team 2" if possession == 0 else "Team 1"
            
            analysis = f"[{possession_team} ATTACKING vs {defending_team_name} DEFENSE]\n"
            analysis += f"Ball Position: X={ball_pos:.1f}\n\n"
            analysis += "🤖 AI TACTICAL ANALYSIS (Groq Llama-3.1):\n"
            analysis += llm_response
            return analysis
    except Exception as e:
        # Silently fall back to local analysis
        pass
    
    # Local analysis as fallback
    possession_team = "Team 1" if possession == 0 else "Team 2"
    defending_team_name = "Team 2" if possession == 0 else "Team 1"
    
    # Analyze defensive gaps
    defender_y_positions = sorted([p['y'] for p in defending_team])
    gaps = []
    gap_analysis = ""
    
    for i in range(len(defender_y_positions) - 1):
        gap_size = defender_y_positions[i+1] - defender_y_positions[i]
        gap_center = (defender_y_positions[i] + defender_y_positions[i+1]) / 2
        if gap_size > 8:
            gaps.append((gap_center, gap_size))
            gap_analysis += f"  * Gap between Y={defender_y_positions[i]:.1f} and Y={defender_y_positions[i+1]:.1f} (size {gap_size:.1f})\n"
    
    # Calculate isolation (defenders away from teammates)
    isolation_count = 0
    isolated_y = []
    isolated_x = []
    for defender in defending_team:
        nearby = sum(1 for d in defending_team if abs(d['y'] - defender['y']) < 10)
        if nearby <= 1:
            isolation_count += 1
            isolated_y.append(defender['y'])
            isolated_x.append(defender['x'])
    
    # Analyze left/right imbalance
    left_half = sum(1 for p in defending_team if p['x'] < 50)
    right_half = len(defending_team) - left_half
    weaker_side = "LEFT" if left_half < right_half else "RIGHT"
    weaker_count = min(left_half, right_half)
    stronger_count = max(left_half, right_half)
    
    # Analyze forward pressure
    defenders_forward = sum(1 for p in defending_team if p['x'] > ball_pos)
    defenders_back = len(defending_team) - defenders_forward
    
    analysis = f"\n[{possession_team} ATTACKING vs {defending_team_name} DEFENSE]\n"
    
    # Get ball Y position from attacking team's ball handler
    ball_y = attacking_team[0]['y'] if attacking_team else 25
    analysis += f"Ball Position: X={ball_pos:.1f}, Y={ball_y:.1f}\n\n"
    
    analysis += "REASONING:\n"
    analysis += f"1. DEFENSIVE SPACING ANALYSIS:\n"
    if gaps:
        analysis += f"   Found {len(gaps)} gaps in defense:\n"
        analysis += gap_analysis
        analysis += f"   => Drive through gaps at these Y-coordinates\n\n"
    else:
        analysis += f"   Defense is tightly packed (no major gaps)\n"
        analysis += f"   => Use screens and picks to create space\n\n"
    
    analysis += f"2. ISOLATION DETECTION:\n"
    analysis += f"   {isolation_count} defenders are isolated from teammates\n"
    if isolated_y:
        analysis += f"   Isolated at Y: {', '.join([f'{y:.1f}' for y in isolated_y])}\n"
    analysis += f"   => Target isolated defenders for pick-and-roll plays\n\n"
    
    analysis += f"3. SIDELINE IMBALANCE:\n"
    analysis += f"   LEFT side: {left_half} defenders | RIGHT side: {right_half} defenders\n"
    analysis += f"   => {weaker_side} side is WEAK ({weaker_count}v{stronger_count})\n"
    analysis += f"   => Attack {weaker_side} side for numerical advantage\n\n"
    
    analysis += f"4. FORWARD PRESSURE:\n"
    analysis += f"   {defenders_forward} defenders ahead of ball | {defenders_back} defenders behind\n"
    if defenders_forward > 2:
        analysis += f"   => Heavy pressure - use backdoor cuts\n\n"
    else:
        analysis += f"   => Light pressure - aggressive driving lanes available\n\n"
    
    # Generate DYNAMIC counter-positions based on analysis
    analysis += "RECOMMENDED COUNTER-POSITIONS:\n"
    
    # Primary gap (largest)
    primary_gap_y = gaps[0][0] if gaps else 25
    primary_gap_size = gaps[0][1] if gaps else 0
    
    # Ball advance direction and weak side positioning
    if possession == 0:  # Team 1 attacking toward right (X=100)
        advance_x = min(ball_pos + 25, 90)
        # Dynamic weak side based on defender distribution
        if weaker_side == "RIGHT":
            weak_side_x = ball_pos + np.random.uniform(10, 20)
        else:
            weak_side_x = ball_pos - np.random.uniform(5, 15)
    else:  # Team 2 attacking toward left (X=0)
        advance_x = max(ball_pos - 25, 10)
        if weaker_side == "LEFT":
            weak_side_x = ball_pos - np.random.uniform(10, 20)
        else:
            weak_side_x = ball_pos + np.random.uniform(5, 15)
    
    # Position 1: Ball handler - varies based on gaps, ball position, and weak side
    if len(gaps) >= 2:
        # If multiple gaps, position between them
        handler_y = (gaps[0][0] + gaps[1][0]) / 2 + np.random.uniform(-5, 5)
    else:
        handler_y = primary_gap_y + np.random.uniform(-8, 8)
    handler_x = max(10, min(90, weak_side_x + np.random.uniform(-10, 10)))
    analysis += f"  P1 (Handler): X={handler_x:.0f}, Y={handler_y:.1f} - control weak side attack\n"
    
    # Position 2: Wing - exploit main gap with varied positioning
    wing_y = primary_gap_y + np.random.uniform(-5, 5)
    wing_x = advance_x + np.random.uniform(-10, 10)
    if primary_gap_size > 15:
        wing_action = "exploit wide gap"
    else:
        wing_action = "cut through gap"
    analysis += f"  P2 (Wing): X={wing_x:.0f}, Y={wing_y:.1f} - {wing_action}\n"
    
    # Position 3: Forward - dynamic based on multiple factors
    if len(gaps) > 1:
        # Target secondary gap with variation
        forward_y = gaps[1][0] + np.random.uniform(-6, 6)
        forward_x = advance_x - np.random.uniform(10, 20)
        forward_action = "secondary gap attack"
    elif isolated_y:
        # Attack near isolated defender
        forward_y = isolated_y[0] + np.random.uniform(-10, 10)
        forward_x = advance_x - np.random.uniform(15, 25)
        forward_action = "isolate defender"
    else:
        # Spread formation
        forward_y = primary_gap_y + np.random.uniform(10, 20) if primary_gap_y < 30 else primary_gap_y - np.random.uniform(10, 20)
        forward_x = advance_x - np.random.uniform(12, 18)
        forward_action = "spread floor"
    analysis += f"  P3 (Forward): X={forward_x:.0f}, Y={forward_y:.1f} - {forward_action}\n"
    
    # Position 4: Guard - highly dynamic based on defensive setup
    if isolated_y and len(isolated_y) > 0:
        # Screen near first isolated defender
        screen_y = isolated_y[0] + np.random.uniform(-8, 8)
        screen_x = isolated_x[0] + np.random.uniform(-10, 10)
        guard_action = "screen isolated defender"
    elif len(gaps) >= 3:
        # Exploit third gap
        screen_y = gaps[2][0] + np.random.uniform(-5, 5)
        screen_x = ball_pos + np.random.uniform(5, 20) if possession == 0 else ball_pos - np.random.uniform(5, 20)
        guard_action = "tertiary gap spacing"
    else:
        # Pick and roll support
        screen_y = primary_gap_y + np.random.uniform(-12, 12)
        screen_x = ball_pos + np.random.uniform(8, 18) if possession == 0 else ball_pos - np.random.uniform(8, 18)
        guard_action = "pick and roll"
    analysis += f"  P4 (Guard): X={screen_x:.0f}, Y={screen_y:.1f} - {guard_action}\n"
    
    # Position 5: Center - complex logic based on pressure and spacing
    if defenders_forward > 2:
        # Heavy pressure - backdoor cut through smallest gap
        if len(gaps) > 1:
            smallest_gap = min(gaps, key=lambda g: g[1])
            center_y = smallest_gap[0] + np.random.uniform(-4, 4)
        else:
            center_y = primary_gap_y + np.random.uniform(-8, 8)
        center_x = ball_pos + np.random.uniform(-25, -15) if possession == 0 else ball_pos + np.random.uniform(15, 25)
        center_action = "backdoor cut"
    elif primary_gap_size > 15:
        # Wide gap - aggressive post-up
        center_y = primary_gap_y + np.random.uniform(-7, 7)
        center_x = advance_x + np.random.uniform(-8, 8)
        center_action = "aggressive post"
    else:
        # Standard high post
        center_y = primary_gap_y + np.random.uniform(8, 15) if primary_gap_y < 25 else primary_gap_y - np.random.uniform(8, 15)
        center_x = ball_pos + np.random.uniform(5, 15) if possession == 0 else ball_pos - np.random.uniform(5, 15)
        center_action = "high post support"
    analysis += f"  P5 (Center): X={center_x:.0f}, Y={center_y:.1f} - {center_action}\n"
    
    return analysis


def tactical_analysis_worker(game_states_queue, analysis_queue, api_key):
    """Worker thread to analyze game states in real-time"""
    last_printed_frame = -10
    last_possession = -1
    last_analysis_type = ""
    
    while True:
        try:
            state_data = game_states_queue.get(timeout=1)
            if state_data is None:
                break
            
            frame, obs = state_data
            score1, score2, possession, time_remaining, ball_pos = obs
            
            # Analyze every frame to track everything
            if frame % 2 != 0:
                continue
            
            attacking_team, defending_team = generate_player_positions(
                ball_pos, possession, frame
            )
            
            # Generate analysis
            analysis = generate_local_tactical_analysis(
                attacking_team, defending_team, ball_pos, possession
            )
            
            analysis_queue.put({
                'frame': frame,
                'analysis': analysis,
                'score1': int(score1),
                'score2': int(score2)
            })
            
            # Print whenever there's meaningful change or time elapsed
            should_print = (
                last_possession != possession or  # possession changed
                frame - last_printed_frame >= 6   # or 6+ frames passed
            )
            
            if should_print:
                print("\n" + "="*70)
                print(f"FRAME {frame} | Time: {int(time_remaining//60):02d}:{int(time_remaining%60):02d} | Score: Team1={int(score1)} Team2={int(score2)}")
                print("="*70)
                print(analysis)
                last_printed_frame = frame
                last_possession = possession
        
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Analysis error: {e}")


def run_realtime_simulation():
    """Run basketball simulation with real-time tactical analysis"""
    print("=" * 70)
    print("BASKETBALL SIMULATION WITH REAL-TIME TACTICAL ANALYSIS")
    print("=" * 70)
    print()
    
    # Setup environment
    env = BasketballEnv()
    obs, _ = env.reset()
    
    # Queues for inter-thread communication
    game_states_queue = queue.Queue()
    analysis_queue = queue.Queue()
    
    # Start analysis worker thread
    api_key = os.getenv('HUGGINGFACE_API_KEY', '')
    analysis_thread = Thread(
        target=tactical_analysis_worker,
        args=(game_states_queue, analysis_queue, api_key),
        daemon=True
    )
    analysis_thread.start()
    
    # Setup figure with two subplots
    fig = plt.figure(figsize=(16, 7))
    ax_court = fig.add_subplot(121)
    ax_info = fig.add_subplot(122)
    
    draw_basketball_court(ax_court)
    
    # Storage for visualization data
    simulation_data = {
        'game_states': [],
        'frame': 0,
        'running': True,
        'analysis': {},
        'score_history': []
    }
    
    def run_simulation():
        """Run game simulation in background"""
        frame = 0
        while frame < 200:  # Simulate 200 frames
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            simulation_data['game_states'].append(obs)
            game_states_queue.put((frame, obs))
            
            score1, score2 = obs[0], obs[1]
            simulation_data['score_history'].append((frame, score1, score2))
            
            frame += 1
            if done or truncated:
                break
        
        simulation_data['running'] = False
        game_states_queue.put(None)
    
    # Start simulation thread
    sim_thread = Thread(target=run_simulation, daemon=True)
    sim_thread.start()
    
    # Animation update function
    def update_frame(frame_idx):
        ax_court.clear()
        ax_info.clear()
        
        draw_basketball_court(ax_court)
        
        # Get current game state
        if frame_idx < len(simulation_data['game_states']):
            obs = simulation_data['game_states'][frame_idx]
            score1, score2, possession, time_remaining, ball_pos = obs
            
            # Generate and display player positions
            team1_players, team2_players = generate_player_positions(
                ball_pos, possession, frame_idx
            )
            
            # Get ball Y from ball handler
            ball_handler = team1_players[0] if possession == 0 else team2_players[0]
            ball_y = ball_handler['y']
            
            # Plot players - colors based on possession status
            for player in team1_players:
                # Team 1 is attacking (cyan) when possession == 0, defending (lightblue) when possession == 1
                color = 'cyan' if possession == 0 else 'lightblue'
                ax_court.plot(player['x'], player['y'], 'o', color=color, markersize=12)
                ax_court.text(player['x'], player['y'], str(player['id']), 
                            ha='center', va='center', fontweight='bold', fontsize=9)
            
            for player in team2_players:
                # Team 2 is attacking (red) when possession == 1, defending (lightcoral) when possession == 0
                color = 'red' if possession == 1 else 'lightcoral'
                ax_court.plot(player['x'], player['y'], 'o', color=color, markersize=12)
                ax_court.text(player['x'], player['y'], str(player['id']), 
                            ha='center', va='center', fontweight='bold', fontsize=9, color='white')
            
            # Plot ball at actual Y position
            ax_court.plot(ball_pos, ball_y, 's', color='orange', markersize=10, label='Ball')
            
            # Display info
            if possession == 0:
                possession_team = "Team 1 (Cyan - Attacking)"
            else:
                possession_team = "Team 2 (Red - Attacking)"
            
            info_text = f"""
FRAME: {frame_idx}
TIME: {int(time_remaining//60):02d}:{int(time_remaining%60):02d}

SCORE:
  Team 1: {int(score1)}
  Team 2: {int(score2)}

BALL POSITION: X={ball_pos:.1f}, Y={ball_y:.1f}

POSSESSION: {possession_team}

"""
            
            # Check for new tactical analysis
            try:
                analysis_data = analysis_queue.get_nowait()
                simulation_data['analysis'] = analysis_data
            except queue.Empty:
                pass
            
            # Add tactical analysis if available
            if simulation_data['analysis']:
                info_text += f"FRAME {simulation_data['analysis']['frame']} ANALYSIS:\n"
                info_text += simulation_data['analysis']['analysis']
            else:
                info_text += "Analyzing play...\n"
            
            ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                        fontfamily='monospace', fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax_info.axis('off')
        
        return [ax_court, ax_info]
    
    # Create animation
    anim = FuncAnimation(fig, update_frame, frames=200, interval=500, 
                        repeat=False, blit=False)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_realtime_simulation()
