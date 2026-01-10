#!/usr/bin/env python3
"""
Real-time Defense Strategy Integration
Integrates the defensive strategy analyzer with the basketball simulation
to provide live strategy recommendations during gameplay
"""

import numpy as np
import time
from basketball_env import BasketballEnv
from defensive_strategy_analyzer import DefensiveStrategyAnalyzer
from threading import Thread
import queue


def extract_attacking_positions(team_players):
    """Extract attacking team positions from player objects"""
    positions = {}
    for player in team_players:
        positions[player['id']] = (player['x'], player['y'])
    return positions


def run_strategy_analysis_session(duration_seconds=5, analysis_interval=2.5):
    """
    Run a complete defensive strategy analysis session
    Analyzes a real game and provides strategy recommendations
    """
    print("\n" + "="*80)
    print("REAL-TIME DEFENSIVE STRATEGY ANALYSIS SESSION")
    print("="*80)
    print(f"Duration: {duration_seconds} seconds | Analysis every {analysis_interval}s\n")
    
    env = BasketballEnv()
    obs, _ = env.reset()
    
    session_number = 1
    session_start = time.time()
    
    while True:
        # Run analysis every N seconds
        analyzer = DefensiveStrategyAnalyzer()
        collection_start = time.time()
        frame = 0
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS SESSION {session_number} - Analyzing attacking positions...")
        print(f"{'='*80}\n")
        
        collection_complete = False
        while not collection_complete:
            # Step the simulation
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            score1, score2, possession, time_remaining, ball_pos = obs
            
            # Generate player positions for this frame
            from realtime_tactical_simulation import generate_player_positions
            team1_players, team2_players = generate_player_positions(ball_pos, possession, frame)
            
            # Extract attacking team positions based on possession
            if possession == 0:  # Team 1 attacking
                attacking_positions = extract_attacking_positions(team1_players)
                attacking_team = "Team 1 (Cyan)"
                defending_team = "Team 2 (Red)"
            else:  # Team 2 attacking
                attacking_positions = extract_attacking_positions(team2_players)
                attacking_team = "Team 2 (Red)"
                defending_team = "Team 1 (Cyan)"
            
            # Record positions
            collection_complete = analyzer.record_attacking_positions(frame, attacking_positions)
            
            frame += 1
            if done or truncated:
                break
            
            time.sleep(0.1)  # Simulate frame timing
        
        # Generate and display strategies
        strategies, formation_analysis = analyzer.generate_all_strategies()
        
        print(f"\n{'='*80}")
        print(f"ATTACKING TEAM: {attacking_team}")
        print(f"DEFENDING TEAM: {defending_team}")
        print(f"{'='*80}\n")
        
        # Display formation analysis
        print("OFFENSIVE FORMATION ANALYSIS:")
        print(f"  • Type: {formation_analysis['formation_density']}")
        print(f"  • Threat Zone: {formation_analysis['primary_threat_zone']}")
        print(f"  • Horizontal Spread: {formation_analysis['x_spread']:.1f} units")
        print(f"  • Vertical Spread: {formation_analysis['y_spread']:.1f} units")
        print(f"  • Ball Handler: Player {formation_analysis['ball_handler'][0]}")
        print()
        
        # Display all 4 strategies in compact format
        for idx, strategy in enumerate(strategies, 1):
            print(f"{'─'*80}")
            print(f"STRATEGY {idx}: {strategy['name'].upper()}")
            print(f"{'─'*80}")
            
            print("Counter-Positions:")
            for def_id in range(1, 6):
                x, y = strategy['positions'][def_id]
                print(f"  D{def_id} → X={x:.1f}, Y={y:.1f}", end="")
                if def_id % 2 == 0:
                    print()
                else:
                    print("  |  ", end="")
            print()
            
            print("\nKey Points:")
            for reason in strategy['reasoning'][:3]:  # Show first 3 points
                print(f"  • {reason}")
            
            print("\nBest For:", end="")
            for strength in strategy['strengths'][:2]:
                print(f" [{strength}]", end="")
            print("\n")
        
        session_number += 1
        
        if done or truncated:
            print(f"{'='*80}")
            print("GAME ENDED - Session analysis complete")
            print(f"{'='*80}\n")
            break


def interactive_mode():
    """
    Interactive mode: User can input attacking positions manually
    """
    print("\n" + "="*80)
    print("INTERACTIVE DEFENSIVE STRATEGY ANALYZER")
    print("="*80)
    print("\nEnter attacking team coordinates (5 players)")
    print("Format: X Y (space-separated, one per line)")
    print("(Or press Ctrl+C to exit)\n")
    
    analyzer = DefensiveStrategyAnalyzer()
    
    print("Collecting positions over 5 seconds...")
    print("Enter coordinates when ready (1 per player, on separate lines):\n")
    
    try:
        for player_id in range(1, 6):
            coord_str = input(f"Player {player_id} (X Y): ").strip()
            if not coord_str:
                print("Skipping - using random position")
                x, y = np.random.uniform(0, 100), np.random.uniform(0, 50)
            else:
                x, y = map(float, coord_str.split())
            
            positions = {player_id: (x, y)}
            analyzer.record_attacking_positions(player_id - 1, positions)
        
        # Generate strategies
        strategies, formation_analysis = analyzer.generate_all_strategies()
        analyzer.print_strategies(strategies, formation_analysis)
    
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        # Run real-time analysis with game simulation
        try:
            run_strategy_analysis_session(duration_seconds=5)
        except KeyboardInterrupt:
            print("\n\nAnalysis session stopped by user.")
