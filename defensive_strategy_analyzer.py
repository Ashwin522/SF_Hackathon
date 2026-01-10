#!/usr/bin/env python3
"""
Defensive Strategy Analyzer
Analyzes attacking team coordinates over time (5 seconds) and generates 4 defensive strategies
with counter-positions and detailed reasoning
"""

import numpy as np
from typing import List, Dict, Tuple
import time


class DefensiveStrategyAnalyzer:
    """Analyzes attacking formations and generates defensive strategies"""
    
    def __init__(self):
        self.attacking_positions_history = []  # Store positions over time
        self.collection_start_time = None
        self.COLLECTION_DURATION = 5  # seconds
        self.frame_count = 0
    
    def record_attacking_positions(self, frame: int, positions: Dict[int, Tuple[float, float]]) -> bool:
        """
        Record attacking team positions (5 players)
        positions: {1: (x, y), 2: (x, y), ..., 5: (x, y)}
        Returns: True if collection complete, False if still collecting
        """
        if self.collection_start_time is None:
            self.collection_start_time = time.time()
        
        self.attacking_positions_history.append({
            'frame': frame,
            'timestamp': time.time() - self.collection_start_time,
            'positions': positions.copy()
        })
        
        self.frame_count += 1
        elapsed = time.time() - self.collection_start_time
        
        print(f"[Collection {self.frame_count}] Frame {frame} | Elapsed: {elapsed:.1f}s")
        print(f"  Attacking positions: {positions}")
        
        return elapsed >= self.COLLECTION_DURATION
    
    def analyze_attacking_formation(self) -> Dict:
        """Analyze the collected attacking formation data"""
        if not self.attacking_positions_history:
            return {}
        
        # Calculate average positions and movement
        all_positions = [h['positions'] for h in self.attacking_positions_history]
        
        avg_positions = {}
        for player_id in range(1, 6):
            x_coords = [pos[player_id][0] for pos in all_positions]
            y_coords = [pos[player_id][1] for pos in all_positions]
            avg_positions[player_id] = (np.mean(x_coords), np.mean(y_coords))
        
        # Detect formation patterns
        player_y_values = sorted([pos[1] for pos in avg_positions.values()])
        
        # Calculate gaps between players
        gaps = []
        for i in range(len(player_y_values) - 1):
            gap_size = player_y_values[i+1] - player_y_values[i]
            gaps.append(gap_size)
        
        # Identify ball handler (usually at lower X)
        ball_handler = min(avg_positions.items(), key=lambda x: x[1][0])
        
        # Calculate spacing analysis
        x_spread = max([pos[0] for pos in avg_positions.values()]) - min([pos[0] for pos in avg_positions.values()])
        y_spread = max([pos[1] for pos in avg_positions.values()]) - min([pos[1] for pos in avg_positions.values()])
        
        return {
            'avg_positions': avg_positions,
            'player_y_values': player_y_values,
            'gaps': gaps,
            'ball_handler': ball_handler,
            'x_spread': x_spread,
            'y_spread': y_spread,
            'formation_density': self._classify_formation_density(gaps),
            'primary_threat_zone': self._identify_threat_zone(avg_positions)
        }
    
    def _classify_formation_density(self, gaps: List[float]) -> str:
        """Classify attacking formation as tight or spread"""
        avg_gap = np.mean(gaps) if gaps else 0
        if avg_gap < 8:
            return "Tight/Compact"
        elif avg_gap < 12:
            return "Balanced"
        else:
            return "Spread/Open"
    
    def _identify_threat_zone(self, avg_positions: Dict) -> str:
        """Identify where the main offensive threat is"""
        y_positions = [pos[1] for pos in avg_positions.values()]
        avg_y = np.mean(y_positions)
        
        if avg_y < 12:
            return "Upper Half (Y < 12)"
        elif avg_y > 38:
            return "Lower Half (Y > 38)"
        else:
            return "Center Court (12-38)"
    
    def generate_strategy_1_man_to_man(self, formation_analysis: Dict) -> Dict:
        """Strategy 1: Man-to-Man Defense"""
        avg_positions = formation_analysis['avg_positions']
        
        # Each defender marks closest attacker
        defensive_positions = {}
        reasoning = []
        
        # Sort attacking players by X position
        sorted_attackers = sorted(avg_positions.items(), key=lambda x: x[1][0])
        
        for idx, (atk_id, (atk_x, atk_y)) in enumerate(sorted_attackers):
            def_id = idx + 1
            # Position defender close to attacker with slight offset for better coverage
            def_x = atk_x + np.random.uniform(-3, 3)
            def_y = atk_y + np.random.uniform(-2, 2)
            
            defensive_positions[def_id] = (def_x, def_y)
        
        reasoning.append("MAN-TO-MAN DEFENSE:")
        reasoning.append("• Each defender assigned to closest attacking player")
        reasoning.append("• Tighter spacing reduces passing lanes")
        reasoning.append("• Better for high-pressure individual defense")
        reasoning.append("• Defenders stay 1-3 units from their assignment")
        reasoning.append(f"• Formation: {formation_analysis['formation_density']}")
        
        return {
            'name': 'Man-to-Man Defense',
            'positions': defensive_positions,
            'reasoning': reasoning,
            'strengths': ['Individual accountability', 'Tight defensive spacing', 'Minimal help defense needed'],
            'weaknesses': ['Vulnerable to pick-and-rolls', 'Can be exploited by ball movement', 'Less team protection']
        }
    
    def generate_strategy_2_zone_defense(self, formation_analysis: Dict) -> Dict:
        """Strategy 2: Zone Defense (2-3 or 3-2)"""
        avg_positions = formation_analysis['avg_positions']
        gaps = formation_analysis['gaps']
        
        # Zone defense positions based on court areas
        defensive_positions = {}
        
        # D1 - Top of key (protects paint)
        avg_y = np.mean([p[1] for p in avg_positions.values()])
        defensive_positions[1] = (25, 15)
        
        # D2, D3 - Wings covering mid-range
        defensive_positions[2] = (15, avg_y - 10)
        defensive_positions[3] = (15, avg_y + 10)
        
        # D4, D5 - Perimeter/wings
        defensive_positions[4] = (45, avg_y - 12)
        defensive_positions[5] = (45, avg_y + 12)
        
        reasoning = [
            "ZONE DEFENSE (2-3 Setup):",
            "• Defenders guard areas rather than specific players",
            "• Strong interior protection (paint coverage)",
            "• Two high defenders on wings/perimeter",
            "• Better for defending drive-and-kick plays",
            f"• Works well against {formation_analysis['formation_density']} formations"
        ]
        
        if formation_analysis['y_spread'] > 20:
            reasoning.append("• Good choice for spread attacking formation")
        
        return {
            'name': 'Zone Defense (2-3)',
            'positions': defensive_positions,
            'reasoning': reasoning,
            'strengths': ['Paint protection', 'Good rebounding position', 'Defends interior scoring'],
            'weaknesses': ['Vulnerable to outside shooting', 'Weak against perimeter passes', 'Can be spread too thin']
        }
    
    def generate_strategy_3_aggressive_pressing(self, formation_analysis: Dict) -> Dict:
        """Strategy 3: Aggressive Full-Court Pressing"""
        avg_positions = formation_analysis['avg_positions']
        ball_handler_id, (ball_x, ball_y) = formation_analysis['ball_handler']
        
        defensive_positions = {}
        
        # D1 - Tight on ball handler
        defensive_positions[1] = (ball_x - 2, ball_y)
        
        # D2, D3 - Trap wings (cut passing lanes)
        remaining = [(k, v) for k, v in avg_positions.items() if k != ball_handler_id]
        remaining_sorted = sorted(remaining, key=lambda x: x[1][1])  # Sort by Y
        
        defensive_positions[2] = (remaining_sorted[0][1][0] - 3, remaining_sorted[0][1][1] - 2)
        defensive_positions[3] = (remaining_sorted[-1][1][0] - 3, remaining_sorted[-1][1][1] + 2)
        
        # D4, D5 - Defensive help/rotation
        defensive_positions[4] = (40, 18)
        defensive_positions[5] = (40, 32)
        
        reasoning = [
            "AGGRESSIVE FULL-COURT PRESS:",
            "• Immediate pressure on ball handler",
            "• Wing defenders set traps on secondary handlers",
            "• High-risk strategy forcing turnovers",
            "• Requires excellent communication",
            "• Best against teams with poor ball handling"
        ]
        
        if formation_analysis['formation_density'] == "Tight/Compact":
            reasoning.append("• Effective vs tight formations (limited passing options)")
        
        return {
            'name': 'Aggressive Press',
            'positions': defensive_positions,
            'reasoning': reasoning,
            'strengths': ['Forces turnovers', 'Disrupts offensive rhythm', 'Creates scoring opportunities'],
            'weaknesses': ['High foul risk', 'Vulnerable to cutters', 'Requires perfect execution']
        }
    
    def generate_strategy_4_sagging_defense(self, formation_analysis: Dict) -> Dict:
        """Strategy 4: Sagging Defense (Weak-side Collapse)"""
        avg_positions = formation_analysis['avg_positions']
        gaps = formation_analysis['gaps']
        
        # Find largest gap - this is weak side
        if gaps:
            max_gap_idx = gaps.index(max(gaps))
            weak_side_y = (formation_analysis['player_y_values'][max_gap_idx] + 
                          formation_analysis['player_y_values'][max_gap_idx + 1]) / 2
        else:
            weak_side_y = 25
        
        defensive_positions = {}
        
        # D1 - Defensive anchor in paint
        defensive_positions[1] = (20, 25)
        
        # D2, D3 - Weak side sagging
        if weak_side_y < 25:
            defensive_positions[2] = (25, 12)
            defensive_positions[3] = (22, weak_side_y)
        else:
            defensive_positions[2] = (22, weak_side_y)
            defensive_positions[3] = (25, 38)
        
        # D4, D5 - Strong side (active on ball)
        defensive_positions[4] = (35, 20)
        defensive_positions[5] = (35, 30)
        
        reasoning = [
            "SAGGING DEFENSE (Weak-Side Collapse):",
            "• Strong-side plays tight defense",
            f"• Weak-side collapses toward largest gap (Y~{weak_side_y:.0f})",
            "• Protects paint while helping on drives",
            "• Focuses on blocking cutting lanes",
            "• Good for defending inside-out threats"
        ]
        
        if formation_analysis['x_spread'] > 40:
            reasoning.append("• Good vs spread formations with wide spacing")
        
        return {
            'name': 'Sagging Defense',
            'positions': defensive_positions,
            'reasoning': reasoning,
            'strengths': ['Interior protection', 'Prevents cuts', 'Rebounding strength'],
            'weaknesses': ['Exposed to three-pointers', 'Open perimeter shooters', 'Weak on weak-side penetration']
        }
    
    def generate_all_strategies(self) -> List[Dict]:
        """Generate all 4 defensive strategies"""
        formation_analysis = self.analyze_attacking_formation()
        
        if not formation_analysis:
            print("ERROR: No attacking positions recorded")
            return []
        
        strategies = [
            self.generate_strategy_1_man_to_man(formation_analysis),
            self.generate_strategy_2_zone_defense(formation_analysis),
            self.generate_strategy_3_aggressive_pressing(formation_analysis),
            self.generate_strategy_4_sagging_defense(formation_analysis)
        ]
        
        return strategies, formation_analysis
    
    def print_strategies(self, strategies: List[Dict], formation_analysis: Dict):
        """Pretty print all strategies with reasoning"""
        print("\n" + "="*80)
        print("DEFENSIVE STRATEGY ANALYSIS")
        print("="*80)
        
        print("\nATTACKING FORMATION ANALYSIS:")
        print(f"  Formation Type: {formation_analysis['formation_density']}")
        print(f"  Primary Threat Zone: {formation_analysis['primary_threat_zone']}")
        print(f"  Horizontal Spread (X): {formation_analysis['x_spread']:.1f} units")
        print(f"  Vertical Spread (Y): {formation_analysis['y_spread']:.1f} units")
        print(f"  Ball Handler: Player {formation_analysis['ball_handler'][0]} at ({formation_analysis['ball_handler'][1][0]:.1f}, {formation_analysis['ball_handler'][1][1]:.1f})")
        
        print("\n" + "-"*80)
        
        for idx, strategy in enumerate(strategies, 1):
            print(f"\n{'='*80}")
            print(f"STRATEGY {idx}: {strategy['name'].upper()}")
            print("="*80)
            
            print("\nDEFENSIVE COUNTER-POSITIONS:")
            for def_id in range(1, 6):
                x, y = strategy['positions'][def_id]
                print(f"  D{def_id}: X={x:.1f}, Y={y:.1f}")
            
            print("\nREASONING:")
            for reason in strategy['reasoning']:
                print(f"  {reason}")
            
            print("\nSTRENGTHS:")
            for strength in strategy['strengths']:
                print(f"  ✓ {strength}")
            
            print("\nWEAKNESSES:")
            for weakness in strategy['weaknesses']:
                print(f"  ✗ {weakness}")
            
            print()
        
        print("="*80)
        print("RECOMMENDATION:")
        print("Choose strategy based on:")
        print("  • Attacking formation density")
        print("  • Opponent's shooting ability")
        print("  • Team defensive strengths")
        print("  • Risk tolerance (aggressive vs conservative)")
        print("="*80 + "\n")


def demo_with_sample_data():
    """Run demo with sample attacking positions"""
    analyzer = DefensiveStrategyAnalyzer()
    
    # Simulate collecting positions over 5 seconds
    print("DEFENSIVE STRATEGY ANALYZER - DEMO")
    print("Collecting attacking team positions over 5 seconds...\n")
    
    # Generate sample attacking positions with slight variations
    base_positions = {
        1: (8, 25),      # Ball handler (PG)
        2: (25, 15),     # SG
        3: (25, 35),     # SF
        4: (45, 20),     # PF
        5: (45, 30)      # C
    }
    
    collection_complete = False
    frame = 0
    
    while not collection_complete:
        # Add slight variations to simulate movement
        current_positions = {}
        for player_id, (base_x, base_y) in base_positions.items():
            var_x = np.random.uniform(-3, 3)
            var_y = np.random.uniform(-2, 2)
            current_positions[player_id] = (base_x + var_x, base_y + var_y)
        
        collection_complete = analyzer.record_attacking_positions(frame, current_positions)
        frame += 1
        time.sleep(0.5)  # Simulate frame timing
    
    # Analyze and generate strategies
    strategies, formation_analysis = analyzer.generate_all_strategies()
    analyzer.print_strategies(strategies, formation_analysis)


if __name__ == "__main__":
    demo_with_sample_data()
