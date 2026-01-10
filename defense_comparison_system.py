#!/usr/bin/env python3
"""
Defense Strategy Comparison & Evaluation System
Compares AI-generated defensive strategies against actual game defense
Scores and evaluates which strategy would have been most effective
"""

import numpy as np
from typing import List, Dict, Tuple
from defensive_strategy_analyzer import DefensiveStrategyAnalyzer


class DefenseComparator:
    """Compares generated strategies against actual defense"""
    
    def __init__(self):
        self.analyzer = DefensiveStrategyAnalyzer()
        self.actual_defense = None
        self.generated_strategies = None
        self.formation_analysis = None
    
    def set_attacking_positions_history(self, positions_list: List[Dict]):
        """
        Set the attacking team positions history
        positions_list: List of {frame: int, positions: {player_id: (x, y)}}
        """
        for entry in positions_list:
            self.analyzer.attacking_positions_history.append({
                'frame': entry['frame'],
                'timestamp': entry.get('timestamp', entry['frame'] * 0.5),
                'positions': entry['positions']
            })
        print(f"✓ Loaded {len(positions_list)} frames of attacking positions")
    
    def set_actual_defense(self, defense_positions: Dict[int, Tuple[float, float]]):
        """
        Set the actual defense that was used
        defense_positions: {1: (x, y), 2: (x, y), ..., 5: (x, y)}
        """
        self.actual_defense = defense_positions
        print(f"✓ Loaded actual defense positions")
        print(f"  Actual Defense:")
        for def_id in range(1, 6):
            x, y = defense_positions[def_id]
            print(f"    D{def_id}: X={x:.1f}, Y={y:.1f}")
    
    def generate_strategies(self):
        """Generate all 4 strategies"""
        self.generated_strategies, self.formation_analysis = self.analyzer.generate_all_strategies()
        print(f"\n✓ Generated 4 defensive strategies")
    
    def calculate_coverage_score(self, strategy_positions: Dict, attacking_positions: Dict) -> float:
        """
        Calculate how well the defensive strategy covers attacking positions
        Score: 0-100 (higher is better)
        """
        if not strategy_positions or not attacking_positions:
            return 0
        
        total_distance = 0
        defender_count = len(strategy_positions)
        
        # For each defender, find closest attacker and measure distance
        for def_id, (def_x, def_y) in strategy_positions.items():
            min_distance = float('inf')
            for atk_id, (atk_x, atk_y) in attacking_positions.items():
                distance = np.sqrt((def_x - atk_x)**2 + (def_y - atk_y)**2)
                min_distance = min(min_distance, distance)
            total_distance += min_distance
        
        avg_distance = total_distance / defender_count
        # Convert to score: closer = higher score
        coverage_score = max(0, 100 - (avg_distance * 5))
        return coverage_score
    
    def calculate_spacing_efficiency(self, strategy_positions: Dict) -> float:
        """
        Calculate defensive spacing efficiency (0-100)
        Good spacing: defenders spread out to cover more court area
        """
        positions_list = [pos for pos in strategy_positions.values()]
        
        if len(positions_list) < 2:
            return 50
        
        # Calculate average distance between defenders
        pairwise_distances = []
        for i in range(len(positions_list)):
            for j in range(i+1, len(positions_list)):
                x1, y1 = positions_list[i]
                x2, y2 = positions_list[j]
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                pairwise_distances.append(distance)
        
        avg_spacing = np.mean(pairwise_distances) if pairwise_distances else 0
        # Optimal spacing is around 15-20 units
        spacing_score = min(100, (avg_spacing / 20) * 100)
        return spacing_score
    
    def calculate_paint_protection(self, strategy_positions: Dict) -> float:
        """
        Calculate interior (paint) protection (0-100)
        Paint is X: 15-35, Y: 15-35 (center area)
        """
        paint_defenders = 0
        for def_id, (x, y) in strategy_positions.items():
            if 15 <= x <= 35 and 15 <= y <= 35:
                paint_defenders += 1
        
        # Ideal: 2-3 defenders in paint
        if paint_defenders <= 1:
            return 40
        elif paint_defenders == 2:
            return 85
        elif paint_defenders == 3:
            return 100
        else:
            return 70  # Too many in paint = bad spacing
    
    def calculate_perimeter_coverage(self, strategy_positions: Dict) -> float:
        """
        Calculate perimeter coverage (0-100)
        Perimeter: X < 15 or X > 35 (wings and backcourt)
        """
        perimeter_defenders = 0
        for def_id, (x, y) in strategy_positions.items():
            if x < 15 or x > 35:
                perimeter_defenders += 1
        
        # Ideal: 2-3 perimeter defenders
        if perimeter_defenders <= 1:
            return 40
        elif perimeter_defenders == 2:
            return 85
        elif perimeter_defenders == 3:
            return 100
        else:
            return 70  # Too many on perimeter = weak paint
    
    def compare_with_actual(self, strategy: Dict, strategy_name: str) -> Dict:
        """
        Compare generated strategy against actual defense
        Returns: comparison scores and analysis
        """
        strategy_positions = strategy['positions']
        
        # Get average attacking positions
        avg_attacking = self.analyzer.analyze_attacking_formation()['avg_positions']
        
        # Calculate various metrics
        coverage = self.calculate_coverage_score(strategy_positions, avg_attacking)
        spacing = self.calculate_spacing_efficiency(strategy_positions)
        paint_prot = self.calculate_paint_protection(strategy_positions)
        perimeter = self.calculate_perimeter_coverage(strategy_positions)
        
        # Overall score (weighted)
        overall_score = (coverage * 0.35 + spacing * 0.25 + paint_prot * 0.20 + perimeter * 0.20)
        
        return {
            'strategy_name': strategy_name,
            'coverage_score': coverage,
            'spacing_score': spacing,
            'paint_protection': paint_prot,
            'perimeter_coverage': perimeter,
            'overall_score': overall_score,
            'positions': strategy_positions
        }
    
    def distance_between_defenses(self, defense1: Dict, defense2: Dict) -> float:
        """Calculate total positional difference between two defenses"""
        total_distance = 0
        for def_id in range(1, 6):
            x1, y1 = defense1[def_id]
            x2, y2 = defense2[def_id]
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            total_distance += distance
        
        return total_distance / 5  # Average distance per defender
    
    def generate_comparison_report(self) -> Dict:
        """Generate comprehensive comparison report"""
        if not self.generated_strategies or not self.actual_defense:
            print("ERROR: Missing strategies or actual defense data")
            return {}
        
        print("\n" + "="*100)
        print("DEFENSIVE STRATEGY COMPARISON & EVALUATION")
        print("="*100)
        
        # Display formation analysis
        print("\nOFFENSIVE FORMATION ANALYZED:")
        print(f"  Formation Type: {self.formation_analysis['formation_density']}")
        print(f"  Threat Zone: {self.formation_analysis['primary_threat_zone']}")
        print(f"  Horizontal Spread: {self.formation_analysis['x_spread']:.1f}")
        print(f"  Vertical Spread: {self.formation_analysis['y_spread']:.1f}")
        
        # Compare each strategy
        comparisons = []
        best_strategy = None
        best_score = -1
        
        print("\n" + "-"*100)
        print("AI-GENERATED STRATEGIES vs ACTUAL DEFENSE")
        print("-"*100)
        
        for idx, strategy in enumerate(self.generated_strategies, 1):
            strategy_name = strategy['name']
            comparison = self.compare_with_actual(strategy, strategy_name)
            comparisons.append(comparison)
            
            # Track best strategy
            if comparison['overall_score'] > best_score:
                best_score = comparison['overall_score']
                best_strategy = strategy_name
            
            # Calculate distance from actual defense
            avg_distance = self.distance_between_defenses(strategy['positions'], self.actual_defense)
            
            print(f"\n{'─'*100}")
            print(f"STRATEGY {idx}: {strategy_name.upper()}")
            print(f"{'─'*100}")
            
            print(f"\nGenerated Positions:")
            for def_id in range(1, 6):
                gen_x, gen_y = strategy['positions'][def_id]
                act_x, act_y = self.actual_defense[def_id]
                diff_x = abs(gen_x - act_x)
                diff_y = abs(gen_y - act_y)
                print(f"  D{def_id}: Gen=({gen_x:.1f},{gen_y:.1f})  |  " +
                      f"Act=({act_x:.1f},{act_y:.1f})  |  " +
                      f"Diff=({diff_x:.1f},{diff_y:.1f})")
            
            print(f"\nEvaluation Scores:")
            print(f"  Coverage Score:        {comparison['coverage_score']:.1f}/100  " +
                  f"{'█'*int(comparison['coverage_score']/5)}  (How well it marks attackers)")
            print(f"  Spacing Efficiency:    {comparison['spacing_score']:.1f}/100  " +
                  f"{'█'*int(comparison['spacing_score']/5)}  (Defender spread)")
            print(f"  Paint Protection:      {comparison['paint_protection']:.1f}/100  " +
                  f"{'█'*int(comparison['paint_protection']/5)}  (Interior defense)")
            print(f"  Perimeter Coverage:    {comparison['perimeter_coverage']:.1f}/100  " +
                  f"{'█'*int(comparison['perimeter_coverage']/5)}  (Outside defense)")
            
            print(f"\n  ⭐ OVERALL STRATEGY SCORE: {comparison['overall_score']:.1f}/100")
            print(f"  Average distance from actual: {avg_distance:.1f} units")
            
            print(f"\nStrategy Description:")
            for reason in strategy['reasoning'][:3]:
                print(f"  • {reason}")
        
        # Final recommendation
        print("\n" + "="*100)
        print("FINAL ANALYSIS & RECOMMENDATION")
        print("="*100)
        
        best_comparison = max(comparisons, key=lambda x: x['overall_score'])
        worst_comparison = min(comparisons, key=lambda x: x['overall_score'])
        actual_avg_score = (best_comparison['overall_score'] + worst_comparison['overall_score']) / 2
        
        print(f"\n✓ Best AI Strategy:    {best_comparison['strategy_name']} ({best_comparison['overall_score']:.1f}/100)")
        print(f"✗ Weakest Strategy:    {worst_comparison['strategy_name']} ({worst_comparison['overall_score']:.1f}/100)")
        print(f"\n📊 Actual Defense Score vs Best Strategy:")
        print(f"   Best AI Strategy:    {best_comparison['overall_score']:.1f}/100")
        print(f"   Difference:          {abs(best_comparison['overall_score'] - actual_avg_score):.1f} points")
        
        if best_comparison['overall_score'] > actual_avg_score:
            print(f"\n   ✓ AI-Generated strategy was BETTER than actual defense")
            improvement = best_comparison['overall_score'] - actual_avg_score
            print(f"   Potential improvement: +{improvement:.1f} points")
        else:
            print(f"\n   ✗ Actual defense was BETTER than AI strategies")
            coach_quality = actual_avg_score - best_comparison['overall_score']
            print(f"   Coaching advantage: +{coach_quality:.1f} points")
        
        print("\n" + "="*100 + "\n")
        
        return {
            'comparisons': comparisons,
            'best_strategy': best_comparison,
            'worst_strategy': worst_comparison,
            'formation_analysis': self.formation_analysis
        }


def demo_comparison():
    """Demo with sample data"""
    print("\n" + "="*100)
    print("DEFENSE STRATEGY COMPARISON SYSTEM - DEMO")
    print("="*100 + "\n")
    
    # Sample attacking positions over time (5 frames)
    attacking_history = [
        {'frame': 0, 'timestamp': 0.0, 'positions': {
            1: (8, 25), 2: (25, 15), 3: (25, 35), 4: (45, 20), 5: (45, 30)
        }},
        {'frame': 1, 'timestamp': 0.5, 'positions': {
            1: (10, 24), 2: (24, 16), 3: (26, 34), 4: (44, 21), 5: (46, 29)
        }},
        {'frame': 2, 'timestamp': 1.0, 'positions': {
            1: (9, 26), 2: (26, 14), 3: (24, 36), 4: (46, 19), 5: (44, 31)
        }},
        {'frame': 3, 'timestamp': 1.5, 'positions': {
            1: (7, 25), 2: (23, 17), 3: (27, 33), 4: (43, 22), 5: (47, 28)
        }},
        {'frame': 4, 'timestamp': 2.0, 'positions': {
            1: (11, 24), 2: (25, 15), 3: (25, 35), 4: (45, 20), 5: (45, 30)
        }}
    ]
    
    # Actual defense used in the game
    actual_defense = {
        1: (15, 25),   # D1
        2: (30, 18),   # D2
        3: (30, 32),   # D3
        4: (50, 22),   # D4
        5: (50, 28)    # D5
    }
    
    # Run comparison
    comparator = DefenseComparator()
    comparator.set_attacking_positions_history(attacking_history)
    comparator.generate_strategies()
    comparator.set_actual_defense(actual_defense)
    report = comparator.generate_comparison_report()


if __name__ == "__main__":
    demo_comparison()
