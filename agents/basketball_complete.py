#!/usr/bin/env python3
"""
Basketball Tactical Analysis System - All-in-One
Complete system with game simulation, LLM integration, strategy analysis, and comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import requests
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Tuple, Optional
from threading import Thread
import queue
import time
import argparse
import json
import csv
import random


# ============================================================================
# PART 1: BASKETBALL ENVIRONMENT (Gymnasium)
# ============================================================================

class BasketballEnv(gym.Env):
    """Custom Basketball Environment for Gymnasium"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self):
        super(BasketballEnv, self).__init__()
        
        # Action space: 0=Pass, 1=Dribble Forward, 2=Dribble Back, 3=Shoot
        self.action_space = spaces.Discrete(4)
        
        # Observation: [team1_score, team2_score, possession, time_remaining, ball_position]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([200, 200, 1, 2400, 100]),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.team1_score = 0
        self.team2_score = 0
        self.possession = 0  # 0 = Team 1, 1 = Team 2
        self.time_remaining = 2400  # 40 minutes in seconds
        self.ball_position = 50  # Center court
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.array([
            self.team1_score,
            self.team2_score,
            self.possession,
            self.time_remaining,
            self.ball_position
        ], dtype=np.float32)
    
    def step(self, action):
        # Time passes
        self.time_remaining -= 5
        
        # Execute action
        if action == 0:  # Pass
            self.ball_position += np.random.randint(-10, 10)
        elif action == 1:  # Dribble Forward
            if self.possession == 0:
                self.ball_position = min(100, self.ball_position + 10)
            else:
                self.ball_position = max(0, self.ball_position - 10)
        elif action == 2:  # Dribble Back
            if self.possession == 0:
                self.ball_position = max(0, self.ball_position - 5)
            else:
                self.ball_position = min(100, self.ball_position + 5)
        elif action == 3:  # Shoot
            if np.random.random() < 0.4:  # 40% shooting accuracy
                if self.possession == 0:
                    self.team1_score += 2
                    reward = 10
                else:
                    self.team2_score += 2
                    reward = 10
                self.possession = 1 - self.possession
                self.ball_position = 50
            else:
                reward = -5
                self.possession = 1 - self.possession
        
        # Random turnover
        if np.random.random() < 0.1:
            self.possession = 1 - self.possession
        
        # Clip ball position
        self.ball_position = np.clip(self.ball_position, 0, 100)
        
        # Check if game is over
        done = self.time_remaining <= 0
        truncated = False
        reward = 0
        
        return self._get_obs(), reward, done, truncated, {}


# ============================================================================
# PART 2: OPEN SOURCE LLM INTEGRATION
# ============================================================================

class OpenSourceLLM:
    """Integration with open-source LLM APIs (Groq, Together AI, Hugging Face)"""
    
    def __init__(self, provider="groq"):
        self.provider = provider
        self.api_key = None
        self.api_url = None
        self.model_name = None
        self._setup_provider()
    
    def _setup_provider(self):
        """Setup API configuration based on provider"""
        if self.provider == "groq":
            self.api_key = os.getenv('GROQ_API_KEY')
            self.api_url = "https://api.groq.com/openai/v1/chat/completions"
            self.model_name = "llama-3.1-8b-instant"
        elif self.provider == "together":
            self.api_key = os.getenv('TOGETHER_API_KEY')
            self.api_url = "https://api.together.xyz/v1/chat/completions"
            self.model_name = "meta-llama/Llama-3-8b-chat-hf"
        elif self.provider == "huggingface":
            self.api_key = os.getenv('HUGGINGFACE_API_KEY')
            self.api_url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
            self.model_name = "Meta-Llama-3-8B-Instruct"
    
    def generate_tactical_analysis(self, attacking_positions, defending_positions, ball_pos, possession):
        """Generate tactical analysis using LLM"""
        if not self.api_key:
            raise ValueError(f"API key not set for {self.provider}")
        
        prompt = self._format_tactical_prompt(attacking_positions, defending_positions, ball_pos, possession)
        
        try:
            response = self._call_api(prompt)
            return response
        except Exception as e:
            raise Exception(f"LLM API Error: {e}")
    
    def _format_tactical_prompt(self, attacking_positions, defending_positions, ball_pos, possession):
        """Format basketball game state into LLM prompt"""
        team_name = "Team 1" if possession == 0 else "Team 2"
        
        atk_str = "\n".join([f"  P{pid}: X={x:.1f}, Y={y:.1f}" 
                             for pid, (x, y) in sorted(attacking_positions.items())])
        def_str = "\n".join([f"  D{pid}: X={x:.1f}, Y={y:.1f}" 
                             for pid, (x, y) in sorted(defending_positions.items())])
        
        prompt = f"""You are a professional basketball tactical analyst. Analyze this defensive formation:

GAME STATE:
Ball Position: X={ball_pos:.1f}
Attacking Team: {team_name}
Court: X-axis 0-100 (length), Y-axis 0-50 (width)

ATTACKING POSITIONS:
{atk_str}

DEFENDING POSITIONS:
{def_str}

TASK: Provide tactical analysis with:
1. DEFENSIVE GAPS: Identify spacing weaknesses (gaps > 10 units between defenders)
2. ISOLATED DEFENDERS: Players >15 units from teammates
3. COUNTER-STRATEGY: Recommend specific attacking moves
4. OPTIMAL POSITIONS: Where should attacking players move (give X,Y coordinates)

Keep response under 200 words, be specific and actionable."""
        
        return prompt
    
    def _call_api(self, prompt):
        """Call the LLM API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        if self.provider in ["groq", "together"]:
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a professional basketball tactical analyst."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        
        elif self.provider == "huggingface":
            data = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            }
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', str(result))
            return str(result)


# ============================================================================
# PART 3: DEFENSIVE STRATEGY ANALYZER
# ============================================================================

class DefensiveStrategyAnalyzer:
    """Analyzes attacking formations and generates defensive strategies"""
    
    def __init__(self):
        self.attacking_positions_history = []
        self.collection_start_time = None
        self.COLLECTION_DURATION = 5
        self.frame_count = 0
    
    def record_attacking_positions(self, frame: int, positions: Dict[int, Tuple[float, float]]) -> bool:
        """Record attacking team positions over time"""
        if self.collection_start_time is None:
            self.collection_start_time = time.time()
        
        self.attacking_positions_history.append({
            'frame': frame,
            'timestamp': time.time() - self.collection_start_time,
            'positions': positions.copy()
        })
        
        self.frame_count += 1
        elapsed = time.time() - self.collection_start_time
        
        return elapsed >= self.COLLECTION_DURATION
    
    def analyze_attacking_formation(self) -> Dict:
        """Analyze the collected attacking formation data"""
        if not self.attacking_positions_history:
            return {}
        
        all_positions = [h['positions'] for h in self.attacking_positions_history]
        
        avg_positions = {}
        for player_id in range(1, 6):
            x_coords = [pos[player_id][0] for pos in all_positions]
            y_coords = [pos[player_id][1] for pos in all_positions]
            avg_positions[player_id] = (np.mean(x_coords), np.mean(y_coords))
        
        player_y_values = sorted([pos[1] for pos in avg_positions.values()])
        
        gaps = []
        for i in range(len(player_y_values) - 1):
            gap_size = player_y_values[i+1] - player_y_values[i]
            gaps.append(gap_size)
        
        ball_handler = min(avg_positions.items(), key=lambda x: x[1][0])
        
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
    
    def generate_all_strategies(self) -> Tuple[List[Dict], Dict]:
        """Generate all 4 defensive strategies"""
        formation_analysis = self.analyze_attacking_formation()
        
        if not formation_analysis:
            return [], {}
        
        strategies = [
            self._strategy_man_to_man(formation_analysis),
            self._strategy_zone_defense(formation_analysis),
            self._strategy_aggressive_press(formation_analysis),
            self._strategy_sagging_defense(formation_analysis)
        ]
        
        return strategies, formation_analysis
    
    def _strategy_man_to_man(self, analysis: Dict) -> Dict:
        """Strategy 1: Man-to-Man Defense"""
        avg_positions = analysis['avg_positions']
        sorted_attackers = sorted(avg_positions.items(), key=lambda x: x[1][0])
        
        defensive_positions = {}
        for idx, (atk_id, (atk_x, atk_y)) in enumerate(sorted_attackers):
            def_id = idx + 1
            def_x = atk_x + np.random.uniform(-3, 3)
            def_y = atk_y + np.random.uniform(-2, 2)
            defensive_positions[def_id] = (def_x, def_y)
        
        return {
            'name': 'Man-to-Man Defense',
            'positions': defensive_positions,
            'reasoning': [
                'Each defender assigned to closest attacking player',
                'Tighter spacing reduces passing lanes',
                'Better for high-pressure individual defense'
            ],
            'strengths': ['Individual accountability', 'Tight defensive spacing'],
            'weaknesses': ['Vulnerable to pick-and-rolls', 'Can be exploited by ball movement']
        }
    
    def _strategy_zone_defense(self, analysis: Dict) -> Dict:
        """Strategy 2: Zone Defense (2-3)"""
        avg_positions = analysis['avg_positions']
        avg_y = np.mean([p[1] for p in avg_positions.values()])
        
        defensive_positions = {
            1: (25, 15),
            2: (15, avg_y - 10),
            3: (15, avg_y + 10),
            4: (45, avg_y - 12),
            5: (45, avg_y + 12)
        }
        
        return {
            'name': 'Zone Defense (2-3)',
            'positions': defensive_positions,
            'reasoning': [
                'Defenders guard areas rather than specific players',
                'Strong interior protection (paint coverage)',
                'Better for defending drive-and-kick plays'
            ],
            'strengths': ['Paint protection', 'Good rebounding position'],
            'weaknesses': ['Vulnerable to outside shooting', 'Weak against perimeter passes']
        }
    
    def _strategy_aggressive_press(self, analysis: Dict) -> Dict:
        """Strategy 3: Aggressive Full-Court Pressing"""
        avg_positions = analysis['avg_positions']
        ball_handler_id, (ball_x, ball_y) = analysis['ball_handler']
        
        defensive_positions = {
            1: (ball_x - 2, ball_y),
            2: (ball_x - 5, ball_y - 8),
            3: (ball_x - 5, ball_y + 8),
            4: (40, 18),
            5: (40, 32)
        }
        
        return {
            'name': 'Aggressive Press',
            'positions': defensive_positions,
            'reasoning': [
                'Immediate pressure on ball handler',
                'Wing defenders set traps on secondary handlers',
                'High-risk strategy forcing turnovers'
            ],
            'strengths': ['Forces turnovers', 'Disrupts offensive rhythm'],
            'weaknesses': ['High foul risk', 'Vulnerable to cutters']
        }
    
    def _strategy_sagging_defense(self, analysis: Dict) -> Dict:
        """Strategy 4: Sagging Defense (Weak-side Collapse)"""
        gaps = analysis['gaps']
        
        if gaps:
            max_gap_idx = gaps.index(max(gaps))
            weak_side_y = (analysis['player_y_values'][max_gap_idx] + 
                          analysis['player_y_values'][max_gap_idx + 1]) / 2
        else:
            weak_side_y = 25
        
        defensive_positions = {
            1: (20, 25),
            2: (22, weak_side_y),
            3: (25, 38),
            4: (35, 20),
            5: (35, 30)
        }
        
        return {
            'name': 'Sagging Defense',
            'positions': defensive_positions,
            'reasoning': [
                'Strong-side plays tight defense',
                f'Weak-side collapses toward gap (Y~{weak_side_y:.0f})',
                'Protects paint while helping on drives'
            ],
            'strengths': ['Interior protection', 'Prevents cuts'],
            'weaknesses': ['Exposed to three-pointers', 'Weak on weak-side penetration']
        }


# ============================================================================
# PART 4: DEFENSE COMPARISON SYSTEM
# ============================================================================

class DefenseComparator:
    """Compares generated strategies against actual defense"""
    
    def __init__(self):
        self.analyzer = DefensiveStrategyAnalyzer()
        self.actual_defense = None
        self.generated_strategies = None
        self.formation_analysis = None
    
    def set_attacking_positions_history(self, positions_list: List[Dict]):
        """Set the attacking team positions history"""
        for entry in positions_list:
            self.analyzer.attacking_positions_history.append({
                'frame': entry['frame'],
                'timestamp': entry.get('timestamp', entry['frame'] * 0.5),
                'positions': entry['positions']
            })
    
    def set_actual_defense(self, defense_positions: Dict[int, Tuple[float, float]]):
        """Set the actual defense that was used"""
        self.actual_defense = defense_positions
    
    def generate_strategies(self):
        """Generate all 4 strategies"""
        self.generated_strategies, self.formation_analysis = self.analyzer.generate_all_strategies()
    
    def calculate_coverage_score(self, strategy_positions: Dict, attacking_positions: Dict) -> float:
        """Calculate how well the defensive strategy covers attacking positions"""
        if not strategy_positions or not attacking_positions:
            return 0
        
        total_distance = 0
        defender_count = len(strategy_positions)
        
        for def_id, (def_x, def_y) in strategy_positions.items():
            min_distance = float('inf')
            for atk_id, (atk_x, atk_y) in attacking_positions.items():
                distance = np.sqrt((def_x - atk_x)**2 + (def_y - atk_y)**2)
                min_distance = min(min_distance, distance)
            total_distance += min_distance
        
        avg_distance = total_distance / defender_count
        coverage_score = max(0, 100 - (avg_distance * 5))
        return coverage_score
    
    def compare_with_actual(self, strategy: Dict) -> Dict:
        """Compare generated strategy against actual defense"""
        strategy_positions = strategy['positions']
        avg_attacking = self.analyzer.analyze_attacking_formation()['avg_positions']
        
        coverage = self.calculate_coverage_score(strategy_positions, avg_attacking)
        
        overall_score = coverage
        
        return {
            'strategy_name': strategy['name'],
            'coverage_score': coverage,
            'overall_score': overall_score,
            'positions': strategy_positions
        }
    
    def generate_comparison_report(self) -> Dict:
        """Generate comprehensive comparison report"""
        if not self.generated_strategies or not self.actual_defense:
            print("ERROR: Missing strategies or actual defense data")
            return {}
        
        print("\n" + "="*80)
        print("DEFENSIVE STRATEGY COMPARISON & EVALUATION")
        print("="*80)
        
        comparisons = []
        for idx, strategy in enumerate(self.generated_strategies, 1):
            comparison = self.compare_with_actual(strategy)
            comparisons.append(comparison)
            
            print(f"\nSTRATEGY {idx}: {strategy['name']}")
            print(f"Overall Score: {comparison['overall_score']:.1f}/100")
        
        best_comparison = max(comparisons, key=lambda x: x['overall_score'])
        print(f"\n✓ Best Strategy: {best_comparison['strategy_name']} ({best_comparison['overall_score']:.1f}/100)")
        
        return {'comparisons': comparisons, 'best_strategy': best_comparison}


# ============================================================================
    # PART 5: REAL-TIME SIMULATION WITH VISUALIZATION
# ============================================================================

def load_custom_frames(path: str) -> List[Dict]:
    """Load custom coordinate frames from JSON or CSV.

    JSON schema (list of frames):
    [
      {"frame":0, "possession":0, "ball":{"x":12.0,"y":25.0},
       "team1":[{"id":1,"x":..,"y":..},...,{"id":5,...}],
       "team2":[{"id":1,"x":..,"y":..},...,{"id":5,...}]}, ...]

    CSV columns required:
      frame,possession,ball_x,ball_y,
      t1p1_x,t1p1_y,...,t1p5_x,t1p5_y,
      t2p1_x,t2p1_y,...,t2p5_x,t2p5_y
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Custom data file not found: {path}")

    _, ext = os.path.splitext(path)
    ext = ext.lower()
    frames: List[Dict] = []

    if ext in (".json", ".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON must be a list of frame objects")
        for i, item in enumerate(data):
            possession = int(item.get("possession", 0))
            ball = item.get("ball", {})
            bx = float(ball.get("x", 50))
            by = float(ball.get("y", 25))
            t1 = item.get("team1", [])
            t2 = item.get("team2", [])
            if len(t1) != 5 or len(t2) != 5:
                raise ValueError("Each frame must have 5 players for team1 and team2")
            frames.append({
                "frame": int(item.get("frame", i)),
                "possession": possession,
                "ball_x": bx,
                "ball_y": by,
                "team1": [{"id": int(p.get("id", idx+1)), "x": float(p["x"]), "y": float(p["y"]) } for idx, p in enumerate(t1)],
                "team2": [{"id": int(p.get("id", idx+1)), "x": float(p["x"]), "y": float(p["y"]) } for idx, p in enumerate(t2)],
            })
    elif ext == ".csv":
        required = ["frame","possession","ball_x","ball_y"] 
        required += [f"t1p{k}_{c}" for k in range(1,6) for c in ("x","y")]
        required += [f"t2p{k}_{c}" for k in range(1,6) for c in ("x","y")]
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            missing = [c for c in required if c not in header]
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}")
            for row in reader:
                possession = int(float(row["possession"]))
                bx = float(row["ball_x"]) ; by = float(row["ball_y"])
                team1 = [] ; team2 = []
                for k in range(1,6):
                    team1.append({"id": k, "x": float(row[f"t1p{k}_x"]), "y": float(row[f"t1p{k}_y"])})
                    team2.append({"id": k, "x": float(row[f"t2p{k}_x"]), "y": float(row[f"t2p{k}_y"])})
                frames.append({
                    "frame": int(float(row["frame"])),
                    "possession": possession,
                    "ball_x": bx,
                    "ball_y": by,
                    "team1": team1,
                    "team2": team2,
                })
    else:
        raise ValueError("Unsupported file extension. Use .json or .csv")

    return frames

def generate_player_positions(ball_pos, possession, frame):
    """Generate player positions with defensive status"""
    team1_players = []
    team2_players = []
    
    ball_y = 25 + np.random.uniform(-10, 10)
    
    def_variation_y = 8
    def_variation_x = 5
    
    if possession == 0:  # Team 1 attacking
        team1_players.append({'id': 1, 'x': ball_pos, 'y': ball_y, 'status': 'attacking'})
        team1_players.append({'id': 2, 'x': min(ball_pos + 15, 95), 'y': ball_y + np.random.uniform(-10, 10), 'status': 'attacking'})
        team1_players.append({'id': 3, 'x': min(ball_pos + 10, 90), 'y': ball_y + np.random.uniform(-15, 15), 'status': 'attacking'})
        team1_players.append({'id': 4, 'x': max(ball_pos - 15, 5), 'y': ball_y + np.random.uniform(-15, 15), 'status': 'attacking'})
        team1_players.append({'id': 5, 'x': max(ball_pos - 20, 0), 'y': ball_y + np.random.uniform(-10, 10), 'status': 'attacking'})
        
        team2_players.append({'id': 1, 'x': ball_pos + np.random.uniform(-12, 12), 'y': ball_y + np.random.uniform(-def_variation_y, def_variation_y), 'status': 'defending'})
        team2_players.append({'id': 2, 'x': min(ball_pos + 20, 98), 'y': 10 + np.random.uniform(-def_variation_y, def_variation_y), 'status': 'defending'})
        team2_players.append({'id': 3, 'x': min(ball_pos + 18, 96), 'y': 40 + np.random.uniform(-def_variation_y, def_variation_y), 'status': 'defending'})
        team2_players.append({'id': 4, 'x': max(ball_pos - 20, 2), 'y': 15 + np.random.uniform(-def_variation_y, def_variation_y), 'status': 'defending'})
        team2_players.append({'id': 5, 'x': max(ball_pos - 25, 0), 'y': 35 + np.random.uniform(-def_variation_y, def_variation_y), 'status': 'defending'})
    else:  # Team 2 attacking
        team2_players.append({'id': 1, 'x': ball_pos, 'y': ball_y, 'status': 'attacking'})
        team2_players.append({'id': 2, 'x': min(ball_pos + 15, 95), 'y': ball_y + np.random.uniform(-10, 10), 'status': 'attacking'})
        team2_players.append({'id': 3, 'x': min(ball_pos + 10, 90), 'y': ball_y + np.random.uniform(-15, 15), 'status': 'attacking'})
        team2_players.append({'id': 4, 'x': max(ball_pos - 15, 5), 'y': ball_y + np.random.uniform(-15, 15), 'status': 'attacking'})
        team2_players.append({'id': 5, 'x': max(ball_pos - 20, 0), 'y': ball_y + np.random.uniform(-10, 10), 'status': 'attacking'})
        
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
    
    ax.plot([0, 100, 100, 0, 0], [0, 0, 50, 50, 0], 'k-', lw=2)
    ax.plot([50, 50], [0, 50], 'k--', lw=1)
    
    circle = plt.Circle((50, 25), 6, color='black', fill=False, lw=1)
    ax.add_patch(circle)
    
    ax.plot([5], [25], 'co', markersize=8, label='Team 1 Basket (Cyan)')
    ax.plot([95], [25], 'o', color='red', markersize=8, label='Team 2 Basket (Red)')
    
    ax.set_xlabel('Court Length (X)', fontsize=10)
    ax.set_ylabel('Court Width (Y)', fontsize=10)
    ax.set_title('Basketball Court - Real-Time Tactical Analysis', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)


def run_simulation(custom_frames: Optional[List[Dict]] = None):
    """Run the complete basketball simulation with visualization.

    If custom_frames is provided, replay those frames instead of random sim.
    """
    print("="*70)
    print("BASKETBALL SIMULATION WITH REAL-TIME TACTICAL ANALYSIS")
    print("="*70)
    print()
    
    env = None
    if custom_frames is None:
        env = BasketballEnv()
        obs, _ = env.reset()
    
    fig = plt.figure(figsize=(16, 7))
    ax_court = fig.add_subplot(121)
    ax_info = fig.add_subplot(122)
    
    draw_basketball_court(ax_court)
    
    simulation_data = {
        'game_states': [],
        'frame': 0,
        'running': True
    }
    
    if custom_frames is None:
        def run_simulation_loop():
            """Run game simulation in background"""
            frame = 0
            while frame < 200:
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                simulation_data['game_states'].append(obs)
                frame += 1
                if done or truncated:
                    break
            simulation_data['running'] = False
        
        sim_thread = Thread(target=run_simulation_loop, daemon=True)
        sim_thread.start()
    
    def update_frame(frame_idx):
        ax_court.clear()
        ax_info.clear()
        
        draw_basketball_court(ax_court)
        
        if custom_frames is not None and frame_idx < len(custom_frames):
            fr = custom_frames[frame_idx]
            possession = int(fr.get('possession', 0))
            ball_pos = float(fr.get('ball_x', 50.0))
            ball_y = float(fr.get('ball_y', 25.0))
            team1_players = fr.get('team1', [])
            team2_players = fr.get('team2', [])
            score1 = 0
            score2 = 0
            time_remaining = max(0, 2400 - frame_idx * 5)
        elif custom_frames is None and frame_idx < len(simulation_data['game_states']):
            obs = simulation_data['game_states'][frame_idx]
            score1, score2, possession, time_remaining, ball_pos = obs
            
            team1_players, team2_players = generate_player_positions(ball_pos, possession, frame_idx)
            
            ball_handler = team1_players[0] if possession == 0 else team2_players[0]
            ball_y = ball_handler['y']
            
            # Plot players with correct colors
            for player in team1_players:
                color = 'cyan' if possession == 0 else 'lightblue'
                ax_court.plot(player['x'], player['y'], 'o', color=color, markersize=12)
                ax_court.text(player['x'], player['y'], str(player['id']), 
                            ha='center', va='center', fontweight='bold', fontsize=9)
            
            for player in team2_players:
                color = 'red' if possession == 1 else 'lightcoral'
                ax_court.plot(player['x'], player['y'], 'o', color=color, markersize=12)
                ax_court.text(player['x'], player['y'], str(player['id']), 
                            ha='center', va='center', fontweight='bold', fontsize=9, color='white')
            
            ax_court.plot(ball_pos, ball_y, 's', color='orange', markersize=10, label='Ball')
            
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

AI Tactical Analysis Active
Using Groq LLM (Llama-3.1-8b)
"""
            
            ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                        fontfamily='monospace', fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax_info.axis('off')
        
        return [ax_court, ax_info]
    
    total_frames = len(custom_frames) if custom_frames is not None else 200
    anim = FuncAnimation(fig, update_frame, frames=total_frames, interval=500, 
                        repeat=False, blit=False)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# PART 6: AGENTIC DEFENSE COORDINATOR
# ============================================================================

class AgentMemory:
    """Keeps short-term scores and decisions for adaptation."""
    def __init__(self, window: int = 15):
        self.window = window
        self.history: List[Dict] = []  # each: {frame, choice, score, coverage_by_strategy}

    def add(self, record: Dict):
        self.history.append(record)
        if len(self.history) > self.window:
            self.history.pop(0)

    def last_scores(self, strategy_name: str, k: int = 10) -> List[float]:
        vals = [h['coverage_by_strategy'].get(strategy_name) for h in self.history if 'coverage_by_strategy' in h]
        vals = [v for v in vals if v is not None]
        return vals[-k:]

    def trend(self, strategy_name: str, k: int = 6) -> float:
        vals = self.last_scores(strategy_name, k)
        if len(vals) < 2:
            return 0.0
        # simple slope estimate
        xs = list(range(len(vals)))
        mean_x = sum(xs)/len(xs)
        mean_y = sum(vals)/len(vals)
        num = sum((x-mean_x)*(y-mean_y) for x,y in zip(xs, vals))
        den = sum((x-mean_x)**2 for x in xs) or 1.0
        return num/den


class AgenticDefenseCoordinator:
    """Agent that selects strategies to maximize coverage over time.

    Modes:
    - heuristic: pick best coverage with epsilon-greedy exploration
    - llm: ask LLM to choose among candidates using context + recent scores
    """
    def __init__(self, use_llm: bool = False, epsilon: float = 0.1):
        self.use_llm = use_llm
        self.epsilon = epsilon
        self.memory = AgentMemory()
        self.llm: Optional[OpenSourceLLM] = OpenSourceLLM("groq") if use_llm else None

    def select_strategy(self, frame: int, candidates: List[Dict], coverage_by: Dict[str, float]) -> Dict:
        # Exploration
        if not self.use_llm and random.random() < self.epsilon:
            return random.choice(candidates)

        # LLM-driven selection
        if self.use_llm and self.llm and self.llm.api_key:
            summary = "\n".join([f"- {c['name']}: {coverage_by.get(c['name'], 0):.1f}" for c in candidates])
            memo_lines = []
            for s in candidates:
                trend = self.memory.trend(s['name'])
                memo_lines.append(f"{s['name']} trend: {trend:+.2f}/frame")
            memo = "\n".join(memo_lines)
            prompt = f"""
You are selecting a defensive strategy to maximize coverage.
Frame {frame}. Candidate coverage now:
{summary}

Recent trends (positive means improving):
{memo}

Choose one strategy name only (exact match) that will likely improve coverage next.
Respond with just the name.
"""
            try:
                resp = self.llm._call_api(prompt)
                name = (resp or "").strip()
                for c in candidates:
                    if c['name'].lower() in name.lower():
                        return c
            except Exception:
                pass  # fall back to heuristic

        # Heuristic: pick highest coverage; break ties by positive trend
        best = None
        best_score = -1e9
        for c in candidates:
            score = coverage_by.get(c['name'], 0.0)
            t = self.memory.trend(c['name'])
            score += 0.5 * t  # slight boost for improving trend
            if score > best_score:
                best_score = score
                best = c
        return best or candidates[0]


def run_agentic_simulation(custom_frames: Optional[List[Dict]] = None, use_llm: bool = False):
    """Run simulation where an agent chooses a defensive strategy each frame."""
    print("="*70)
    print("AGENTIC DEFENSE SIMULATION (objective: maximize coverage)")
    print("="*70)
    print()

    # We will either generate frames on-the-fly (random) or use custom frames
    env = None
    if custom_frames is None:
        env = BasketballEnv()
        obs, _ = env.reset()

    fig = plt.figure(figsize=(16, 7))
    ax_court = fig.add_subplot(121)
    ax_info = fig.add_subplot(122)
    draw_basketball_court(ax_court)

    coordinator = AgenticDefenseCoordinator(use_llm=use_llm)
    analyzer = DefensiveStrategyAnalyzer()
    comparator = DefenseComparator()

    state = {
        'game_states': [],
        'frame': 0,
        'running': True
    }

    # Background sim only if random mode
    if custom_frames is None:
        def sim_loop():
            frame = 0
            while frame < 200:
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                state['game_states'].append(obs)
                frame += 1
                if done or truncated:
                    break
            state['running'] = False
        Thread(target=sim_loop, daemon=True).start()

    # Keep a small history window for analyzer
    history_window = 8

    def update(frame_idx):
        ax_court.clear()
        ax_info.clear()
        draw_basketball_court(ax_court)

        # Build current frame players and ball
        if custom_frames is not None and frame_idx < len(custom_frames):
            fr = custom_frames[frame_idx]
            possession = int(fr.get('possession', 0))
            ball_pos = float(fr.get('ball_x', 50.0))
            ball_y = float(fr.get('ball_y', 25.0))
            team1_players = fr.get('team1', [])
            team2_players = fr.get('team2', [])
            score1 = 0 ; score2 = 0
            time_remaining = max(0, 2400 - frame_idx*5)
        elif custom_frames is None and frame_idx < len(state['game_states']):
            obs = state['game_states'][frame_idx]
            score1, score2, possession, time_remaining, ball_pos = obs
            team1_players, team2_players = generate_player_positions(ball_pos, possession, frame_idx)
            ball_handler = team1_players[0] if possession == 0 else team2_players[0]
            ball_y = ball_handler['y']
        else:
            return [ax_court, ax_info]

        # Plot base players
        for p in team1_players:
            color = 'cyan' if possession == 0 else 'lightblue'
            ax_court.plot(p['x'], p['y'], 'o', color=color, markersize=10)
            ax_court.text(p['x'], p['y'], str(p['id']), ha='center', va='center', fontsize=8)
        for p in team2_players:
            color = 'red' if possession == 1 else 'lightcoral'
            ax_court.plot(p['x'], p['y'], 'o', color=color, markersize=10)
            ax_court.text(p['x'], p['y'], str(p['id']), ha='center', va='center', fontsize=8, color='white')
        ax_court.plot(ball_pos, ball_y, 's', color='orange', markersize=9)

        # Build attacking positions dict and record for analyzer history
        if possession == 0:
            atk_dict = {p['id']: (p['x'], p['y']) for p in team1_players}
        else:
            atk_dict = {p['id']: (p['x'], p['y']) for p in team2_players}
        analyzer.attacking_positions_history.append({'frame': frame_idx, 'timestamp': frame_idx*0.5, 'positions': atk_dict})
        if len(analyzer.attacking_positions_history) > history_window:
            analyzer.attacking_positions_history.pop(0)

        # Generate candidate strategies from recent history
        candidates, analysis = analyzer.generate_all_strategies()
        coverage_by = {}
        if candidates:
            avg_attacking = analyzer.analyze_attacking_formation().get('avg_positions', atk_dict)
            for s in candidates:
                coverage_by[s['name']] = comparator.calculate_coverage_score(s['positions'], avg_attacking)

            chosen = coordinator.select_strategy(frame_idx, candidates, coverage_by)
            chosen_pos = chosen['positions']
            chosen_score = coverage_by.get(chosen['name'], 0.0)
            coordinator.memory.add({
                'frame': frame_idx,
                'choice': chosen['name'],
                'score': chosen_score,
                'coverage_by_strategy': coverage_by
            })

            # Plot chosen defender positions as green triangles
            for did, (dx, dy) in chosen_pos.items():
                ax_court.plot(dx, dy, marker='^', color='green', markersize=11)
                ax_court.text(dx, dy, f"D{did}", color='black', fontsize=8, ha='center', va='center')
        else:
            chosen = {'name': 'N/A'}
            chosen_score = 0.0

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

BALL: X={ball_pos:.1f}, Y={ball_y:.1f}
POSSESSION: {possession_team}

Agent Mode: {'LLM' if use_llm else 'Heuristic'}
Chosen Strategy: {chosen['name']}
Coverage Now: {chosen_score:.1f}
"""
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                     fontfamily='monospace', fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax_info.axis('off')
        return [ax_court, ax_info]

    frames = len(custom_frames) if custom_frames is not None else 200
    anim = FuncAnimation(fig, update, frames=frames, interval=500, repeat=False, blit=False)
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basketball Tactical Analysis System")
    parser.add_argument("--test-llm", action="store_true", help="Test LLM integration and exit")
    parser.add_argument("--analyze", action="store_true", help="Run defensive strategy analyzer demo and exit")
    parser.add_argument("--compare", action="store_true", help="Run defense comparison demo and exit")
    parser.add_argument("--from-json", type=str, help="Path to JSON file with custom coordinate frames")
    parser.add_argument("--from-csv", type=str, help="Path to CSV file with custom coordinate frames")
    parser.add_argument("--agent", action="store_true", help="Run agentic defense simulation (heuristic)")
    parser.add_argument("--agent-llm", action="store_true", help="Run agentic defense simulation (LLM-guided)")
    args = parser.parse_args()

    if args.test_llm:
        print("Testing LLM Integration...")
        llm = OpenSourceLLM(provider="groq")
        if llm.api_key:
            print("✓ Groq API key found")
            test_positions = {1: (10, 25), 2: (25, 15), 3: (25, 35), 4: (45, 20), 5: (45, 30)}
            test_defense = {1: (15, 25), 2: (30, 18), 3: (30, 32), 4: (50, 22), 5: (50, 28)}
            response = llm.generate_tactical_analysis(test_positions, test_defense, 10, 0)
            print("\nLLM Response:")
            print(response)
        else:
            print("✗ Groq API key not set. Set GROQ_API_KEY environment variable.")

    elif args.analyze:
        print("Running Defensive Strategy Analysis...")
        analyzer = DefensiveStrategyAnalyzer()
        for i in range(10):
            positions = {1: (8+i, 25), 2: (25, 15+i), 3: (25, 35-i), 4: (45, 20+i), 5: (45, 30-i)}
            analyzer.record_attacking_positions(i, positions)
            time.sleep(0.5)
        strategies, analysis = analyzer.generate_all_strategies()
        print(f"\nGenerated {len(strategies)} strategies:")
        for idx, strategy in enumerate(strategies, 1):
            print(f"{idx}. {strategy['name']}")

    elif args.compare:
        print("Running Defense Comparison...")
        comparator = DefenseComparator()
        attacking_history = [
            {'frame': i, 'positions': {1: (8+i, 25), 2: (25, 15), 3: (25, 35), 4: (45, 20), 5: (45, 30)}}
            for i in range(5)
        ]
        actual_defense = {1: (15, 25), 2: (30, 18), 3: (30, 32), 4: (50, 22), 5: (50, 28)}
        comparator.set_attacking_positions_history(attacking_history)
        comparator.set_actual_defense(actual_defense)
        comparator.generate_strategies()
        _ = comparator.generate_comparison_report()

    else:
        custom_frames: Optional[List[Dict]] = None
        if args.from_json:
            custom_frames = load_custom_frames(args.from_json)
        elif args.from_csv:
            custom_frames = load_custom_frames(args.from_csv)

        if args.agent or args.agent_llm:
            run_agentic_simulation(custom_frames=custom_frames, use_llm=args.agent_llm)
        else:
            run_simulation(custom_frames=custom_frames)
