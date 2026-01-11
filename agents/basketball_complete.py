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
from typing import List, Dict, Tuple
from threading import Thread
import queue
import time


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


def run_simulation():
    """Run the complete basketball simulation with visualization"""
    print("="*70)
    print("BASKETBALL SIMULATION WITH REAL-TIME TACTICAL ANALYSIS")
    print("="*70)
    print()
    
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
        
        if frame_idx < len(simulation_data['game_states']):
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
    
    anim = FuncAnimation(fig, update_frame, frames=200, interval=500, 
                        repeat=False, blit=False)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-llm":
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
        
        elif sys.argv[1] == "--analyze":
            print("Running Defensive Strategy Analysis...")
            analyzer = DefensiveStrategyAnalyzer()
            
            # Sample data
            for i in range(10):
                positions = {
                    1: (8+i, 25), 2: (25, 15+i), 3: (25, 35-i), 
                    4: (45, 20+i), 5: (45, 30-i)
                }
                analyzer.record_attacking_positions(i, positions)
                time.sleep(0.5)
            
            strategies, analysis = analyzer.generate_all_strategies()
            print(f"\nGenerated {len(strategies)} strategies:")
            for idx, strategy in enumerate(strategies, 1):
                print(f"{idx}. {strategy['name']}")
        
        elif sys.argv[1] == "--compare":
            print("Running Defense Comparison...")
            comparator = DefenseComparator()
            
            # Sample data
            attacking_history = [
                {'frame': i, 'positions': {
                    1: (8+i, 25), 2: (25, 15), 3: (25, 35), 4: (45, 20), 5: (45, 30)
                }} for i in range(5)
            ]
            actual_defense = {1: (15, 25), 2: (30, 18), 3: (30, 32), 4: (50, 22), 5: (50, 28)}
            
            comparator.set_attacking_positions_history(attacking_history)
            comparator.set_actual_defense(actual_defense)
            comparator.generate_strategies()
            report = comparator.generate_comparison_report()
    else:
        # Default: Run the full simulation
        run_simulation()
