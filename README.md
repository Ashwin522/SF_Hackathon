# Basketball Tactical Analysis System

AI-powered basketball game simulation with real-time tactical analysis using Gymnasium (OpenAI Gym) and open-source LLMs.

## 🏀 Features

- **Real-time Game Simulation** - Basketball gameplay with Gymnasium environment
- **Live Tactical Analysis** - AI-powered defensive strategy recommendations
- **Visual Animation** - Matplotlib visualization of court, players, and ball movement
- **Defensive Strategy Generator** - 4 different defensive strategies with detailed analysis
- **Performance Comparison** - Compare AI strategies against actual game defense
- **Open-Source LLM Integration** - Uses Groq (Llama 3.1), Together AI, or Hugging Face

## 📊 Demo

The system provides:
- Real-time visualization of basketball gameplay
- 4 defensive strategies (Man-to-Man, Zone Defense, Aggressive Press, Sagging Defense)
- Tactical analysis with gap detection, isolation identification, and counter-positioning
- Performance metrics: coverage score, spacing efficiency, paint protection, perimeter coverage

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/basketball-tactical-analysis.git
cd basketball-tactical-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup LLM API (Choose One)

**Option 1: Groq (Recommended - Fastest)**
```bash
# Sign up at https://console.groq.com
export GROQ_API_KEY='your-api-key-here'
```

**Option 2: Together AI**
```bash
# Sign up at https://together.ai
export TOGETHER_API_KEY='your-api-key-here'
```

**Option 3: Hugging Face**
```bash
# Sign up at https://huggingface.co
export HUGGINGFACE_API_KEY='your-token-here'
```

### Run the Simulation

```bash
python realtime_tactical_simulation.py
```

## 📁 Project Structure

```
basketball-tactical-analysis/
├── basketball_env.py                    # Gymnasium environment
├── realtime_tactical_simulation.py      # Main simulation with visualization
├── defensive_strategy_analyzer.py       # Strategy generation system
├── defense_comparison_system.py         # AI vs actual defense comparison
├── opensource_llm_integration.py        # LLM API integration
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## 🎮 Usage Examples

### 1. Run Real-time Simulation
```python
python realtime_tactical_simulation.py
```
- Displays animated basketball court
- Shows live tactical analysis every 2-6 frames
- Updates based on possession changes

### 2. Generate Defensive Strategies
```python
from defensive_strategy_analyzer import DefensiveStrategyAnalyzer

analyzer = DefensiveStrategyAnalyzer()

# Record attacking positions over 5 seconds
attacking_positions = {
    1: (10, 25), 2: (25, 15), 3: (25, 35), 
    4: (45, 20), 5: (45, 30)
}
analyzer.record_attacking_positions(0, attacking_positions)

# Generate 4 strategies
strategies, analysis = analyzer.generate_all_strategies()
analyzer.print_strategies(strategies, analysis)
```

### 3. Compare AI vs Actual Defense
```python
from defense_comparison_system import DefenseComparator

comparator = DefenseComparator()
comparator.set_attacking_positions_history(attacking_history)
comparator.set_actual_defense(actual_defense_positions)
comparator.generate_strategies()
report = comparator.generate_comparison_report()
```

## 🔧 Configuration

### Court Dimensions
- X-axis: 0-100 (length)
- Y-axis: 0-50 (width)
- Team 1 basket: X=5 (left, cyan)
- Team 2 basket: X=95 (right, red)

### Game Parameters
- Duration: 2400 seconds (40 minutes)
- Action interval: 5 seconds
- Shot accuracy: 40%
- Turnover rate: 10%

### LLM Models
- **Groq**: llama-3.1-8b-instant (fastest)
- **Together AI**: meta-llama/Llama-3-8b-chat-hf
- **Hugging Face**: Meta-Llama-3-8B-Instruct

## 📈 Evaluation Metrics

The system scores defensive strategies on:
- **Coverage Score** (35%): How well defenders mark attackers
- **Spacing Efficiency** (25%): Defender distribution across court
- **Paint Protection** (20%): Interior defense strength
- **Perimeter Coverage** (20%): Outside defense coverage

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Gymnasium](https://gymnasium.farama.org/) (OpenAI Gym)
- LLM integration via [Groq](https://groq.com/), [Together AI](https://together.ai/), and [Hugging Face](https://huggingface.co/)
- Visualization with [Matplotlib](https://matplotlib.org/)

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational/research project for basketball tactical analysis. API keys shown in demo are for illustration only - use your own keys in production.
