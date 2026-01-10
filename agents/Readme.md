# Agents

Multi-agent system powering the basketball game analysis engine.

## Overview

This layer contains specialized AI agents that collaborate to analyze basketball games from multiple perspectives—statistics, play-by-play breakdowns, strategy evaluation, and more.

## Agent Types

| Agent | Role |
|-------|------|
| **Coordinator** | Orchestrates agent collaboration and routes queries |
| **Stats Analyst** | Processes box scores, shooting charts, and player metrics |
| **Play-by-Play** | Breaks down game flow, momentum shifts, and key sequences |
| **Strategy** | Evaluates offensive/defensive schemes and matchups |
| **Comparator** | Handles historical comparisons and trend analysis |

## Structure

```
agents/
├── coordinator/     # Agent orchestration logic
├── analysts/        # Specialized analysis agents
├── prompts/         # Agent system prompts
└── tools/           # Shared tools and utilities
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

## Usage

```python
from agents import CoordinatorAgent

coordinator = CoordinatorAgent()
response = coordinator.analyze("Why did the Lakers lose in the 4th quarter?")
```
