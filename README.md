# CourtIQ - Basketball Game Analysis Platform

A multi-agent system that analyzes basketball games, providing intelligent insights and a chat-based interface to probe, challenge, or explore game results.

## Overview

CourtIQ uses specialized AI agents to break down basketball games—from player performance to strategic patterns—and lets you interact with the analysis through natural conversation.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                         UI                              │
│              (Chat Interface & Visualizations)          │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                        API                              │
│           (Orchestration & Session Management)          │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                      Agents                             │
│    (Analysis, Stats, Play-by-Play, Strategy Agents)     │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

| Directory | Description |
|-----------|-------------|
| [`/agents`](./agents) | Multi-agent system for game analysis and reasoning |
| [`/api`](./api) | Backend API for orchestration and data flow |
| [`/ui`](./ui) | Frontend chat interface and visualizations |

## Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd CourtIQ

# See individual component READMEs for setup instructions
```

## License

MIT
