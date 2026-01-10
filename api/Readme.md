# API

Backend service handling orchestration, session management, and data flow between the UI and agents.

## Overview

The API layer serves as the central hub—managing chat sessions, routing requests to the appropriate agents, and aggregating responses for the frontend.

## Key Features

- **Session Management** - Maintains conversation context and history
- **Agent Orchestration** - Routes queries to specialized agents
- **Data Aggregation** - Combines multi-agent outputs into coherent responses
- **Game Data Integration** - Fetches and caches game statistics

## Structure

```
api/
├── routes/          # API endpoints
├── services/        # Business logic
├── models/          # Data models
└── middleware/      # Auth, logging, error handling
```

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a message and get analysis |
| `GET` | `/games` | List available games |
| `GET` | `/games/:id` | Get game details |
| `GET` | `/sessions/:id` | Retrieve session history |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload
```

## Environment Variables

```
OPENAI_API_KEY=your_key
DATABASE_URL=sqlite:///./courtiq.db
```
