# UI

Frontend chat interface and visualizations for interacting with game analysis.

## Overview

A conversational interface that lets users query basketball game analysis, explore insights, and visualize key statistics—all through natural language.

## Key Features

- **Chat Interface** - Ask questions about games in plain English
- **Game Selector** - Browse and select games to analyze
- **Visualizations** - Shot charts, player comparisons, timeline views
- **Context Preservation** - Follow-up questions maintain conversation flow

## Structure

```
ui/
├── src/
│   ├── components/  # Reusable UI components
│   ├── pages/       # Route-level views
│   ├── hooks/       # Custom React hooks
│   └── utils/       # Helper functions
├── public/          # Static assets
└── styles/          # Global styles
```

## Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

## Environment Variables

```
VITE_API_URL=http://localhost:8000
```

## Tech Stack

- React + TypeScript
- Vite
- Tailwind CSS
