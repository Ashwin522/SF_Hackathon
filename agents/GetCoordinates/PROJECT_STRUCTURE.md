# Project Structure

This document describes the cleaned and organized folder structure of the Basketball Player Mapping Tool.

## Directory Layout

```
.
├── src/                                    # Source code
│   ├── map_players.py                     # Main mapping script
│   └── basketball_court_coordinates.py    # Court drawing utilities
│
├── data/                                   # Input data
│   └── images/                            # Input basketball images
│       ├── Screenshot 2026-01-10 122231.png
│       └── NBA-BASKETBALL-COURT-DIMENSIONS_1.webp
│
├── output/                                 # Generated outputs
│   ├── mapped_players.png                 # Visualization image
│   ├── mapped_players.json                # Mapping data
│   └── mapped_players_visualization.png   # Alternative visualization
│
├── tests/                                  # Test scripts (optional)
│
├── run.py                                  # Main entry point script
├── requirements.txt                        # Python dependencies
├── README.md                               # Main documentation
├── .gitignore                              # Git ignore rules
└── PROJECT_STRUCTURE.md                    # This file
```

## File Descriptions

### Source Files (`src/`)

- **`map_players.py`**: Main script that handles:
  - Interactive landmark collection
  - Player position collection with offense/defense labeling
  - Ball carrier selection
  - Homography computation
  - Visualization generation
  - JSON export

- **`basketball_court_coordinates.py`**: Utility module for:
  - NBA court dimension constants
  - Court drawing functions (grid, markings, hoops, etc.)
  - Player visualization helpers

### Data Files (`data/`)

- **`images/`**: Contains input basketball game images
  - Place your basketball screenshots here
  - Currently includes example image: `Screenshot 2026-01-10 122231.png`

### Output Files (`output/`)

- **`mapped_players.png`**: Side-by-side visualization showing:
  - Left: Original image with clicked landmarks and players
  - Right: Top-down court view with mapped positions
  - Color-coded by offense/defense
  - Ball carrier marked with ⚽

- **`mapped_players.json`**: Structured data containing:
  - Landmark pixel and court coordinates
  - Player positions (pixel, unclamped court, clamped court)
  - Team assignments (offense/defense)
  - Ball carrier information

### Configuration Files

- **`requirements.txt`**: Python package dependencies
- **`.gitignore`**: Git ignore patterns for Python projects
- **`run.py`**: Convenience script to run the tool from project root

## Running the Tool

### Method 1: Using run.py (Recommended)
```bash
python run.py
```

### Method 2: Direct execution
```bash
python src/map_players.py
```

### Method 3: Module execution
```bash
python -m src.map_players
```

## Path References

The code uses relative paths:
- Input images: `data/images/`
- Output files: `output/`
- Source imports: Relative imports within `src/`

## Cleanup Summary

The following files were removed during cleanup:
- `test_landmarks.py` - Old test script
- `preview_image.py` - Preview utility
- `PLAN_map_players.md` - Planning document
- `basketball_court_grid.png` - Old test output
- `image_preview.png` - Old preview output
- `README_TESTING.md` - Merged into README.md
- `__pycache__/` directories - Python cache (auto-generated)

## Next Steps

1. Add your basketball images to `data/images/`
2. Run `python run.py` to start mapping
3. Check `output/` for results
