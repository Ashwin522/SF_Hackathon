# Basketball Player Mapping Tool

A Python tool for mapping player positions from broadcast basketball images to a 2D court coordinate system using homography transformation.

## Features

- **Interactive Player Mapping**: Click on player feet positions in images and map them to court coordinates
- **Offense/Defense Differentiation**: Label players as offense or defense with color-coded visualization
- **Ball Carrier Selection**: Identify which player has the ball
- **Multiple Preset Support**: Choose from different landmark presets based on image visibility
- **Robust Homography**: Uses RANSAC for robust perspective transformation
- **Visualization**: Side-by-side view of original image and top-down court with mapped players
- **JSON Export**: Save all mapping data including pixel coordinates, court coordinates, teams, and ball carrier

## Project Structure

```
.
├── src/                          # Source code
│   ├── map_players.py           # Main mapping script
│   └── basketball_court_coordinates.py  # Court drawing utilities
├── data/                         # Input data
│   └── images/                   # Input basketball images
├── output/                       # Generated outputs
│   ├── mapped_players.png       # Visualization image
│   └── mapped_players.json       # Mapping data (coordinates, teams, ball carrier)
├── tests/                        # Test scripts (optional)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Install Python 3.8 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `matplotlib >= 3.5.0`
- `numpy >= 1.21.0`
- `opencv-python >= 4.5.0`

## Usage

### Basic Usage

**Option 1: Use the run script (recommended)**
```bash
python run.py
```

**Option 2: Run directly from src directory**
```bash
python src/map_players.py
```

**Option 3: Run with module syntax**
```bash
python -m src.map_players
```

### Workflow

1. **Select Visible Half**: Choose which half of the court is visible (left/right)

2. **Select Preset**: Choose a preset based on what's visible:
   - `lane_and_ft` - Best for most screenshots (when baseline corners aren't visible)
   - `baseline` - Only if baseline corners are clearly visible

3. **Collect Landmarks**: Click on 4 required landmark points:
   - Follow the on-screen instructions for each point
   - Optionally collect 2 additional landmarks for better accuracy

4. **Collect Players**: 
   - Click on each player's feet (bottom center)
   - After each click, specify if the player is OFFENSE (o) or DEFENSE (d)
   - Players are labeled as O1, O2, ... for offense and D1, D2, ... for defense

5. **Select Ball Carrier**: 
   - After collecting all players, a numbered list appears
   - Select which player has the ball (or skip if no one has it)

6. **View Results**: 
   - Visualization opens showing original image and court view
   - Output files saved to `output/` directory

## Output Files

### `output/mapped_players.png`
Side-by-side visualization:
- Left: Original image with clicked landmarks and players
- Right: Top-down court view with mapped player positions
- Color coding: Green/Yellow = Offense, Red/Orange = Defense
- Ball carrier marked with ⚽ emoji

### `output/mapped_players.json`
JSON file containing:
```json
{
  "landmarks": {
    "landmark_name": {
      "pixel": [u, v],
      "court": [x, y]
    }
  },
  "players": [
    {
      "pixel": [u, v],
      "team": "offense" | "defense",
      "has_ball": true | false,
      "court_unclamped": [x, y],
      "court_clamped": [x, y]
    }
  ]
}
```

## Court Coordinate System

- **Origin (0, 0)**: Bottom-left corner
- **X-axis**: 0 to 94 feet (left to right baseline)
- **Y-axis**: 0 to 50 feet (bottom to top sideline)
- **Midcourt line**: x = 47 feet
- **Free throw line (left)**: x = 19 feet
- **Free throw line (right)**: x = 75 feet

## Tips for Best Results

1. **Landmark Selection**:
   - Click precisely at intersection points (e.g., where lane meets baseline)
   - Be consistent with your clicking accuracy
   - Use optional landmarks for better homography accuracy

2. **Player Selection**:
   - Always click on the player's FEET (bottom center of body)
   - Be consistent with the reference point
   - Collect all 10 players if possible for complete analysis

3. **Troubleshooting**:
   - If validation fails, your landmark points may be too close or collinear
   - If players map to wrong locations, check landmark selections
   - "CALIBRATION BROKEN!" warnings indicate landmark selection issues

## Technical Details

- **Homography Method**: Uses OpenCV's `cv2.findHomography` with RANSAC for robust estimation
- **Validation**: Checks for unique points, non-collinearity, and degenerate transformations
- **Coordinate Clamping**: Clamps mapped coordinates to court bounds [0,94] × [0,50]
- **Visualization**: Uses matplotlib for interactive clicking and court rendering

## License

This project is part of a MongoDB Hackathon submission.
