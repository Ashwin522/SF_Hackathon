# How to Run the Basketball Player Mapping Tool

## Quick Start

1. **Open PowerShell/Terminal** and navigate to this directory:
   ```powershell
   cd "agents\GetCoordinates"
   ```

2. **Install dependencies** (if not already installed):
   ```powershell
   pip install -r requirements.txt
   ```

3. **Run the batch processing script**:
   ```powershell
   python run.py
   ```

## What the Script Does

- Processes all 4 images in `data/images/` directory
- Saves outputs to `outputs/` directory
- Each image gets its own output files:
  - `{image_name}_mapped_players.png` - Visualization image
  - `{image_name}_mapped_players.json` - Coordinates data

## Interactive Process (for each image)

You'll be asked to:

1. **Select visible half**: left or right
2. **Choose preset**: lane_and_ft (recommended) or baseline
3. **Click 4 required landmarks** on the image (instructions provided)
4. **Optionally click 2 more landmarks** (press Enter to skip)
5. **Click on each player's feet** position
6. **Label players**: type 'o' for offense or 'd' for defense
7. **Select ball carrier**: choose which player has the ball

## Output Files

After processing, check the `outputs/` directory for:
- `Screenshot 2026-01-10 122231_mapped_players.png`
- `Screenshot 2026-01-10 122231_mapped_players.json`
- `Screenshot 2026-01-10 154821_mapped_players.png`
- `Screenshot 2026-01-10 154821_mapped_players.json`
- `Screenshot 2026-01-10 154850_mapped_players.png`
- `Screenshot 2026-01-10 154850_mapped_players.json`
- `Screenshot 2026-01-10 154929_mapped_players.png`
- `Screenshot 2026-01-10 154929_mapped_players.json`

## Troubleshooting

- Make sure you're in the correct directory
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Images should be in `data/images/` directory
- Outputs will be created in `outputs/` directory (created automatically)
