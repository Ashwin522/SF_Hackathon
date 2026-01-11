"""
NBA Basketball Court 2D Grid Coordinate System

Creates a 94ft × 50ft rectangle with a coordinate grid and court markings.

Coordinate System:
- Origin (0, 0) at the bottom-left corner
- X-axis: 0 to 94 feet (left to right)
- Y-axis: 0 to 50 feet (bottom to top)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np

# NBA Court Dimensions (in feet) - based on official specifications
COURT_DIMENSIONS = {
    'length': 94,              # X-axis
    'width': 50,               # Y-axis
    'center_x': 47,            # Midcourt line x-coordinate
    'center_y': 25,            # Center of court y-coordinate
    'center_circle_diameter': 12,
    'center_circle_radius': 6,
    'free_throw_distance': 19,  # Distance from baseline to free throw line
    'paint_width': 16,          # Lane (paint) width
    'free_throw_circle_diameter': 12,
    'free_throw_circle_radius': 6,
    'rim_center_from_baseline': 5.25,  # 5'3" = 5.25 feet
    'backboard_from_baseline': 4,
    'three_point_arc_radius': 23.75,   # 23'9" = 23.75 feet from rim center
    'corner_three_from_sideline': 3,    # Corner 3 is 3 ft from each sideline
}

# Display Configuration Flags
SHOW_GRID = True      # Show coordinate grid behind court markings
SHOW_LABELS = False   # Annotate key points (rim centers, free throw line, 3pt arc)
SHOW_DIMENSIONS = True  # Show dimension arrows (not implemented, default False)
SHOW_PLAYERS = True   # Show players on the court (5 offense, 5 defense)


def create_grid(width=50, length=94, grid_spacing=2, show_grid=None):
    """
    Create axes and optionally draw grid. Does not draw court markings.
    
    Args:
        width: Width of rectangle (Y-axis) in feet
        length: Length of rectangle (X-axis) in feet
        grid_spacing: Spacing between grid lines in feet
        show_grid: Override global SHOW_GRID flag (None uses global setting)
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    # Use global SHOW_GRID if show_grid parameter is not provided
    if show_grid is None:
        show_grid = SHOW_GRID
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Set background
    ax.set_facecolor('#1a472a')  # Dark green
    
    # Draw rectangle outline (court boundary)
    rect = patches.Rectangle((0, 0), length, width,
                            linewidth=3, edgecolor='white',
                            facecolor='none')
    ax.add_patch(rect)
    
    # Draw grid lines only if SHOW_GRID is True
    if show_grid:
        # Label interval - show labels every 10 feet to avoid clutter
        label_interval = max(grid_spacing, 10) if grid_spacing < 10 else grid_spacing
        
        for x in range(0, int(length) + 1, grid_spacing):
            ax.axvline(x, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            # Label X-axis at intervals
            if x % label_interval == 0:
                ax.text(x, -2, str(x), ha='center', va='top', 
                       fontsize=10, color='white', fontweight='bold')
        
        for y in range(0, int(width) + 1, grid_spacing):
            ax.axhline(y, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            # Label Y-axis at intervals
            if y % label_interval == 0:
                ax.text(-2, y, str(y), ha='right', va='center',
                       fontsize=10, color='white', fontweight='bold')
        
        # Add coordinate info box if grid is shown
        info_text = (
            f"Origin (0, 0): Bottom-left corner\n"
            f"X-axis: 0 to {length} feet\n"
            f"Y-axis: 0 to {width} feet\n"
            f"Grid spacing: {grid_spacing} feet"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
               color='white', family='monospace')
    
    # Set labels and formatting
    ax.set_xlabel('X Coordinate (feet from left baseline)', 
                 fontsize=14, color='white', fontweight='bold')
    ax.set_ylabel('Y Coordinate (feet from bottom sideline)', 
                 fontsize=14, color='white', fontweight='bold')
    ax.set_title(f'NBA Basketball Court - 2D Coordinate System\n{length}ft × {width}ft', 
                fontsize=16, color='white', fontweight='bold')
    
    # Set limits with padding for labels
    ax.set_xlim(-5, length + 5)
    ax.set_ylim(-5, width + 5)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.tick_params(colors='white', labelsize=10)
    
    plt.tight_layout()
    return fig, ax


def draw_center(ax):
    """Draw center court markings: midcourt line and center circle."""
    dims = COURT_DIMENSIONS
    
    # Draw midcourt line
    midcourt_x = dims['center_x']
    ax.axvline(midcourt_x, color='white', linestyle='-', linewidth=2)
    
    # Draw center circle
    center_circle = patches.Circle((dims['center_x'], dims['center_y']),
                                  dims['center_circle_radius'],
                                  linewidth=2, edgecolor='white', fill=False)
    ax.add_patch(center_circle)


def draw_hoop_and_backboard(ax, side="left"):
    """
    Draw hoop (rim), backboard, and restricted area arc for left or right side.
    
    NBA Positions:
    - Left rim center: (5.25, 25)
    - Right rim center: (88.75, 25)
    - Left backboard plane: x = 4
    - Right backboard plane: x = 90
    - Backboard: 6ft width, centered at y=25 (from y=22 to y=28)
    - Rim: circle with radius ~0.75ft (18in diameter)
    - Restricted area arc: radius = 4ft from rim center
    
    Args:
        ax: Matplotlib axes object
        side: "left" or "right"
    """
    dims = COURT_DIMENSIONS
    y_center = dims['center_y']  # 25
    
    # Calculate rim and backboard positions based on side
    if side == "left":
        rim_x = 5.25  # Left rim center
        backboard_x = 4  # Left backboard plane
    else:  # right
        rim_x = dims['length'] - 5.25  # Right rim center = 94 - 5.25 = 88.75
        backboard_x = dims['length'] - 4  # Right backboard plane = 94 - 4 = 90
    
    rim_y = y_center  # 25
    
    # Draw backboard: vertical line segment, 6ft width, centered at y=25
    # Backboard extends from y=22 to y=28
    backboard_width = 6
    backboard_y_start = y_center - backboard_width / 2  # 25 - 3 = 22
    backboard_y_end = y_center + backboard_width / 2    # 25 + 3 = 28
    ax.plot([backboard_x, backboard_x],
           [backboard_y_start, backboard_y_end],
           color='white', linewidth=3)
    
    # Draw rim as a circle (18in diameter = 0.75ft radius)
    rim_radius = 0.75
    rim = patches.Circle((rim_x, rim_y), rim_radius,
                        linewidth=2, edgecolor='orange', fill=True, 
                        facecolor='orange', alpha=0.8)
    ax.add_patch(rim)
    
    # Draw restricted area arc (radius = 4ft from rim center)
    # This is a semicircle extending from the baseline toward the court
    restricted_radius = 4
    if side == "left":
        # Arc extends to the right (toward center court) from baseline
        # Semicircle: from -90 degrees to +90 degrees (relative to rim center)
        theta_start = -np.pi / 2
        theta_end = np.pi / 2
        theta = np.linspace(theta_start, theta_end, 100)
        arc_x = rim_x + restricted_radius * np.cos(theta)
        arc_y = rim_y + restricted_radius * np.sin(theta)
        ax.plot(arc_x, arc_y, color='white', linewidth=2, linestyle='--', alpha=0.8)
    else:  # right
        # Arc extends to the left (toward center court) from baseline
        # Semicircle: from 90 degrees to 270 degrees
        theta_start = np.pi / 2
        theta_end = 3 * np.pi / 2
        theta = np.linspace(theta_start, theta_end, 100)
        arc_x = rim_x + restricted_radius * np.cos(theta)
        arc_y = rim_y + restricted_radius * np.sin(theta)
        ax.plot(arc_x, arc_y, color='white', linewidth=2, linestyle='--', alpha=0.8)


def draw_paint_and_free_throw(ax, side="left"):
    """
    Draw the paint (key/lane) and free throw area for left or right side.
    
    Dimensions:
    - Free throw line: 19ft from baseline
      * Left: x = 19
      * Right: x = 75 (94 - 19)
    - Lane width: 16ft centered on y=25
      * y from 17 to 33 (25 - 8 to 25 + 8)
    - Lane rectangle:
      * Left: x from 0 to 19
      * Right: x from 75 to 94
    - Free throw circle: diameter 12ft (radius 6), centered at (19,25) and (75,25)
      * Half-circle inside lane (toward baseline) is dashed
      * Half-circle outside lane (toward center court) is solid
    
    Args:
        ax: Matplotlib axes object
        side: "left" or "right"
    """
    dims = COURT_DIMENSIONS
    center_y = dims['center_y']  # 25
    
    # Free throw line positions
    if side == "left":
        ft_line_x = 19
        lane_x_start = 0
        lane_x_end = 19
    else:  # right
        ft_line_x = 75  # 94 - 19
        lane_x_start = 75
        lane_x_end = 94
    
    # Lane dimensions (paint/key)
    lane_y_start = 17  # 25 - 8
    lane_y_end = 33    # 25 + 8
    lane_width = 16    # 33 - 17
    lane_length = 19   # Distance from baseline to free throw line
    
    # Draw lane rectangle outline
    if side == "left":
        lane_rect = patches.Rectangle((lane_x_start, lane_y_start),
                                     lane_length,  # x extent: 0 to 19
                                     lane_width,   # y extent: 17 to 33
                                     linewidth=2, edgecolor='white', fill=False)
    else:  # right
        lane_rect = patches.Rectangle((lane_x_start, lane_y_start),
                                     lane_length,  # x extent: 75 to 94
                                     lane_width,   # y extent: 17 to 33
                                     linewidth=2, edgecolor='white', fill=False)
    ax.add_patch(lane_rect)
    
    # Draw free throw line as vertical line spanning y=17 to y=33
    ax.plot([ft_line_x, ft_line_x],
           [lane_y_start, lane_y_end],
           color='white', linestyle='-', linewidth=2)
    
    # Draw free throw circle (diameter 12ft, radius 6)
    ft_radius = 6  # dims['free_throw_circle_radius']
    ft_center_y = center_y  # 25
    
    if side == "left":
        # Circle centered at (19, 25)
        ft_center_x = 19
        
        # Solid semicircle: extends to the right (toward center court, x > 19)
        # This is the half away from the baseline
        theta_solid_start = -np.pi / 2  # -90 degrees
        theta_solid_end = np.pi / 2     # +90 degrees
        theta_solid = np.linspace(theta_solid_start, theta_solid_end, 100)
        circle_x_solid = ft_center_x + ft_radius * np.cos(theta_solid)
        circle_y_solid = ft_center_y + ft_radius * np.sin(theta_solid)
        ax.plot(circle_x_solid, circle_y_solid, color='white', 
               linestyle='-', linewidth=2)
        
        # Dashed semicircle: extends to the left (toward baseline, x < 19)
        # This is the half inside the lane, toward decreasing x
        theta_dashed_start = np.pi / 2   # +90 degrees
        theta_dashed_end = 3 * np.pi / 2  # +270 degrees (or -90)
        theta_dashed = np.linspace(theta_dashed_start, theta_dashed_end, 100)
        circle_x_dashed = ft_center_x + ft_radius * np.cos(theta_dashed)
        circle_y_dashed = ft_center_y + ft_radius * np.sin(theta_dashed)
        ax.plot(circle_x_dashed, circle_y_dashed, color='white',
               linestyle='--', linewidth=2, dashes=(5, 3))
        
    else:  # right
        # Circle centered at (75, 25)
        ft_center_x = 75
        
        # Solid semicircle: extends to the left (toward center court, x < 75)
        # This is the half away from the baseline
        theta_solid_start = np.pi / 2    # +90 degrees
        theta_solid_end = 3 * np.pi / 2  # +270 degrees
        theta_solid = np.linspace(theta_solid_start, theta_solid_end, 100)
        circle_x_solid = ft_center_x + ft_radius * np.cos(theta_solid)
        circle_y_solid = ft_center_y + ft_radius * np.sin(theta_solid)
        ax.plot(circle_x_solid, circle_y_solid, color='white',
               linestyle='-', linewidth=2)
        
        # Dashed semicircle: extends to the right (toward baseline, x > 75)
        # This is the half inside the lane, toward increasing x
        theta_dashed_start = -np.pi / 2  # -90 degrees
        theta_dashed_end = np.pi / 2     # +90 degrees
        theta_dashed = np.linspace(theta_dashed_start, theta_dashed_end, 100)
        circle_x_dashed = ft_center_x + ft_radius * np.cos(theta_dashed)
        circle_y_dashed = ft_center_y + ft_radius * np.sin(theta_dashed)
        ax.plot(circle_x_dashed, circle_y_dashed, color='white',
               linestyle='--', linewidth=2, dashes=(5, 3))


def draw_three_point(ax, side="left"):
    """
    Draw three-point line for left or right side using exact geometry.
    
    Geometry:
    - 3pt arc radius = 23.75ft from rim center
    - Corner lines at y=3 and y=47 (3ft from each sideline)
    - Compute arc intersection: dy = 22, r = 23.75, dx = sqrt(r^2 - dy^2)
    
    Args:
        ax: Matplotlib axes object
        side: "left" or "right"
    """
    dims = COURT_DIMENSIONS
    
    # Rim center positions
    if side == "left":
        rim_x = 5.25  # Left rim center x
        baseline_x = 0
    else:  # right
        rim_x = 88.75  # Right rim center x = 94 - 5.25
        baseline_x = 94
    
    rim_y = 25  # Rim center y (court centerline)
    r = 23.75  # Three-point arc radius
    top_corner_y = 47  # 50 - 3
    bottom_corner_y = 3  # 3 ft from bottom sideline
    
    # Compute arc intersection x-position using geometry
    dy = abs(25 - 3)  # dy = 22 (same for top and bottom due to symmetry)
    dx = np.sqrt(r**2 - dy**2)  # dx = sqrt(23.75^2 - 22^2) ≈ 8.947
    
    # Calculate intersection x-coordinate
    if side == "left":
        x_intersect = rim_x + dx  # x_intersect = 5.25 + dx ≈ 14.197
    else:  # right
        x_intersect = rim_x - dx  # x_intersect = 88.75 - dx ≈ 79.803
    
    # Draw the two corner three horizontal segments (parallel to sidelines)
    # Top corner line (y=47) and bottom corner line (y=3)
    if side == "left":
        # Left: from baseline to arc intersection
        ax.plot([baseline_x, x_intersect], [top_corner_y, top_corner_y],
               color='white', linewidth=2)
        ax.plot([baseline_x, x_intersect], [bottom_corner_y, bottom_corner_y],
               color='white', linewidth=2)
    else:  # right
        # Right: from baseline to arc intersection
        ax.plot([baseline_x, x_intersect], [top_corner_y, top_corner_y],
               color='white', linewidth=2)
        ax.plot([baseline_x, x_intersect], [bottom_corner_y, bottom_corner_y],
               color='white', linewidth=2)
    
    # Draw the arc connecting the endpoints using patches.Arc
    # Arc is centered at rim center (rim_x, rim_y)
    # Width and height are both 2*r = 47.5
    
    # Calculate start and end angles using atan2
    # For top corner (y=47): relative to rim center
    if side == "left":
        # Left side: arc extends to the right (increasing x)
        dx_top = dx
        dy_top = top_corner_y - rim_y  # 47 - 25 = 22
        theta_top = np.degrees(np.arctan2(dy_top, dx_top))  # Angle for top corner
        
        dx_bottom = dx
        dy_bottom = bottom_corner_y - rim_y  # 3 - 25 = -22
        theta_bottom = np.degrees(np.arctan2(dy_bottom, dx_bottom))  # Angle for bottom corner
        
        # Arc runs from bottom corner (negative angle) to top corner (positive angle)
        theta1 = theta_bottom  # Start angle (more negative)
        theta2 = theta_top     # End angle (more positive)
        
    else:  # right
        # Right side: arc extends to the left (decreasing x)
        dx_top = -dx  # Negative for left direction
        dy_top = top_corner_y - rim_y  # 47 - 25 = 22
        theta_top = np.degrees(np.arctan2(dy_top, dx_top))  # Angle for top corner
        
        dx_bottom = -dx  # Negative for left direction
        dy_bottom = bottom_corner_y - rim_y  # 3 - 25 = -22
        theta_bottom = np.degrees(np.arctan2(dy_bottom, dx_bottom))  # Angle for bottom corner
        
        # Arc runs from top corner to bottom corner (swap for right side)
        # Right side arc curves inward from top to bottom
        theta1 = theta_top     # Start angle
        theta2 = theta_bottom  # End angle
    
    # Create and add the arc using patches.Arc
    # patches.Arc uses angles in degrees, where theta=0 is right (positive x)
    arc = patches.Arc((rim_x, rim_y), width=2*r, height=2*r,
                     angle=0,  # Rotation angle (0 = no rotation)
                     theta1=theta1, theta2=theta2,
                     linewidth=2, edgecolor='white', fill=False)
    ax.add_patch(arc)


def draw_throw_in_lines(ax):
    """
    Draw throw-in line marks on the sidelines.
    
    Throw-in lines are placed 28 feet from each baseline on both sidelines.
    - Left side: x = 28
    - Right side: x = 94 - 28 = 66
    - Bottom sideline: y = 0 to y = 1 (short vertical tick)
    - Top sideline: y = 50 to y = 49 (mirrored for symmetry)
    
    Args:
        ax: Matplotlib axes object
    """
    dims = COURT_DIMENSIONS
    throw_in_distance = 28  # 28 feet from baseline
    
    # Left side throw-in lines (x = 28)
    left_x = throw_in_distance
    
    # Bottom sideline (y = 0 to y = 1)
    ax.plot([left_x, left_x], [0, 1],
           color='white', linewidth=2)
    
    # Top sideline (y = 50 to y = 49, mirrored for symmetry)
    ax.plot([left_x, left_x], [dims['width'], dims['width'] - 1],
           color='white', linewidth=2)
    
    # Right side throw-in lines (x = 94 - 28 = 66)
    right_x = dims['length'] - throw_in_distance
    
    # Bottom sideline (y = 0 to y = 1)
    ax.plot([right_x, right_x], [0, 1],
           color='white', linewidth=2)
    
    # Top sideline (y = 50 to y = 49, mirrored for symmetry)
    ax.plot([right_x, right_x], [dims['width'], dims['width'] - 1],
           color='white', linewidth=2)


def draw_labels(ax, show_labels=None):
    """
    Annotate key points on the court if SHOW_LABELS is True.
    
    Labels:
    - Rim centers (left and right)
    - Free throw lines (left and right)
    - Three-point arc radius indicators
    
    Args:
        ax: Matplotlib axes object
        show_labels: Override global SHOW_LABELS flag (None uses global setting)
    """
    # Use global SHOW_LABELS if show_labels parameter is not provided
    if show_labels is None:
        show_labels = SHOW_LABELS
    
    if not show_labels:
        return
    
    dims = COURT_DIMENSIONS
    center_y = dims['center_y']  # 25
    
    # Label rim centers (without arrows)
    # Left rim center: (5.25, 25)
    ax.text(5.25, 25 + 5, 'Rim\n(5.25, 25)', 
           fontsize=9, color='yellow', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='yellow'),
           ha='center', va='bottom', zorder=15)
    
    # Right rim center: (88.75, 25)
    ax.text(88.75, 25 + 5, 'Rim\n(88.75, 25)', 
           fontsize=9, color='yellow', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='yellow'),
           ha='center', va='bottom', zorder=15)
    
    # Label free throw lines (without arrows)
    # Left free throw line: x = 19
    ax.text(19, center_y + 15, 'Free Throw\nLine (19ft)', 
           fontsize=9, color='yellow', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='yellow'),
           ha='center', va='bottom', zorder=15)
    
    # Right free throw line: x = 75
    ax.text(75, center_y + 15, 'Free Throw\nLine (75ft)', 
           fontsize=9, color='yellow', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='yellow'),
           ha='center', va='bottom', zorder=15)
    
    # Label three-point arc radius (without arrows)
    # Left side: from rim center, extend in direction of arc
    r = dims['three_point_arc_radius']  # 23.75
    left_rim_x = 5.25
    # Point on arc directly to the right of rim center
    arc_point_x = left_rim_x + r * np.cos(0)  # 5.25 + 23.75 = 29
    arc_point_y = center_y + r * np.sin(0)  # 25
    
    ax.text(arc_point_x + 3, arc_point_y + 5, f'3pt Arc\nr={r}ft', 
           fontsize=9, color='yellow', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='yellow'),
           ha='left', va='bottom', zorder=15)
    
    # Right side: from rim center, extend in direction of arc
    right_rim_x = 88.75
    arc_point_x_right = right_rim_x + r * np.cos(np.pi)  # 88.75 - 23.75 = 65
    arc_point_y_right = center_y + r * np.sin(np.pi)  # 25
    
    ax.text(arc_point_x_right - 3, arc_point_y_right + 5, f'3pt Arc\nr={r}ft', 
           fontsize=9, color='yellow', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='yellow'),
           ha='right', va='bottom', zorder=15)


def draw_players(ax, show_players=None, num_offense=5, num_defense=5, seed=None):
    """
    Draw random players on the court.
    
    Args:
        ax: Matplotlib axes object
        show_players: Override global SHOW_PLAYERS flag (None uses global setting)
        num_offense: Number of offensive players to draw
        num_defense: Number of defensive players to draw
        seed: Random seed for reproducible player positions (None for random)
    """
    # Use global SHOW_PLAYERS if show_players parameter is not provided
    if show_players is None:
        show_players = SHOW_PLAYERS
    
    if not show_players:
        return
    
    dims = COURT_DIMENSIONS
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
    
    # Player size (radius in feet)
    player_radius = 1.0
    
    # Offensive players (light blue/cyan color)
    offense_color = '#00BFFF'  # Deep sky blue
    offense_edge = '#0066CC'   # Darker blue for edge
    
    # Defensive players (red/orange color)
    defense_color = '#FF6B6B'  # Light red/coral
    defense_edge = '#CC0000'   # Dark red for edge
    
    # Generate random positions for offensive players (on left half of court)
    # Offense attacks the right basket, so they're on the left side (x < 47)
    offense_positions = []
    ball_player_index = 0  # First offensive player has the ball
    
    for i in range(num_offense):
        # Random position in left half of court (x: 2 to 45, y: 2 to 48)
        # Leave small margins to avoid edge overlap
        x = np.random.uniform(2, 45)
        y = np.random.uniform(2, 48)
        offense_positions.append((x, y))
        
        # Draw offensive player as a circle
        player = patches.Circle((x, y), player_radius,
                               linewidth=2, edgecolor=offense_edge, 
                               facecolor=offense_color, alpha=0.9, zorder=10)
        ax.add_patch(player)
        
        # Draw ball with the first offensive player (player with index 0)
        if i == ball_player_index:
            # Draw basketball near the player (offset to the right side)
            ball_radius = 0.4  # Basketball radius (smaller than player)
            ball_x = x + player_radius + 0.3  # Position ball to the right of player
            ball_y = y + 0.2  # Slightly above player center
            
            # Basketball with orange/brown color and characteristic lines
            ball = patches.Circle((ball_x, ball_y), ball_radius,
                                 linewidth=1.5, edgecolor='#8B4513',  # Brown edge
                                 facecolor='#FF8C00', alpha=0.95, zorder=11)  # Dark orange
            ax.add_patch(ball)
            
            # Draw characteristic basketball lines (two curved lines)
            # Horizontal line through center
            line1_y_start = ball_y - ball_radius * 0.7
            line1_y_end = ball_y + ball_radius * 0.7
            ax.plot([ball_x - ball_radius * 0.6, ball_x + ball_radius * 0.6],
                   [ball_y, ball_y], color='#8B4513', linewidth=1, zorder=12)
            # Vertical curved line
            theta_line = np.linspace(-np.pi/2, np.pi/2, 20)
            line2_x = ball_x + ball_radius * 0.4 * np.cos(theta_line)
            line2_y = ball_y + ball_radius * 0.6 * np.sin(theta_line)
            ax.plot(line2_x, line2_y, color='#8B4513', linewidth=1, zorder=12)
    
    # Generate random positions for defensive players (on right half of court)
    # Defense protects the right basket, so they're on the right side (x > 47)
    defense_positions = []
    for _ in range(num_defense):
        # Random position in right half of court (x: 49 to 92, y: 2 to 48)
        x = np.random.uniform(49, 92)
        y = np.random.uniform(2, 48)
        defense_positions.append((x, y))
        
        # Draw defensive player as a circle
        player = patches.Circle((x, y), player_radius,
                               linewidth=2, edgecolor=defense_edge,
                               facecolor=defense_color, alpha=0.9, zorder=10)
        ax.add_patch(player)
    
    # Add legend in a non-overlapping position (lower right)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=offense_color,
              markeredgecolor=offense_edge, markersize=10, label=f'Offense ({num_offense})', 
              markeredgewidth=2, linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=defense_color,
              markeredgecolor=defense_edge, markersize=10, label=f'Defense ({num_defense})',
              markeredgewidth=2, linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='lower right', 
             facecolor='black', edgecolor='white', labelcolor='white', fontsize=9,
             framealpha=0.8, bbox_to_anchor=(0.98, 0.02))
    
    return offense_positions, defense_positions


def draw_court(ax, show_labels=None, show_players=None):
    """
    Draw all court markings by calling helper functions.
    
    Args:
        ax: Matplotlib axes object
        show_labels: Override global SHOW_LABELS flag (None uses global setting)
        show_players: Override global SHOW_PLAYERS flag (None uses global setting)
    """
    # Draw center court markings
    draw_center(ax)
    
    # Draw left side
    draw_hoop_and_backboard(ax, side="left")
    draw_paint_and_free_throw(ax, side="left")
    draw_three_point(ax, side="left")
    
    # Draw right side (mirrored using symmetry: x -> 94 - x)
    draw_hoop_and_backboard(ax, side="right")
    draw_paint_and_free_throw(ax, side="right")
    draw_three_point(ax, side="right")
    
    # Draw throw-in line marks
    draw_throw_in_lines(ax)
    
    # Draw labels if SHOW_LABELS is True
    draw_labels(ax, show_labels=show_labels)
    
    # Draw players if SHOW_PLAYERS is True
    draw_players(ax, show_players=show_players)


if __name__ == "__main__":
    # Create the grid and axes (grid drawn only if SHOW_GRID=True)
    dims = COURT_DIMENSIONS
    fig, ax = create_grid(dims['width'], dims['length'], grid_spacing=2, show_grid=SHOW_GRID)
    
    # Draw all court markings (labels drawn only if SHOW_LABELS=True)
    draw_court(ax, show_labels=SHOW_LABELS, show_players=SHOW_PLAYERS)
    
    # Save and show
    plt.savefig('basketball_court_grid.png', dpi=300, 
               facecolor='black', bbox_inches='tight')
    print(f"Court visualization saved as 'basketball_court_grid.png'")
    print(f"Dimensions: {dims['length']}ft × {dims['width']}ft")
    print(f"SHOW_GRID: {SHOW_GRID}, SHOW_LABELS: {SHOW_LABELS}, SHOW_DIMENSIONS: {SHOW_DIMENSIONS}, SHOW_PLAYERS: {SHOW_PLAYERS}")
    plt.show()
