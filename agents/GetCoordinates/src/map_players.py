"""
Map Players from Broadcast Image to Court Coordinates

Maps manually clicked player feet positions from a broadcast image to court
coordinates (feet) on a 94×50 court using homography transformation.

Coordinate System:
- Origin (0, 0) at bottom-left corner
- X-axis: 0 to 94 feet (left to right baseline)
- Y-axis: 0 to 50 feet (bottom to top sideline)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from basketball_court_coordinates import COURT_DIMENSIONS, create_grid, draw_court


# Court Landmark Coordinates (all coordinates in feet)
# Coordinate system: Origin (0,0) at bottom-left, X [0,94], Y [0,50]
#
# FREE THROW LINE ENDS (always used):
# - Left half: x=19 (19ft from baseline)
# - Right half: x=75 (94-19=75ft from baseline)
# - Bottom end: y=17 (lane bottom edge)
# - Top end: y=33 (lane top edge)

LANDMARKS = {
    "left": {
        # Free throw line ends (LEFT side of court, x=19)
        "ft_left_end": (19, 17),      # Free throw line bottom end (y=17, bottom of lane)
        "ft_right_end": (19, 33),     # Free throw line top end (y=33, top of lane)
        
        # Baseline corners (if baseline is visible)
        "baseline_TL": (0, 50),       # Top-left corner (baseline, top sideline)
        "baseline_TR": (0, 0),        # Bottom-left corner (baseline, bottom sideline)
        
        # Lane corners (paint/key corners)
        "lane_TL": (0, 33),           # Top-left corner of lane (baseline, top of lane)
        "lane_BL": (0, 17),           # Bottom-left corner of lane (baseline, bottom of lane)
        "lane_TR": (19, 33),          # Top-right corner of lane (FT line, top of lane) - SAME as ft_right_end!
        "lane_BR": (19, 17),          # Bottom-right corner of lane (FT line, bottom of lane) - SAME as ft_left_end!
        
        # Midcourt line ends
        "midcourt_T": (47, 50),       # Midcourt line top end (midcourt, top sideline)
        "midcourt_B": (47, 0),        # Midcourt line bottom end (midcourt, bottom sideline)
        
        # Other useful points
        "ft_center": (19, 25),        # Free throw line center (midpoint of lane)
        "rim_center": (5.25, 25),     # Rim center (left side)
        "backboard_center": (4, 25),  # Backboard center (left side)
    },
    "right": {
        # Free throw line ends (RIGHT side of court, x=75)
        "ft_left_end": (75, 17),      # Free throw line bottom end (y=17, bottom of lane)
        "ft_right_end": (75, 33),     # Free throw line top end (y=33, top of lane)
        
        # Baseline corners (if baseline is visible)
        "baseline_TL": (94, 50),      # Top-right corner (baseline, top sideline)
        "baseline_TR": (94, 0),       # Bottom-right corner (baseline, bottom sideline)
        
        # Lane corners (paint/key corners)
        "lane_TL": (75, 33),          # Top-left corner of lane (FT line, top of lane) - SAME as ft_right_end!
        "lane_BL": (75, 17),          # Bottom-left corner of lane (FT line, bottom of lane) - SAME as ft_left_end!
        "lane_TR": (94, 33),          # Top-right corner of lane (baseline, top of lane)
        "lane_BR": (94, 17),          # Bottom-right corner of lane (baseline, bottom of lane)
        
        # Midcourt line ends
        "midcourt_T": (47, 50),       # Midcourt line top end (midcourt, top sideline)
        "midcourt_B": (47, 0),        # Midcourt line bottom end (midcourt, bottom sideline)
        
        # Other useful points
        "ft_center": (75, 25),        # Free throw line center (midpoint of lane)
        "rim_center": (88.75, 25),    # Rim center (right side, 94-5.25)
        "backboard_center": (90, 25), # Backboard center (right side, 94-4)
    }
}


def get_required_points(visible_side, preset):
    """
    Get required and optional landmark points for homography computation.
    
    Each preset provides exactly 4 REQUIRED unique points and 2 OPTIONAL unique points.
    All required points are guaranteed to be unique (no duplicates).
    
    Args:
        visible_side: "left" or "right" - which half of court is visible
        preset: "baseline" or "lane_and_ft" - landmark selection preset
    
    Returns:
        Tuple of (required_points, optional_points) where each is a list of
        (name, (x, y), description) tuples.
        
        required_points: List of exactly 4 unique landmark points
        optional_points: List of exactly 2 unique landmark points
    
    Preset Definitions:
    
    PRESET "lane_and_ft" (Lane Corners + Free Throw Line):
        Best when: Baseline is NOT visible, but lane/paint area is visible (best for most screenshots)
        
        For LEFT side (required, 4 unique points):
        - lane_baseline_bottom = (0, 17) - Bottom-left corner of lane at baseline
        - lane_baseline_top = (0, 33) - Top-left corner of lane at baseline  
        - ft_left_end = (19, 17) - Free throw line bottom end (bottom of lane)
        - ft_center = (19, 25) - Free throw line center (midpoint of lane)
          IMPORTANT: Uses ft_center, NOT ft_right_end, to avoid duplicate with lane_TR
        
        Optional (2 unique points):
        - backboard_center = (4, 25) - Center of backboard
        - rim_center = (5.25, 25) - Center of rim
        
        For RIGHT side (mirror x: x' = 94 - x):
        - lane_baseline_bottom = (94, 17) - Bottom-right corner of lane at baseline
        - lane_baseline_top = (94, 33) - Top-right corner of lane at baseline
        - ft_left_end = (75, 17) - Free throw line bottom end (bottom of lane)
        - ft_center = (75, 25) - Free throw line center (midpoint of lane)
        
        Optional:
        - backboard_center = (90, 25) - Center of backboard (94-4)
        - rim_center = (88.75, 25) - Center of rim (94-5.25)
    
    PRESET "baseline" (Baseline Corners + Free Throw Line Ends):
        Best when: Baseline corners are clearly visible in the image
        
        For LEFT side (required, 4 unique points):
        - baseline_bottom_corner = (0, 0) - Bottom-left corner of court (baseline, bottom sideline)
        - baseline_top_corner = (0, 50) - Top-left corner of court (baseline, top sideline)
        - ft_left_end = (19, 17) - Free throw line bottom end (bottom of lane)
        - ft_right_end = (19, 33) - Free throw line top end (top of lane)
        
        Optional (2 unique points):
        - midcourt_bottom = (47, 0) - Midcourt line bottom end
        - midcourt_top = (47, 50) - Midcourt line top end
        
        For RIGHT side (mirror x: x' = 94 - x):
        - baseline_bottom_corner = (94, 0) - Bottom-right corner of court
        - baseline_top_corner = (94, 50) - Top-right corner of court
        - ft_left_end = (75, 17) - Free throw line bottom end
        - ft_right_end = (75, 33) - Free throw line top end
        
        Optional:
        - midcourt_bottom = (47, 0) - Midcourt line bottom end
        - midcourt_top = (47, 50) - Midcourt line top end
    """
    landmarks = LANDMARKS[visible_side]
    
    if preset == "lane_and_ft":
        # Preset "lane_and_ft": Lane corners at baseline + FT line (bottom end + center)
        # Uses ft_center instead of ft_right_end to avoid duplicate with lane_TR
        if visible_side == "left":
            required = [
                ("lane_baseline_bottom", landmarks["lane_BL"], "Bottom-left corner of LANE at baseline - click where lane/paint meets baseline at bottom (x=0, y=17)"),
                ("lane_baseline_top", landmarks["lane_TL"], "Top-left corner of LANE at baseline - click where lane/paint meets baseline at top (x=0, y=33)"),
                ("ft_left_end", landmarks["ft_left_end"], "Free throw line BOTTOM END - click bottom edge of free throw line where it meets the lane (x=19, y=17)"),
                ("ft_center", landmarks["ft_center"], "Free throw line CENTER - click center point of free throw line (midpoint, x=19, y=25)"),
            ]
            optional = [
                ("backboard_center", landmarks["backboard_center"], "Center of BACKBOARD - click center of backboard rectangle (x=4, y=25)"),
                ("rim_center", landmarks["rim_center"], "Center of RIM/HOOP - click center of basketball rim (x=5.25, y=25)"),
            ]
        else:  # right
            required = [
                ("lane_baseline_bottom", landmarks["lane_BR"], "Bottom-right corner of LANE at baseline - click where lane/paint meets baseline at bottom (x=94, y=17)"),
                ("lane_baseline_top", landmarks["lane_TR"], "Top-right corner of LANE at baseline - click where lane/paint meets baseline at top (x=94, y=33)"),
                ("ft_left_end", landmarks["ft_left_end"], "Free throw line BOTTOM END - click bottom edge of free throw line where it meets the lane (x=75, y=17)"),
                ("ft_center", landmarks["ft_center"], "Free throw line CENTER - click center point of free throw line (midpoint, x=75, y=25)"),
            ]
            optional = [
                ("backboard_center", landmarks["backboard_center"], "Center of BACKBOARD - click center of backboard rectangle (x=90, y=25)"),
                ("rim_center", landmarks["rim_center"], "Center of RIM/HOOP - click center of basketball rim (x=88.75, y=25)"),
            ]
    
    elif preset == "baseline":
        # Preset "baseline": Baseline corners + Free throw line ends
        if visible_side == "left":
            required = [
                ("baseline_bottom_corner", landmarks["baseline_TR"], "Bottom-left CORNER of court - click where baseline meets bottom sideline (x=0, y=0)"),
                ("baseline_top_corner", landmarks["baseline_TL"], "Top-left CORNER of court - click where baseline meets top sideline (x=0, y=50)"),
                ("ft_left_end", landmarks["ft_left_end"], "Free throw line BOTTOM END - click bottom edge of free throw line where it meets the lane (x=19, y=17)"),
                ("ft_right_end", landmarks["ft_right_end"], "Free throw line TOP END - click top edge of free throw line where it meets the lane (x=19, y=33)"),
            ]
            optional = [
                ("midcourt_bottom", landmarks["midcourt_B"], "Midcourt line BOTTOM END - click where midcourt line meets bottom sideline (x=47, y=0)"),
                ("midcourt_top", landmarks["midcourt_T"], "Midcourt line TOP END - click where midcourt line meets top sideline (x=47, y=50)"),
            ]
        else:  # right
            required = [
                ("baseline_bottom_corner", landmarks["baseline_TR"], "Bottom-right CORNER of court - click where baseline meets bottom sideline (x=94, y=0)"),
                ("baseline_top_corner", landmarks["baseline_TL"], "Top-right CORNER of court - click where baseline meets top sideline (x=94, y=50)"),
                ("ft_left_end", landmarks["ft_left_end"], "Free throw line BOTTOM END - click bottom edge of free throw line where it meets the lane (x=75, y=17)"),
                ("ft_right_end", landmarks["ft_right_end"], "Free throw line TOP END - click top edge of free throw line where it meets the lane (x=75, y=33)"),
            ]
            optional = [
                ("midcourt_bottom", landmarks["midcourt_B"], "Midcourt line BOTTOM END - click where midcourt line meets bottom sideline (x=47, y=0)"),
                ("midcourt_top", landmarks["midcourt_T"], "Midcourt line TOP END - click where midcourt line meets top sideline (x=47, y=50)"),
            ]
    else:
        raise ValueError(f"Invalid preset: {preset}. Must be 'baseline' or 'lane_and_ft'.")
    
    # Verify all required points are unique (CRITICAL)
    required_coords = [pt[1] for pt in required]
    unique_coords = set((round(x, 2), round(y, 2)) for x, y in required_coords)
    if len(unique_coords) != len(required_coords):
        duplicates = []
        seen = set()
        for i, coord in enumerate(required_coords):
            coord_key = (round(coord[0], 2), round(coord[1], 2))
            if coord_key in seen:
                duplicates.append(i + 1)
            else:
                seen.add(coord_key)
        raise ValueError(
            f"PRESET {preset} ERROR: Duplicate coordinates in required points at indices {duplicates}!\n"
            f"This should never happen - preset definitions must be corrected."
        )
    
    # Verify all optional points are unique among themselves
    optional_coords = [pt[1] for pt in optional]
    optional_unique = set((round(x, 2), round(y, 2)) for x, y in optional_coords)
    if len(optional_unique) != len(optional_coords):
        print(f"  ⚠ Warning: Preset {preset} has duplicate coordinates in optional points")
    
    # Verify optional points don't duplicate required points
    all_coords = required_coords + optional_coords
    all_unique = set((round(x, 2), round(y, 2)) for x, y in all_coords)
    if len(all_unique) < len(required_coords) + len(optional_coords):
        overlapping = []
        req_set = set((round(x, 2), round(y, 2)) for x, y in required_coords)
        for i, coord in enumerate(optional_coords):
            coord_key = (round(coord[0], 2), round(coord[1], 2))
            if coord_key in req_set:
                overlapping.append(i + 5)  # Optional indices start at (e) = 5
        if overlapping:
            print(f"  ⚠ Warning: Optional points at indices {overlapping} duplicate required points")
    
    return required, optional




def validate_landmark_selection(img_pts, court_pts):
    """
    Validate landmark selection for non-degenerate homography.
    
    Checks:
    1. All court points are unique
    2. All image points are unique (warns if extremely close)
    3. Court points are not collinear (area of triangle formed by any 3 required points)
    
    Args:
        img_pts: List of (u, v) pixel coordinates
        court_pts: List of (x, y) court coordinates
    
    Returns:
        Tuple of (is_valid, error_message)
        is_valid: True if valid, False otherwise
        error_message: Empty string if valid, error message otherwise
    """
    if len(img_pts) < 4:
        return False, f"Need at least 4 correspondences, got {len(img_pts)}"
    
    if len(img_pts) != len(court_pts):
        return False, f"Mismatch: {len(img_pts)} image points vs {len(court_pts)} court points"
    
    # Check 1: All court points are unique
    unique_court = set()
    duplicate_court = []
    for i, (x, y) in enumerate(court_pts):
        coord_key = (round(x, 2), round(y, 2))
        if coord_key in unique_court:
            duplicate_court.append(i + 1)
        else:
            unique_court.add(coord_key)
    
    if len(unique_court) < len(court_pts):
        return False, f"Found duplicate court coordinates at indices {duplicate_court}. All court points must be unique."
    
    if len(unique_court) < 4:
        return False, f"Only {len(unique_court)} unique court coordinates, need at least 4."
    
    # Check 2: All image points are unique (warn if extremely close)
    unique_img = set()
    duplicate_img = []
    close_img = []  # Points that are very close but not identical
    for i, (u, v) in enumerate(img_pts):
        img_key = (round(u), round(v))
        # Check if this point is very close to any existing point (< 5 pixels)
        is_close = False
        for existing_u, existing_v in unique_img:
            dist = np.sqrt((u - existing_u)**2 + (v - existing_v)**2)
            if dist < 5.0 and dist > 0.1:
                close_img.append((i + 1, dist))
                is_close = True
        
        if img_key in unique_img:
            duplicate_img.append(i + 1)
        else:
            unique_img.add((u, v))
    
    if len(unique_img) < len(img_pts):
        return False, f"Found duplicate image coordinates at indices {duplicate_img}. All image points must be unique."
    
    if close_img:
        warnings = [f"Point {idx} is very close to another ({dist:.1f} pixels)" for idx, dist in close_img]
        print(f"  ⚠ Warning: {len(close_img)} image point(s) are very close to others:")
        for warning in warnings:
            print(f"    {warning}")
    
    # Check 3: Non-collinearity check on court points (for first 4 required points)
    if len(court_pts) >= 4:
        court_array = np.array(court_pts[:4], dtype=np.float32)
        
        # Check area of triangles formed by first 3 points with each of the others
        p0, p1, p2 = court_array[0], court_array[1], court_array[2]
        min_area = float('inf')
        
        # Triangle area formula: 0.5 * |det([p1-p0, p2-p0])|
        base_area = abs(0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])))
        min_area = min(min_area, base_area)
        
        # Check triangle with p3
        if len(court_pts) >= 4:
            p3 = court_array[3]
            area1 = abs(0.5 * ((p1[0] - p0[0]) * (p3[1] - p0[1]) - (p3[0] - p0[0]) * (p1[1] - p0[1])))
            area2 = abs(0.5 * ((p2[0] - p0[0]) * (p3[1] - p0[1]) - (p3[0] - p0[0]) * (p2[1] - p0[1])))
            min_area = min(min_area, area1, area2)
        
        # Epsilon threshold: minimum area for valid triangle (in square feet)
        epsilon = 0.5  # 0.5 square feet minimum
        
        if min_area < epsilon:
            return False, (
                f"Court points appear to be collinear (min triangle area: {min_area:.4f} sq ft, "
                f"minimum required: {epsilon} sq ft). Points must be spread across different areas of the court. "
                f"Try selecting landmarks from corners, edges, and center areas."
            )
    
    return True, ""


def collect_landmarks(image_rgb, required_labels, optional_labels):
    """
    Collect landmark clicks from user in specified order.
    
    Shows image with locked axes (no autoscaling). For each required label,
    waits for exactly one click. For optional labels, asks user if they want to click.
    
    If validation fails after collecting required landmarks, restarts selection.
    
    Args:
        image_rgb: RGB image array
        required_labels: List of (name, (court_x, court_y), description) tuples
        optional_labels: List of (name, (court_x, court_y), description) tuples
    
    Returns:
        Tuple of (img_points, court_points) as numpy arrays, both in matching order.
        Returns (None, None) only if user cancels or unrecoverable error occurs.
    """
    img_height, img_width = image_rgb.shape[:2]
    
    # Retry loop: restart selection if validation fails
    while True:
        img_points = []
        court_points = []
        current_click = [None]  # Use list to allow modification in nested function
        click_received = [False]  # Flag to indicate click was received
        
        def on_click(event):
            if event.inaxes is not None and event.button == 1:  # Left mouse button
                if not click_received[0]:  # Only accept first click per landmark
                    u, v = int(event.xdata), int(event.ydata)
                    current_click[0] = (u, v)
                    click_received[0] = True
                    print(f"    Clicked at pixel ({u}, {v})")
                    # Draw temporary marker
                    ax.plot(u, v, 'o', markersize=12, markeredgecolor='yellow', 
                           markeredgewidth=3, markerfacecolor='red', alpha=0.7, zorder=10)
                    ax.set_xlim(0, img_width)  # Re-lock after plotting
                    ax.set_ylim(img_height, 0)
                    plt.draw()
        
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.imshow(image_rgb, extent=[0, img_width, img_height, 0])
        
        # Lock axes to image dimensions (CRITICAL - no autoscaling)
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Inverted: image origin is top-left
        ax.set_aspect('equal')
        ax.autoscale(enable=False)  # Disable autoscaling
        
        ax.axis('off')
        
        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        plt.tight_layout()
        plt.ion()  # Interactive mode
        plt.show()
        plt.pause(0.2)  # Brief pause to ensure window is displayed
        
        # Collect required labels
        print(f"\n{'='*70}")
        print(f"REQUIRED LANDMARKS COLLECTION")
        print(f"{'='*70}")
        
        for i, (label_name, (court_x, court_y), description) in enumerate(required_labels, 1):
            current_click[0] = None
            click_received[0] = False
            
            ax.set_title(f"({chr(96+i)}) Click: {label_name}\n{description}", 
                        fontsize=13, fontweight='bold', color='white')
            plt.draw()
            plt.pause(0.1)
            
            print(f"\nClick <{label_name}> now (expected court: ({court_x}, {court_y}) feet)")
            print(f"  Description: {description}")
            
            # Wait for click (polling loop)
            timeout_counter = 0
            max_timeout = 3000  # 300 seconds timeout (5 minutes)
            while not click_received[0]:
                plt.pause(0.1)
                timeout_counter += 1
                if timeout_counter > max_timeout:
                    plt.close(fig)
                    print(f"\n❌ Timeout waiting for click on {label_name}")
                    return None, None
            
            if current_click[0] is None:
                plt.close(fig)
                print(f"\n❌ No click detected for {label_name}")
                return None, None
            
            u, v = current_click[0]
            img_points.append((u, v))
            court_points.append((court_x, court_y))
            
            # Draw permanent marker with label
            ax.plot(u, v, 'o', markersize=14, markeredgecolor='yellow', 
                   markeredgewidth=3, markerfacecolor='red', alpha=0.8, zorder=10)
            ax.text(u + 25, v, f"({chr(96+i)})\n{label_name}", 
                   color='yellow', fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', 
                            edgecolor='yellow', alpha=0.9, linewidth=2),
                   zorder=11)
            ax.set_xlim(0, img_width)  # Re-lock after plotting
            ax.set_ylim(img_height, 0)
            plt.draw()
            plt.pause(0.1)
            
            print(f"  ✓ Recorded: Image ({u}, {v}) pixels -> Court ({court_x}, {court_y}) feet")
        
        # Validate required landmarks
        if len(img_points) < 4:
            plt.close(fig)
            print(f"\n❌ VALIDATION FAILED: Only {len(img_points)} points collected, need at least 4")
            print(f"Restarting landmark selection...\n")
            continue
        
        is_valid, error_msg = validate_landmark_selection(img_points, court_points)
        if not is_valid:
            plt.close(fig)
            print(f"\n❌ VALIDATION FAILED: {error_msg}")
            print(f"Restarting landmark selection...\n")
            continue
        
        print(f"\n✓ Required landmarks validated: {len(img_points)} unique points")
        
        # Collect optional labels (after required validation passed)
        if optional_labels:
            print(f"\n{'='*70}")
            print(f"OPTIONAL LANDMARKS COLLECTION")
            print(f"{'='*70}")
            
            for idx, (label_name, (court_x, court_y), description) in enumerate(optional_labels):
                i = len(required_labels) + idx + 1  # Index for labeling (e), (f), etc.
                # Check if this court coordinate is already selected
                coord_key = (round(court_x, 2), round(court_y, 2))
                existing_coords = [(round(x, 2), round(y, 2)) for x, y in court_points]
                if coord_key in existing_coords:
                    print(f"\n⚠ Landmark {label_name} has court coordinate ({court_x}, {court_y})")
                    print(f"  This coordinate is already selected. Skipping.")
                    continue
                
                response = input(f"\nPress Enter to skip <{label_name}> or type 'y' then Enter to click it: ").strip().lower()
                
                if response != 'y':
                    print(f"  Skipped {label_name}")
                    continue
                
                current_click[0] = None
                click_received[0] = False
                
                ax.set_title(f"({chr(96+i)}) Click: {label_name}\n{description}", 
                            fontsize=13, fontweight='bold', color='white')
                plt.draw()
                plt.pause(0.1)
                
                print(f"Click <{label_name}> now (expected court: ({court_x}, {court_y}) feet)")
                print(f"  Description: {description}")
                
                # Wait for click (polling loop)
                timeout_counter = 0
                max_timeout = 3000  # 300 seconds timeout
                while not click_received[0]:
                    plt.pause(0.1)
                    timeout_counter += 1
                    if timeout_counter > max_timeout:
                        print(f"  Timeout waiting for click, skipping {label_name}")
                        break
                
                if current_click[0] is None or not click_received[0]:
                    print(f"  No click detected, skipping {label_name}")
                    continue
                
                u, v = current_click[0]
                img_points.append((u, v))
                court_points.append((court_x, court_y))
                
                # Draw permanent marker with label
                ax.plot(u, v, 'o', markersize=14, markeredgecolor='cyan', 
                       markeredgewidth=3, markerfacecolor='blue', alpha=0.8, zorder=10)
                ax.text(u + 25, v, f"({chr(96+i)})\n{label_name}", 
                       color='cyan', fontweight='bold', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='black', 
                                edgecolor='cyan', alpha=0.9, linewidth=2),
                       zorder=11)
                ax.set_xlim(0, img_width)  # Re-lock after plotting
                ax.set_ylim(img_height, 0)
                plt.draw()
                plt.pause(0.1)
                
                print(f"  ✓ Recorded: Image ({u}, {v}) pixels -> Court ({court_x}, {court_y}) feet")
        
        plt.close(fig)
        
        # Final validation (after optional landmarks)
        if len(img_points) < 4:
            print(f"\n❌ VALIDATION FAILED: Only {len(img_points)} points collected, need at least 4")
            print(f"Restarting landmark selection...\n")
            continue
        
        is_valid, error_msg = validate_landmark_selection(img_points, court_points)
        if not is_valid:
            print(f"\n❌ VALIDATION FAILED: {error_msg}")
            print(f"Restarting landmark selection...\n")
            continue
        
        print(f"\n✓ Landmark collection complete: {len(img_points)} landmarks validated")
        
        return np.array(img_points, dtype=np.float32), np.array(court_points, dtype=np.float32)


def compute_homography(img_pts, court_pts, img_width=None, img_height=None):
    """
    Compute homography matrix from image coordinates to court coordinates.
    
    If exactly 4 points: uses cv2.getPerspectiveTransform
    If >4 points: uses cv2.findHomography with RANSAC
    
    Args:
        img_pts: Numpy array of (u, v) pixel coordinates (N x 2)
        court_pts: Numpy array of (x, y) court coordinates (N x 2)
        img_width: Optional image width for validation (if None, uses max u coordinate)
        img_height: Optional image height for validation (if None, uses max v coordinate)
    
    Returns:
        Tuple of (H, mask) where H is 3x3 homography matrix, mask is inlier mask (None for 4-point case)
    """
    if len(img_pts) < 4:
        raise ValueError(f"Need at least 4 correspondences, got {len(img_pts)}")
    
    # Reshape for OpenCV (N x 1 x 2)
    src_points = img_pts.reshape(-1, 1, 2)
    dst_points = court_pts.reshape(-1, 1, 2)
    
    print(f"\nComputing homography with {len(img_pts)} correspondences...")
    
    # If exactly 4 points: use getPerspectiveTransform
    if len(img_pts) == 4:
        print(f"  Using cv2.getPerspectiveTransform (exact 4-point solution)")
        H = cv2.getPerspectiveTransform(src_points, dst_points)
        mask = None  # No mask for direct transform
    else:
        # If >4 points: use findHomography with RANSAC
        print(f"  Using cv2.findHomography with RANSAC (robust estimation)")
        H, mask = cv2.findHomography(src_points, dst_points,
                                     method=cv2.RANSAC,
                                     ransacReprojThreshold=5.0)
        
        if H is None:
            raise ValueError("Failed to compute homography matrix")
        
        inliers = np.sum(mask) if mask is not None else len(img_pts)
        print(f"  Inliers: {inliers}/{len(img_pts)} points used in homography")
        
        if inliers < 4:
            raise ValueError(f"Only {inliers} inliers, need at least 4. Check landmark selections.")
    
    # Validate H is not NaN/Inf
    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        raise ValueError("Homography matrix contains NaN or Inf values - degenerate configuration")
    
    # Validate it does not collapse test points: transform 4 corners of the image and check spread
    if img_width is None:
        img_width = int(np.max(img_pts[:, 0])) + 1
    if img_height is None:
        img_height = int(np.max(img_pts[:, 1])) + 1
    
    # Four corners of the image: (0,0), (width,0), (width,height), (0,height)
    image_corners = np.array([
        [0, 0],
        [img_width, 0],
        [img_width, img_height],
        [0, img_height]
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    transformed_corners = cv2.perspectiveTransform(image_corners, H)
    transformed_2d = transformed_corners.reshape(-1, 2)
    
    # Check spread: calculate ranges in x and y
    x_range = np.max(transformed_2d[:, 0]) - np.min(transformed_2d[:, 0])
    y_range = np.max(transformed_2d[:, 1]) - np.min(transformed_2d[:, 1])
    
    print(f"  Image corners transform to court coordinates:")
    print(f"    Top-left:    ({transformed_2d[0, 0]:.2f}, {transformed_2d[0, 1]:.2f})")
    print(f"    Top-right:   ({transformed_2d[1, 0]:.2f}, {transformed_2d[1, 1]:.2f})")
    print(f"    Bottom-right: ({transformed_2d[2, 0]:.2f}, {transformed_2d[2, 1]:.2f})")
    print(f"    Bottom-left:  ({transformed_2d[3, 0]:.2f}, {transformed_2d[3, 1]:.2f})")
    print(f"  Spread: X range = {x_range:.2f} feet, Y range = {y_range:.2f} feet")
    
    # Check if transformed corners collapse (all very close together)
    if x_range < 1.0 and y_range < 1.0:
        raise ValueError(
            f"Homography is degenerate: image corners collapse to nearly same location "
            f"(X range: {x_range:.4f} ft, Y range: {y_range:.4f} ft). "
            f"Check landmark selections."
        )
    
    # Check if spread is reasonable (court is 94x50 feet, so corners should span significant portion)
    if x_range < 10.0 or y_range < 10.0:
        print(f"  ⚠ Warning: Small spread in transformed corners (X: {x_range:.2f} ft, Y: {y_range:.2f} ft)")
        print(f"    Expected larger spread for full court view. Check landmark selections.")
    
    print(f"  ✓ Homography computed and validated successfully")
    
    return H, mask


def collect_players(image_rgb):
    """
    Collect player feet clicks from user with offense/defense designation.
    
    Uses the same locked extent / no autoscale approach as collect_landmarks.
    Each left click appends a point and prompts for offense/defense.
    After collecting all players, allows selection of ball carrier.
    Close window after collecting all desired players.
    
    Args:
        image_rgb: RGB image array
    
    Returns:
        Tuple of (player_positions, player_teams, ball_carrier_index)
        - player_positions: List of (u, v) pixel coordinate tuples
        - player_teams: List of 'offense' or 'defense' strings (same length as player_positions)
        - ball_carrier_index: Index (0-based) of player with ball, or None if not selected
    """
    player_positions = []
    player_teams = []  # 'offense' or 'defense'
    current_click = [None]  # Use list to allow modification in nested function
    click_received = [False]  # Flag to indicate click was received
    collection_active = [True]  # Flag to control when to stop collecting
    img_height, img_width = image_rgb.shape[:2]
    
    # Colors for offense (green/yellow) and defense (red/orange)
    offense_color = {'marker': 'go', 'edge': 'yellow', 'face': 'green', 'text': 'yellow'}
    defense_color = {'marker': 'ro', 'edge': 'orange', 'face': 'red', 'text': 'orange'}
    
    def on_click(event):
        if event.inaxes is not None and event.button == 1 and collection_active[0]:  # Left mouse button
            if not click_received[0]:  # Only accept first click per player
                u, v = int(event.xdata), int(event.ydata)
                current_click[0] = (u, v)
                click_received[0] = True
    
    def on_close(event):
        """Handle window close event."""
        collection_active[0] = False
        print(f"\n  Window closed. Finishing player collection...")
    
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.imshow(image_rgb, extent=[0, img_width, img_height, 0])
    
    # Lock axes to image dimensions (same approach as collect_landmarks)
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # Inverted: image origin is top-left
    ax.set_aspect('equal')
    ax.autoscale(enable=False)  # Disable autoscaling
    
    ax.set_title("Click on each player's FEET (bottom center) in the image.\nAfter each click, you'll be asked if it's offense or defense.", 
                fontsize=14, fontweight='bold', color='white', pad=20)
    ax.axis('off')
    
    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('close_event', on_close)
    
    plt.tight_layout()
    plt.ion()  # Interactive mode
    plt.show()
    plt.pause(0.2)  # Brief pause to ensure window is displayed
    
    print(f"\n{'='*70}")
    print(f"PLAYER COLLECTION")
    print(f"{'='*70}")
    print(f"\nClick on each player's FEET (bottom center) in the image.")
    print(f"After each click, you'll be asked if the player is OFFENSE or DEFENSE.")
    print(f"Close the window when you've clicked all players (up to 10 players).\n")
    
    player_num = 0
    max_players = 10
    
    # Collect players one by one
    while collection_active[0] and player_num < max_players:
        current_click[0] = None
        click_received[0] = False
        
        # Update title
        ax.set_title(f"Click player {player_num + 1} FEET (bottom center)\nAfter click, you'll select offense/defense", 
                    fontsize=14, fontweight='bold', color='white', pad=20)
        plt.draw()
        plt.pause(0.1)
        
        print(f"\nPlayer {player_num + 1}:")
        print(f"  Click on the player's FEET (bottom center) in the image...")
        
        # Wait for click
        timeout_counter = 0
        max_timeout = 3000  # 5 minutes timeout
        while not click_received[0] and collection_active[0]:
            plt.pause(0.1)
            timeout_counter += 1
            if timeout_counter > max_timeout:
                print(f"  Timeout waiting for click")
                collection_active[0] = False
                break
        
        if not collection_active[0] or current_click[0] is None:
            break  # Window closed or timeout
        
        u, v = current_click[0]
        player_positions.append((u, v))
        
        # Ask for offense/defense
        while True:
            team = input(f"  Is player {player_num + 1} OFFENSE or DEFENSE? (o/d): ").strip().lower()
            if team in ['o', 'offense']:
                team_str = 'offense'
                break
            elif team in ['d', 'defense']:
                team_str = 'defense'
                break
            else:
                print(f"    Invalid input. Please enter 'o' for offense or 'd' for defense.")
        
        player_teams.append(team_str)
        
        # Draw marker with color based on team
        if team_str == 'offense':
            color = offense_color
        else:
            color = defense_color
        
        ax.plot(u, v, color['marker'], markersize=12, markeredgecolor=color['edge'], 
               markeredgewidth=3, markerfacecolor=color['face'], alpha=0.8, zorder=10)
        
        # Label: O1, O2, D1, D2, etc.
        if team_str == 'offense':
            label = f"O{len([t for t in player_teams if t == 'offense'])}"
        else:
            label = f"D{len([t for t in player_teams if t == 'defense'])}"
        
        ax.text(u + 20, v, label, 
               color=color['text'], fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', 
                        edgecolor=color['edge'], alpha=0.9, linewidth=2),
               zorder=11)
        
        # Re-lock axes after plotting (no autoscaling)
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)
        plt.draw()
        plt.pause(0.1)
        
        print(f"  ✓ Recorded: Player {player_num + 1} ({team_str.upper()}) at pixel ({u}, {v})")
        player_num += 1
        
        # Ask if user wants to continue (if not at max)
        if player_num < max_players:
            continue_choice = input(f"\n  Add another player? (y/n, or close window to finish): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                collection_active[0] = False
                break
    
    # Window was closed or user finished, close figure
    plt.close(fig)
    
    if not player_positions:
        print(f"\n⚠ No players collected. Exiting.")
        return [], [], None
    
    print(f"\n✓ Player collection complete: {len(player_positions)} players recorded")
    offense_count = sum(1 for t in player_teams if t == 'offense')
    defense_count = sum(1 for t in player_teams if t == 'defense')
    print(f"  Offense: {offense_count} players")
    print(f"  Defense: {defense_count} players")
    
    # Now ask who has the ball (only if we have players)
    ball_carrier_index = None
    if len(player_positions) > 0:
        print(f"\n{'='*70}")
        print(f"BALL CARRIER SELECTION")
        print(f"{'='*70}")
        print(f"\nAll players collected. Who has the ball?\n")
        
        # List all players
        for i, (pos, team) in enumerate(zip(player_positions, player_teams)):
            u, v = pos
            if team == 'offense':
                label = f"O{len([t for t in player_teams[:i+1] if t == 'offense'])}"
            else:
                label = f"D{len([t for t in player_teams[:i+1] if t == 'defense'])}"
            print(f"  {i+1}. Player {i+1} ({team.upper()}, {label}) - Pixel ({u}, {v})")
        
        print(f"  {len(player_positions)+1}. No one has the ball / Skip")
        
        while True:
            try:
                choice = input(f"\nEnter player number (1-{len(player_positions)+1}): ").strip()
                choice_num = int(choice)
                if 1 <= choice_num <= len(player_positions):
                    ball_carrier_index = choice_num - 1
                    team_str = player_teams[ball_carrier_index]
                    if team_str == 'offense':
                        label = f"O{len([t for t in player_teams[:ball_carrier_index+1] if t == 'offense'])}"
                    else:
                        label = f"D{len([t for t in player_teams[:ball_carrier_index+1] if t == 'defense'])}"
                    print(f"  ✓ Ball carrier: Player {choice_num} ({team_str.upper()}, {label})")
                    break
                elif choice_num == len(player_positions) + 1:
                    print(f"  ✓ No ball carrier selected")
                    ball_carrier_index = None
                    break
                else:
                    print(f"    Invalid choice. Please enter a number between 1 and {len(player_positions)+1}.")
            except ValueError:
                print(f"    Invalid input. Please enter a number.")
    
    return player_positions, player_teams, ball_carrier_index


def map_points(player_pixels, H, player_teams=None):
    """
    Map player pixel positions to court coordinates using homography.
    
    Uses cv2.perspectiveTransform to transform points, then clamps coordinates
    to court bounds [0,94] for x and [0,50] for y. Prints unclamped values for debugging.
    Shows team (OFF/DEF) in the output table.
    
    Args:
        player_pixels: List of (u, v) pixel coordinate tuples
        H: 3x3 homography matrix
        player_teams: Optional list of 'offense' or 'defense' strings for labeling
    
    Returns:
        Tuple of (court_positions_clamped, court_positions_unclamped) where each is a list of (x, y) tuples
    """
    if not player_pixels:
        return [], []
    
    # Convert to numpy array and reshape for OpenCV (N x 1 x 2)
    points = np.array(player_pixels, dtype=np.float32).reshape(-1, 1, 2)
    
    # Transform using homography
    transformed = cv2.perspectiveTransform(points, H)
    
    # Extract 2D coordinates (unclamped and clamped)
    court_positions_clamped = []
    court_positions_unclamped = []
    
    # Track offense/defense counts for labeling (O1, O2, D1, D2, etc.)
    offense_count = 0
    defense_count = 0
    
    print(f"\nMapping {len(player_pixels)} player points to court coordinates:")
    print(f"  {'Player':<8} {'Team':<8} {'Pixel (u,v)':<20} {'Unclamped Court (x,y)':<25} {'Clamped Court (x,y)':<25}")
    print(f"  {'-'*95}")
    
    for i, (pixel, transformed_pt) in enumerate(zip(player_pixels, transformed)):
        u, v = pixel
        x_unclamped = float(transformed_pt[0][0])
        y_unclamped = float(transformed_pt[0][1])
        
        # Store unclamped values
        court_positions_unclamped.append((x_unclamped, y_unclamped))
        
        # Clamp x to [0, 94] and y to [0, 50]
        x_clamped = float(np.clip(x_unclamped, 0, 94))
        y_clamped = float(np.clip(y_unclamped, 0, 50))
        
        court_positions_clamped.append((x_clamped, y_clamped))
        
        # Determine label based on team
        if player_teams and i < len(player_teams):
            team = player_teams[i]
            if team == 'offense':
                offense_count += 1
                label = f"O{offense_count}"
                team_str = "OFF"
            elif team == 'defense':
                defense_count += 1
                label = f"D{defense_count}"
                team_str = "DEF"
            else:
                label = f"P{i+1}"
                team_str = "?"
        else:
            label = f"P{i+1}"
            team_str = "?"
        
        # Print unclamped vs clamped for debugging
        clamped_marker = ""
        if x_unclamped != x_clamped or y_unclamped != y_clamped:
            clamped_marker = " [CLAMPED]"
            if abs(x_unclamped) > 200 or abs(y_unclamped) > 200:
                clamped_marker += " ⚠ CALIBRATION BROKEN!"
        
        print(f"  {label:<7} {team_str:<8} ({u:4.0f},{v:4.0f})       "
              f"({x_unclamped:7.2f}, {y_unclamped:6.2f}){clamped_marker:<20}  "
              f"({x_clamped:7.2f}, {y_clamped:6.2f})")
    
    return court_positions_clamped, court_positions_unclamped


def print_results_table(player_pixels, player_court):
    """
    Print formatted table of player positions.
    
    Args:
        player_pixels: List of (u, v) pixel coordinate tuples
        player_court: List of (x, y) court coordinate tuples
    """
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Player':<8} {'Pixel (u, v)':<20} {'Court (x, y) feet':<25}")
    print("-" * 70)
    
    for i, (px, py), (cx, cy) in enumerate(zip(player_pixels, player_court), 1):
        print(f"{i:<8} ({px:4.0f}, {py:4.0f}){'':<8} ({cx:6.2f}, {cy:6.2f})")


def create_visualization(image_path, player_pixels, player_teams, ball_carrier_index,
                         player_court_clamped, landmark_img_pts, landmark_court_pts, required_labels, optional_labels):
    """
    Create side-by-side visualization: original image with clicks and top-down court view.
    
    Uses basketball_court_coordinates.py to draw the court with create_grid() and draw_court().
    Colors players by offense (green/yellow) and defense (red/orange).
    Shows ball carrier with ball emoji (⚽).
    
    Args:
        image_path: Path to original image
        player_pixels: List of (u, v) pixel coordinate tuples
        player_teams: List of 'offense' or 'defense' strings (same length as player_pixels)
        ball_carrier_index: Index (0-based) of player with ball, or None
        player_court_clamped: List of (x, y) clamped court coordinate tuples
        landmark_img_pts: Image points for landmarks (numpy array)
        landmark_court_pts: Court points for landmarks (numpy array)
        required_labels: List of (name, (x, y), description) tuples for required landmarks
        optional_labels: List of (name, (x, y), description) tuples for optional landmarks
    
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor('black')
    
    # Left subplot: Original image
    ax1 = fig.add_subplot(1, 2, 1)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width = image_rgb.shape[:2]
    ax1.imshow(image_rgb, extent=[0, img_width, img_height, 0])
    ax1.set_xlim(0, img_width)
    ax1.set_ylim(img_height, 0)
    ax1.set_aspect('equal')
    ax1.set_title("Original Image with Clicks", fontsize=14, fontweight='bold', color='white')
    ax1.axis('off')
    
    # Get landmark names (combine required and optional in order)
    all_landmark_names = [label[0] for label in required_labels]
    # Only add optional labels that were actually collected
    if landmark_img_pts is not None and len(landmark_img_pts) > len(required_labels):
        for i in range(len(required_labels), len(landmark_img_pts)):
            if i - len(required_labels) < len(optional_labels):
                all_landmark_names.append(optional_labels[i - len(required_labels)][0])
    
    # Draw landmark points (cyan)
    if landmark_img_pts is not None and len(landmark_img_pts) > 0:
        for i, (u, v) in enumerate(landmark_img_pts):
            label_name = all_landmark_names[i] if i < len(all_landmark_names) else f"L{i+1}"
            ax1.plot(u, v, 'o', markersize=10, markeredgecolor='cyan', 
                    markeredgewidth=2, markerfacecolor='blue', alpha=0.7, zorder=10)
            ax1.text(u + 15, v, label_name[:8], color='cyan', fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Draw player points with offense/defense colors
    # Color scheme: offense = green/yellow, defense = red/orange
    offense_color = {'marker': 'go', 'edge': 'yellow', 'face': 'green', 'text': 'yellow'}
    defense_color = {'marker': 'ro', 'edge': 'orange', 'face': 'red', 'text': 'orange'}
    
    # Track offense/defense counts for labeling (O1, O2, D1, D2, etc.)
    offense_count = 0
    defense_count = 0
    
    if player_teams is None:
        player_teams = ['unknown'] * len(player_pixels)
    
    for i, (u, v) in enumerate(player_pixels):
        # Determine team and color
        if i < len(player_teams):
            team = player_teams[i]
        else:
            team = 'unknown'
        
        if team == 'offense':
            color = offense_color
            offense_count += 1
            label = f"O{offense_count}"
        elif team == 'defense':
            color = defense_color
            defense_count += 1
            label = f"D{defense_count}"
        else:
            color = {'marker': 'ro', 'edge': 'yellow', 'face': 'gray', 'text': 'yellow'}
            label = f"P{i+1}"
        
        # Draw marker with appropriate color
        ax1.plot(u, v, color['marker'], markersize=12, markeredgecolor=color['edge'], 
                markeredgewidth=2, markerfacecolor=color['face'], alpha=0.8, zorder=10)
        
        # Add ball indicator if this player has the ball
        ball_indicator = " ⚽" if ball_carrier_index == i else ""
        ax1.text(u + 15, v, f"{label}{ball_indicator}", 
               color=color['text'], fontweight='bold', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
               zorder=11)
    
    # Right subplot: Top-down court view using basketball_court_coordinates.py
    ax2 = fig.add_subplot(1, 2, 2)
    dims = COURT_DIMENSIONS
    
    # Use create_grid() approach: set up court background and rectangle manually
    ax2.set_facecolor('#1a472a')  # Dark green court background
    ax2.set_xlim(-2, dims['length'] + 2)
    ax2.set_ylim(-2, dims['width'] + 2)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X Coordinate (feet from left baseline)', 
                  fontsize=12, color='white', fontweight='bold')
    ax2.set_ylabel('Y Coordinate (feet from bottom sideline)', 
                  fontsize=12, color='white', fontweight='bold')
    ax2.set_title("Mapped Players on Court", fontsize=14, color='white', fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.grid(False)
    
    # Draw rectangle outline (court boundary) - this matches create_grid behavior
    court_rect = patches.Rectangle((0, 0), dims['length'], dims['width'],
                                   linewidth=3, edgecolor='white', facecolor='none')
    ax2.add_patch(court_rect)
    
    # Draw full court markings using draw_court() - this calls draw_center, draw_hoop_and_backboard, etc.
    # Pass show_labels=False and show_players=False since we're plotting our own
    draw_court(ax2, show_labels=False, show_players=False)
    
    # Draw landmark points on court (cyan)
    if landmark_court_pts is not None and len(landmark_court_pts) > 0:
        for i, (x, y) in enumerate(landmark_court_pts):
            label_name = all_landmark_names[i] if i < len(all_landmark_names) else f"L{i+1}"
            ax2.plot(x, y, 'o', markersize=8, markeredgecolor='cyan', 
                    markeredgewidth=2, markerfacecolor='blue', alpha=0.7, zorder=9)
            ax2.text(x + 1, y + 1, label_name[:8], color='cyan', fontweight='bold', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    # Draw mapped players with offense/defense colors and ball carrier
    # Color scheme: offense = green/yellow, defense = red/orange
    offense_color = {'marker': 'go', 'edge': 'yellow', 'face': 'green', 'text': 'yellow'}
    defense_color = {'marker': 'ro', 'edge': 'orange', 'face': 'red', 'text': 'orange'}
    
    # Track offense/defense counts for labeling (O1, O2, D1, D2, etc.)
    offense_count = 0
    defense_count = 0
    
    if player_teams is None:
        player_teams = ['unknown'] * len(player_court_clamped)
    
    for i, (x, y) in enumerate(player_court_clamped):
        # Determine team and color
        if i < len(player_teams):
            team = player_teams[i]
        else:
            team = 'unknown'
        
        if team == 'offense':
            color = offense_color
            offense_count += 1
            label = f"O{offense_count}"
        elif team == 'defense':
            color = defense_color
            defense_count += 1
            label = f"D{defense_count}"
        else:
            color = {'marker': 'ro', 'edge': 'yellow', 'face': 'gray', 'text': 'yellow'}
            label = f"P{i+1}"
        
        # Draw marker with appropriate color
        ax2.plot(x, y, color['marker'], markersize=14, markeredgecolor=color['edge'], 
                markeredgewidth=3, markerfacecolor=color['face'], alpha=0.8, zorder=10)
        
        # Add ball indicator if this player has the ball
        ball_indicator = " ⚽" if ball_carrier_index == i else ""
        ax2.text(x + 1.5, y + 1.5, f"{label}{ball_indicator}", 
               color=color['text'], fontweight='bold', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7), zorder=11)
    
    # Add legend for offense/defense
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markeredgecolor='yellow', markersize=12, label='Offense', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markeredgecolor='orange', markersize=12, label='Defense', markeredgewidth=2),
    ]
    if ball_carrier_index is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markersize=0, label='⚽ = Ball Carrier', markeredgewidth=0))
    ax2.legend(handles=legend_elements, loc='upper right', facecolor='black', 
               edgecolor='white', labelcolor='white', fontsize=10)
    
    plt.tight_layout()
    
    return fig


def save_json_output(image_path, landmark_img_pts, landmark_court_pts, required_labels, optional_labels,
                     player_pixels, player_teams, ball_carrier_index,
                     player_court_clamped, player_court_unclamped, output_path='mapped_players.json'):
    """
    Save landmarks and players data to JSON file.
    
    Args:
        image_path: Path to original image
        landmark_img_pts: Image points for landmarks (numpy array)
        landmark_court_pts: Court points for landmarks (numpy array)
        required_labels: List of (name, (x, y), description) tuples for required landmarks
        optional_labels: List of (name, (x, y), description) tuples for optional landmarks
        player_pixels: List of (u, v) pixel coordinate tuples
        player_teams: List of 'offense' or 'defense' strings (same length as player_pixels)
        ball_carrier_index: Index (0-based) of player with ball, or None
        player_court_clamped: List of (x, y) clamped court coordinate tuples
        player_court_unclamped: List of (x, y) unclamped court coordinate tuples
        output_path: Path to save JSON file
    """
    # Build landmarks dictionary - match collected landmarks with their labels
    landmarks_dict = {}
    
    # Required landmarks come first, then optional ones that were collected
    landmark_index = 0
    
    # Add required landmarks (all should be collected)
    for label_name, (court_x, court_y), description in required_labels:
        if landmark_index < len(landmark_img_pts):
            landmarks_dict[label_name] = {
                "pixel": [float(landmark_img_pts[landmark_index][0]), float(landmark_img_pts[landmark_index][1])],
                "court": [float(landmark_court_pts[landmark_index][0]), float(landmark_court_pts[landmark_index][1])]
            }
            landmark_index += 1
    
    # Add optional landmarks that were collected
    # Note: optional landmarks might not all be collected, so we check if we have enough points
    for label_name, (court_x, court_y), description in optional_labels:
        if landmark_index < len(landmark_img_pts):
            # Check if this court coordinate matches (to verify it was collected)
            court_pt = landmark_court_pts[landmark_index]
            coord_key_expected = (round(court_x, 2), round(court_y, 2))
            coord_key_actual = (round(court_pt[0], 2), round(court_pt[1], 2))
            
            if coord_key_expected == coord_key_actual:
                landmarks_dict[label_name] = {
                    "pixel": [float(landmark_img_pts[landmark_index][0]), float(landmark_img_pts[landmark_index][1])],
                    "court": [float(court_pt[0]), float(court_pt[1])]
                }
                landmark_index += 1
    
    # Build players list (includes team and has_ball flag)
    players_list = []
    if player_teams is None:
        player_teams = ['unknown'] * len(player_pixels)
    
    for i, (pixel, unclamped, clamped) in enumerate(zip(player_pixels, player_court_unclamped, player_court_clamped)):
        player_data = {
            "pixel": [float(pixel[0]), float(pixel[1])],
            "team": player_teams[i] if i < len(player_teams) else "unknown",
            "court_unclamped": [float(unclamped[0]), float(unclamped[1])],
            "court_clamped": [float(clamped[0]), float(clamped[1])]
        }
        # Add has_ball flag if this player has the ball
        if ball_carrier_index is not None and i == ball_carrier_index:
            player_data["has_ball"] = True
        else:
            player_data["has_ball"] = False
        
        players_list.append(player_data)
    
    # Create output dictionary (do not include image_path in JSON as per user's spec)
    output_data = {
        "landmarks": landmarks_dict,
        "players": players_list
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ JSON data saved to: {output_path}")


def main(image_path=None, output_dir=None):
    """
    Main function.
    
    Args:
        image_path: Path to input image. If None, uses default path.
        output_dir: Directory to save outputs. If None, uses 'output' directory.
                   Output filenames will be based on input image filename.
    """
    print("=" * 70)
    print("BASKETBALL PLAYER MAPPING TOOL")
    print("=" * 70)
    
    # Set default image path if not provided
    if image_path is None:
        image_path = "data/images/Screenshot 2026-01-10 122231.png"
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = "output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames based on input image filename
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    output_image_path = os.path.join(output_dir, f"{image_basename}_mapped_players.png")
    output_json_path = os.path.join(output_dir, f"{image_basename}_mapped_players.json")
    
    # Ask for visible half
    while True:
        visible_side = input("\nWhich half is visible in the image? (left/right): ").strip().lower()
        if visible_side in ["left", "right"]:
            break
        print("Invalid input. Please enter 'left' or 'right'.")
    
    # Ask for preset selection
    print(f"\n{'='*70}")
    print(f"PRESET SELECTION ({visible_side.upper()} HALF)")
    print(f"{'='*70}")
    print(f"\nChoose a preset based on what's visible in your image:")
    print(f"\n  1. 'lane_and_ft' - BEST FOR MOST SCREENSHOTS")
    print(f"     Use when: Baseline corners are NOT visible, but lane/paint area IS visible")
    print(f"     Required: Lane corners at baseline + Free throw line (bottom end + center)")
    print(f"     Works well when you can see the lane/paint area and free throw line")
    print(f"\n  2. 'baseline' - Only if baseline corners are clearly visible")
    print(f"     Use when: Baseline corners ARE clearly visible in the image")
    print(f"     Required: Baseline corners + Free throw line ends")
    print(f"     Works well when you can see the full baseline with both corners")
    
    while True:
        preset = input(f"\nSelect preset (lane_and_ft/baseline): ").strip().lower()
        if preset in ["lane_and_ft", "baseline"]:
            break
        print("Invalid input. Please enter 'lane_and_ft' or 'baseline'.")
    
    # Get required and optional points for selected preset
    required_pts, optional_pts = get_required_points(visible_side, preset)
    
    # Show what points will be collected with clear instructions
    print(f"\n{'='*70}")
    print(f"PRESET: {preset.upper()} - {visible_side.upper()} HALF")
    print(f"{'='*70}")
    print(f"\nREQUIRED landmarks (4 unique points - click in this exact order):")
    print(f"  You will click these points one by one. Follow the instructions carefully.\n")
    
    for i, (name, (x, y), desc) in enumerate(required_pts, 1):
        print(f"  ({chr(96+i)}) {name.upper()}")
        print(f"      Expected court coordinate: ({x:6.2f}, {y:6.2f}) feet")
        print(f"      INSTRUCTION: {desc}")
        print()
    
    if optional_pts:
        print(f"OPTIONAL landmarks (2 points - you can skip these by pressing Enter):")
        print(f"  You will be asked if you want to click each optional point.\n")
        for i, (name, (x, y), desc) in enumerate(optional_pts, 5):
            print(f"  ({chr(96+i)}) {name.upper()}")
            print(f"      Expected court coordinate: ({x:6.2f}, {y:6.2f}) feet")
            print(f"      INSTRUCTION: {desc}")
            print()
    
    input("\nPress Enter to start landmark collection...")
    
    # Load image
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get landmark points for the selected preset
    required_landmarks, optional_landmarks = get_required_points(visible_side, preset)
    
    # Collect landmarks (with retry on validation failure)
    while True:
        landmark_img_pts, landmark_court_pts = collect_landmarks(image_rgb, required_landmarks, optional_landmarks)
        
        if landmark_img_pts is None or landmark_court_pts is None:
            # Validation failed - restart
            print(f"\n{'='*70}")
            print(f"RESTARTING LANDMARK COLLECTION")
            print(f"{'='*70}")
            retry = input("\nRetry landmark collection? (y/n): ").strip().lower()
            if retry != 'y':
                raise ValueError("Landmark collection cancelled by user")
            print("\nRestarting...\n")
            continue
        
        # Success
        break
    
    # Compute homography (pass image dimensions for validation)
    img_height, img_width = image_rgb.shape[:2]
    try:
        H, mask = compute_homography(landmark_img_pts, landmark_court_pts, img_width, img_height)
    except ValueError as e:
        raise ValueError(f"Failed to compute homography: {e}")
    
    # Collect player clicks (with offense/defense and ball carrier selection)
    player_pixel_positions, player_teams, ball_carrier_index = collect_players(image_rgb)
    
    if not player_pixel_positions:
        print("\nNo players selected. Exiting.")
        return
    
    # Map player positions to court coordinates (returns both clamped and unclamped)
    player_court_clamped, player_court_unclamped = map_points(player_pixel_positions, H, player_teams)
    
    # Create and save visualization (with offense/defense colors and ball carrier)
    print(f"\nCreating visualization...")
    fig = create_visualization(image_path, player_pixel_positions, player_teams, ball_carrier_index,
                              player_court_clamped, landmark_img_pts, landmark_court_pts, 
                              required_landmarks, optional_landmarks)
    
    # Save visualization
    fig.savefig(output_image_path, dpi=300, facecolor='black', bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_image_path}")
    
    # Save JSON output (includes teams and ball carrier)
    print(f"\nSaving JSON data...")
    save_json_output(image_path, landmark_img_pts, landmark_court_pts, 
                     required_landmarks, optional_landmarks,
                     player_pixel_positions, player_teams, ball_carrier_index,
                     player_court_clamped, player_court_unclamped,
                     output_path=output_json_path)
    
    # Show visualization
    plt.show()


if __name__ == "__main__":
    main()
