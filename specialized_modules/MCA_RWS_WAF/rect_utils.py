# rect_utils.py
"""
Rectangle utilities for genetic algorithm domain optimization.

This module provides utilities for representing and manipulating
rectangular geographic domains used in the genetic algorithm
optimization of MCA RWS/WAF analysis regions.
"""
import random
import logging
import sys
from pathlib import Path
from typing import NamedTuple, Sequence, Tuple, Optional

# Import and set seeds for reproducibility
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.seed_config import set_all_seeds, GLOBAL_SEED
set_all_seeds(GLOBAL_SEED)

class Rect(NamedTuple):
    """
    Represents an axis-aligned rectangular geographical area.
    Coordinates are typically in degrees latitude and longitude.
    """
    lat_min: int
    lat_max: int
    lon_min: int
    lon_max: int

# -------------------------------------------------------------------
#  Constants for geometry rules
# -------------------------------------------------------------------
# Define the overall search space boundaries for latitude and longitude.
# These should be kept in sync with any global constants in the main workflow script.
LAT_RANGE = (-42, -10)
LON_RANGE = (-45, -20)

# Define the minimum allowable side length for a rectangle (in degrees).
MIN_DEG_LAT = 17    # Minimum latitude span
MIN_DEG_LON = 12    # Minimum longitude span

# -------------------------------------------------------------------
#  Conversion helpers
# -------------------------------------------------------------------
def rect_to_vertices(r: Rect) -> Sequence[Tuple[int, int]]:
    """
    Convert a Rect object to a sequence of (lon, lat) vertices.
    The order is North-West, North-East, South-East, South-West,
    which is a common convention for defining polygons for GIS or plotting.

    Args:
        r: The Rect object to convert.

    Returns:
        A list of four (longitude, latitude) tuples representing the corners.
    """
    return [
        (r.lon_min, r.lat_max),  # North-West (NW)
        (r.lon_max, r.lat_max),  # North-East (NE)
        (r.lon_max, r.lat_min),  # South-East (SE)
        (r.lon_min, r.lat_min)   # South-West (SW)
    ]

def vertices_to_rect(verts: Sequence[Tuple[int, int]]) -> Rect:
    """
    Convert a sequence of four (lon, lat) vertices back to a Rect object.
    This function is primarily intended for debugging or compatibility with
    systems that use vertex lists. It assumes the vertices define an
    axis-aligned rectangle.

    Args:
        verts: A sequence of four (longitude, latitude) tuples.

    Returns:
        A Rect object derived from the vertices.

    Raises:
        ValueError: If the provided vertices do not form a valid
                    axis-aligned rectangle (e.g., not 4 vertices,
                    or more than 2 unique lat/lon values).
    """
    if len(verts) != 4:
        raise ValueError(f"Expected 4 vertices, got {len(verts)}")

    lons = sorted(list(set(v[0] for v in verts))) # Unique longitudes, sorted
    lats = sorted(list(set(v[1] for v in verts))) # Unique latitudes, sorted

    if len(lons) != 2 or len(lats) != 2:
        raise ValueError("Vertices do not form an axis-aligned rectangle. "
                         f"Unique lons: {lons}, Unique lats: {lats}")

    lon_min, lon_max = lons[0], lons[1]
    lat_min, lat_max = lats[0], lats[1]

    return Rect(lat_min, lat_max, lon_min, lon_max)

# -------------------------------------------------------------------
#  Geometry rules
# -------------------------------------------------------------------
def is_valid_rect(r: Rect) -> bool:
    """
    Check if a Rect object meets all defined geographical constraints.
    These constraints include:
    1. Being within the global LAT_RANGE and LON_RANGE.
    2. Having lat_min < lat_max and lon_min < lon_max.
    3. Meeting the minimum side length requirements (MIN_DEG_LAT, MIN_DEG_LON).

    Args:
        r: The Rect object to validate.

    Returns:
        True if the rectangle is valid, False otherwise.
    """
    # Check if coordinates are ordered correctly
    if not (r.lat_min < r.lat_max and r.lon_min < r.lon_max):
        return False

    # Check if the rectangle is within the global boundaries
    in_bounds = (LAT_RANGE[0] <= r.lat_min and r.lat_max <= LAT_RANGE[1] and
                 LON_RANGE[0] <= r.lon_min and r.lon_max <= LON_RANGE[1])

    # Check if the rectangle meets minimum size requirements
    big_enough = ((r.lat_max - r.lat_min) >= MIN_DEG_LAT and
                  (r.lon_max - r.lon_min) >= MIN_DEG_LON)

    return in_bounds and big_enough

# -------------------------------------------------------------------
#  Random generation / Mutation / Crossover for Genetic Algorithm
# -------------------------------------------------------------------
def random_rect() -> Rect:
    """
    Generate a new, valid Rect object with random coordinates.
    The coordinates are chosen to respect LAT_RANGE, LON_RANGE, MIN_DEG_LAT, and MIN_DEG_LON.
    Includes fallback logic if many attempts fail to produce a valid rectangle.

    Returns:
        A randomly generated, valid Rect object.
    """
    max_attempts = 150  # Number of attempts to generate a valid random rectangle
    for _ in range(max_attempts):
        # Ensure lat1 is chosen such that lat1 + MIN_DEG_LAT can be <= LAT_RANGE[1]
        lat1 = random.randint(LAT_RANGE[0], LAT_RANGE[1] - MIN_DEG_LAT)
        # Ensure lat2 is chosen such that it's >= lat1 + MIN_DEG_LAT and <= LAT_RANGE[1]
        lat2 = random.randint(lat1 + MIN_DEG_LAT, LAT_RANGE[1])

        # Similar logic for longitudes
        lon1 = random.randint(LON_RANGE[0], LON_RANGE[1] - MIN_DEG_LON)
        lon2 = random.randint(lon1 + MIN_DEG_LON, LON_RANGE[1])

        # Create the candidate rectangle
        # Ensure correct ordering (min, max) although randint logic should handle it
        r = Rect(min(lat1, lat2), max(lat1, lat2), min(lon1, lon2), max(lon1, lon2))

        if is_valid_rect(r):
            return r

    # Fallback strategy: create a minimal valid rectangle in a corner of the domain
    logging.warning(f"Could not generate a valid random rectangle after {max_attempts} attempts. Using fallback.")
    fallback_lat_min = LAT_RANGE[0]
    fallback_lat_max = LAT_RANGE[0] + MIN_DEG_LAT
    fallback_lon_min = LON_RANGE[0]
    fallback_lon_max = LON_RANGE[0] + MIN_DEG_LON

    # Ensure the fallback itself is within the overall bounds if MIN_DEG_LAT/LON is large
    fallback_lat_max = min(fallback_lat_max, LAT_RANGE[1])
    fallback_lon_max = min(fallback_lon_max, LON_RANGE[1])
    
    fallback_rect = Rect(fallback_lat_min, fallback_lat_max, fallback_lon_min, fallback_lon_max)

    # As a very last resort, if even the default fallback is invalid (e.g., due to extreme MIN_DEG_LAT/LON vs RANGE)
    # return a rectangle spanning the entire valid range (which might violate MIN_DEG_LAT/LON if range is too small).
    # This scenario should be rare with reasonable MIN_DEG_LAT/LON and RANGE settings.
    if is_valid_rect(fallback_rect):
        return fallback_rect
    else:
        logging.warning("Fallback rectangle is also invalid. Returning full range (may violate MIN_DEG_LAT/LON).")
        # This will be valid if LAT_RANGE/LON_RANGE itself respects min < max and side >= MIN_DEG_LAT/LON
        # If not, is_valid_rect will catch it later if used by caller.
        return Rect(LAT_RANGE[0], LAT_RANGE[1], LON_RANGE[0], LON_RANGE[1])


def mutate_rect(r: Rect, strength: float = 0.2) -> Rect:
    """
    Mutate a Rect object by either shifting it or resizing one or more edges.
    The mutation strength influences the magnitude of changes.
    The function attempts to produce a valid rectangle and includes fallback.

    Args:
        r: The original Rect object to mutate.
        strength: A factor controlling the mutation intensity (0.0 to 1.0).
                  Higher values mean larger potential changes.

    Returns:
        A new, mutated, and valid Rect object. Returns the original if
        mutation fails to produce a valid one after several attempts.
    """
    range_lat = LAT_RANGE[1] - LAT_RANGE[0]
    range_lon = LON_RANGE[1] - LON_RANGE[0]
    max_attempts = 100

    for _ in range(max_attempts):
        mutated_r: Rect
        mode = random.choice(['shift', 'resize_one_edge', 'resize_two_edges', 'resize_all_edges'])

        if mode == 'shift':
            # Shift the entire rectangle
            d_lat = int(round(random.gauss(0, strength * range_lat * 0.2))) # Smaller multiplier for shift
            d_lon = int(round(random.gauss(0, strength * range_lon * 0.2)))
            mutated_r = Rect(r.lat_min + d_lat, r.lat_max + d_lat,
                             r.lon_min + d_lon, r.lon_max + d_lon)
        else:
            # Resize one or more edges
            # Calculate potential changes for each edge
            d_lat_min = int(round(random.gauss(0, strength * range_lat * 0.3)))
            d_lat_max = int(round(random.gauss(0, strength * range_lat * 0.3)))
            d_lon_min = int(round(random.gauss(0, strength * range_lon * 0.3)))
            d_lon_max = int(round(random.gauss(0, strength * range_lon * 0.3)))

            new_lat_min, new_lat_max = r.lat_min, r.lat_max
            new_lon_min, new_lon_max = r.lon_min, r.lon_max

            if mode == 'resize_one_edge':
                edge_to_change = random.choice(['lat_min', 'lat_max', 'lon_min', 'lon_max'])
                if edge_to_change == 'lat_min': new_lat_min += d_lat_min
                elif edge_to_change == 'lat_max': new_lat_max += d_lat_max
                elif edge_to_change == 'lon_min': new_lon_min += d_lon_min
                else: new_lon_max += d_lon_max # lon_max
            elif mode == 'resize_two_edges':
                # Change two distinct edges (can be parallel or perpendicular)
                edges_idx = random.sample(range(4), 2)
                temp_edges = ['lat_min', 'lat_max', 'lon_min', 'lon_max']
                chosen_edges = [temp_edges[i] for i in edges_idx]
                if 'lat_min' in chosen_edges: new_lat_min += d_lat_min
                if 'lat_max' in chosen_edges: new_lat_max += d_lat_max
                if 'lon_min' in chosen_edges: new_lon_min += d_lon_min
                if 'lon_max' in chosen_edges: new_lon_max += d_lon_max
            else: # resize_all_edges
                new_lat_min += d_lat_min
                new_lat_max += d_lat_max
                new_lon_min += d_lon_min
                new_lon_max += d_lon_max
            
            # Ensure lat_min <= lat_max and lon_min <= lon_max after mutation
            # Create a temporary rectangle with potentially unordered coords
            temp_r = Rect(new_lat_min, new_lat_max, new_lon_min, new_lon_max)
            # Normalize: ensure min < max
            norm_lat_min = min(temp_r.lat_min, temp_r.lat_max)
            norm_lat_max = max(temp_r.lat_min, temp_r.lat_max)
            norm_lon_min = min(temp_r.lon_min, temp_r.lon_max)
            norm_lon_max = max(temp_r.lon_min, temp_r.lon_max)
            mutated_r = Rect(norm_lat_min, norm_lat_max, norm_lon_min, norm_lon_max)

        if is_valid_rect(mutated_r):
            return mutated_r

    # Fallback: if many attempts fail, return the original rectangle
    # logging.warning(f"Mutation failed to produce a valid rectangle after {max_attempts} attempts. Returning original.")
    return r

def crossover_rect(parent1: Rect, parent2: Rect) -> Rect:
    """
    Perform crossover between two parent Rect objects to produce a child Rect.
    Strategies include:
    1. Taking latitudinal span from one parent and longitudinal span from the other.
    2. Averaging the corresponding coordinates of the parents.
    The function attempts to produce a valid child and includes fallback.

    Args:
        parent1: The first parent Rect object.
        parent2: The second parent Rect object.

    Returns:
        A new child Rect object, validated. Returns parent1 if crossover
        fails to produce a valid child after several attempts.
    """
    max_attempts = 100
    for _ in range(max_attempts):
        child_r: Rect
        strategy = random.random() # Determine crossover strategy

        if strategy < 0.5:
            # Strategy 1: Mix spans (lat from one, lon from other)
            # Randomly decide which parent provides which span
            if random.random() < 0.5:
                child_r = Rect(parent1.lat_min, parent1.lat_max, parent2.lon_min, parent2.lon_max)
            else:
                child_r = Rect(parent2.lat_min, parent2.lat_max, parent1.lon_min, parent1.lon_max)
        else:
            # Strategy 2: Average coordinates (arithmetic mean)
            # This can create children "between" the parents.
            child_lat_min = int(round((parent1.lat_min + parent2.lat_min) / 2))
            child_lat_max = int(round((parent1.lat_max + parent2.lat_max) / 2))
            child_lon_min = int(round((parent1.lon_min + parent2.lon_min) / 2))
            child_lon_max = int(round((parent1.lon_max + parent2.lon_max) / 2))
            
            # Normalize: ensure min < max
            norm_lat_min = min(child_lat_min, child_lat_max)
            norm_lat_max = max(child_lat_min, child_lat_max)
            norm_lon_min = min(child_lon_min, child_lon_max)
            norm_lon_max = max(child_lon_min, child_lon_max)
            child_r = Rect(norm_lat_min, norm_lat_max, norm_lon_min, norm_lon_max)
            
        if is_valid_rect(child_r):
            return child_r

    # Fallback: if many attempts fail, return the first parent
    # logging.warning(f"Crossover failed to produce a valid rectangle after {max_attempts} attempts. Returning parent1.")
    return parent1

# Optionally, expose new symbols in the module's public import list
__all__ = [
    "Rect", "rect_to_vertices", "vertices_to_rect",
    "random_rect", "mutate_rect", "crossover_rect",
    "is_valid_rect",
    "LAT_RANGE", "LON_RANGE", "MIN_DEG_LAT", "MIN_DEG_LON"
]
