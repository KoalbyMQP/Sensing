


def line_intersection(line1, line2):
    """Find intersection point of two lines defined by two points each.
    Returns (x, y) or None if lines are parallel."""
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    
    # Calculate denominators
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None  # Lines are parallel
    
    # Calculate intersection
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return (int(x), int(y))


