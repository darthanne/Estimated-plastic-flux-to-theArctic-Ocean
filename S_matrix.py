from numpy import *


def inside_contour(x, y, perimeter, bbox=None):
    # ---------------------------------------------------------------------
    # Test whether (x,y) is inside a contour given by a vertex
    # list perimeter (without end point duplication)
    # bbox is an optional argument that can be provided for speed-up
    # Otherwise algorithm applies perimeter scan directly (which can be
    # avoided in many cases). bbox can be larger than minimal bbox, but not smaller
    # (then algorithm fails)
    # Based on Ray Casting Method algorithm, from
    #   http://geospatialpython.com/2011/01/point-in-polygon.html
    # augmented with bbox prescreening
    # algorithm is based on external point == (+infinity, y)
    # -------------------------------------------------------------------
    #
    #
    # Proceed to scanning polygon line segments
    #
    n = len(perimeter)
    inside = False
    p1x, p1y = perimeter[0][:2]
    for i in range(1, n + 1):  # http://geospatialpython.com/2011/01/point-in-polygon.html starts loop at i==0, ???
        p2x, p2y = perimeter[i % n][:2]  # implicit closure for i==n
        if y > min(p1y, p2y):  # > operand reflects collinear case
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xints = None  # have xints defined
                    if p1x == p2x or x <= xints:
                        inside = not inside  # toggle
        p1x, p1y = p2x, p2y  # roll points
    return inside
