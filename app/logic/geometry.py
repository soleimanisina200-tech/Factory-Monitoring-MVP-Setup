def get_foot_point(x1, y1, x2, y2):
    # center_x = (x1 + x2) / 2
    # foot_y = y2 (the bottom of the bounding box)
    center_x = int((x1 + x2) / 2)
    foot_y = int(y2)
    return (center_x, foot_y)