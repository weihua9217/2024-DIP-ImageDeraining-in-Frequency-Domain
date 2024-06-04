import matplotlib.pyplot as plt
from skimage.draw import line, disk
import numpy as np

import matplotlib.pyplot as plt
from skimage.draw import line, disk

def set_line_to_values(matrix, point, angle, thickness, values):
    rows, cols = matrix.shape
    x0, y0 = point
    angle_rad = np.deg2rad(angle)
    
    # 計算直線的終點
    length = max(rows, cols)
    x1 = x0 + length * np.cos(angle_rad)
    y1 = y0 + length * np.sin(angle_rad)
    

    rr, cc = line(int(y0), int(x0), int(y1), int(x1))
    
    valid_indices = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
    rr = rr[valid_indices]
    cc = cc[valid_indices]

    for r, c in zip(rr, cc):
        matrix[r-thickness:r+thickness+1, c] = values

    length = max(rows, cols)
    x1 = x0 - length * np.cos(angle_rad)
    y1 = y0 - length * np.sin(angle_rad)
    
    rr, cc = line(int(y0), int(x0), int(y1), int(x1))
    
    valid_indices = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
    rr = rr[valid_indices]
    cc = cc[valid_indices]

    for r, c in zip(rr, cc):
        matrix[r-thickness:r+thickness+1, c] = values