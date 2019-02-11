import numpy as np
import math



def angle(u, v):
    c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)  # -> cosine of the angle
    return np.arccos(np.clip(c, -1, 1))*180/np.pi


