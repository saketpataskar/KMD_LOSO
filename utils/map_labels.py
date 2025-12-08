import numpy as np

def map_labels(y):
    label_map = {
        1:0, 2:1, 3:2, 4:3,
        5:4, 6:5, 7:6, 8:7
    }
    return np.array([label_map[int(label)] for label in y])
