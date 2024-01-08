import numpy as np

def load_data(path):
    data = np.loadtxt(path, delimiter = ',')
    headerless_data = data[1:]
    x = headerless_data[:,:-1]
    y = headerless_data[:,-1]
    return x, y