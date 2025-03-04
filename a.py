import numpy as np

def _cum_sum( arr, factor):
    result = np.zeros_like(arr, dtype=float)
    running = 0
    
    for i in reversed(range(len(arr))):
        running = arr[i] + factor * running
        result[i] = running
        
    return result 

