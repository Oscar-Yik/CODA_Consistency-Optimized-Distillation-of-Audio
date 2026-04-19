import numpy as np
# We use this instead of a deque to hold captured audio. It is faster
class FastBuffer:
    def __init__(self, size):
        self.buffer = np.zeros(size, dtype=np.float32)
        self.size = size
        
    def add(self, chunk):
        n = len(chunk)

        # shift and replace without roll overhead
        self.buffer[:-n] = self.buffer[n:]
        self.buffer[-n:] = chunk
        return self.buffer