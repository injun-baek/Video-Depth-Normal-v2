from collections import deque

class MemoryBank():
    def __init__(self, maxlen):
        self.feature_memory = deque(maxlen=maxlen)
        self.maxlen = maxlen

        self.clear_memory()
        
    def clear_memory(self):
        self.feature_memory.clear()
        
    def get_memory(self):
        return self.feature_memory
        #return [feature.clone() for feature in list(self.feature_memory)]

    def update_memory(self, features):
        if len(self.feature_memory) >= self.maxlen:
            self.feature_memory.popleft()
        self.feature_memory.append(features)
        # memory_features = [feature.clone() for feature in features]
        # self.feature_memory.append(memory_features)
