import math

class EpsilonGreedy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        
    def get_exploration_rate(self, steps_done):
        eps_threshold = self.end + (self.start - self.end) * math.exp(- steps_done / self.decay)
        return eps_threshold