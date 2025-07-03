import numpy as np
from .base import BaseDiscretizer

# 均匀离散化类
class UniformDiscretizer(BaseDiscretizer):
    def __init__(self, low, high, num_bins):
        """
        初始化均匀离散化器 | Initialize the uniform discretizer
        """
        self.low = np.array(low)
        self.high = np.array(high)
        self.num_bins = np.array(num_bins)

    def discretize(self, values):
        """
        将连续值离散化 | Discretize continuous values
        """
        bins = np.floor((values - self.low) * self.num_bins / (self.high - self.low)).astype(int)
        bins = np.clip(bins, 0, self.num_bins - 1)
        return bins

    def decode(self, bins):
        """
        将离散值解码回连续值 | Decode discrete values back to continuous
        """
        values = self.low + (bins + 0.5) * (self.high - self.low) / self.num_bins
        return values