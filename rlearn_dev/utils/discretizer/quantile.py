import numpy as np
from .base import BaseDiscretizer

# 分位数离散化类
class QuantileDiscretizer(BaseDiscretizer):
    def __init__(self, num_bins):
        """
        初始化分位数离散化器，只设置分箱数量 | Initialize the quantile discretizer, only set the number of bins
        """
        self.num_bins = num_bins
        self.bin_edges = None

    def fit(self, data):
        """
        根据输入数据计算分位数边界 | Compute quantile bin edges based on the input data
        """
        self.bin_edges = np.percentile(data, np.linspace(0, 100, self.num_bins + 1))
    
    def discretize(self, values):
        """
        根据分位数对连续值离散化 | Discretize continuous values based on quantiles
        """
        if self.bin_edges is None:
            raise ValueError("The discretizer has not been fitted yet. Call 'fit' with data first.")
        bins = np.digitize(values, self.bin_edges, right=True) - 1
        bins = np.clip(bins, 0, self.num_bins - 1)
        return bins
    
    def decode(self, bins):
        """
        解码离散值为近似连续值 | Decode discrete values to approximate continuous values
        """
        if self.bin_edges is None:
            raise ValueError("The discretizer has not been fitted yet. Call 'fit' with data first.")
        bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        return bin_centers[bins]