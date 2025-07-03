from abc import ABC, abstractmethod

# 离散化基类
class BaseDiscretizer(ABC):
    @abstractmethod
    def discretize(self, values):
        """
        将连续值离散化 | Discretize continuous values
        """
        pass

    @abstractmethod
    def decode(self, bins):
        """
        将离散值解码回连续值 | Decode discrete values back to continuous values
        """
        pass