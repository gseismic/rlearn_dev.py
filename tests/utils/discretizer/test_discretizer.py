import numpy as np
from rlearn_dev.utils.discretizer.quantile import QuantileDiscretizer
from rlearn_dev.utils.discretizer.uniform import UniformDiscretizer

def test_uniform_discretizer():
    values = np.array([0.05, 0.1, 1.25])
    low = np.array([0.0, 0.0, 0.0])
    high = np.array([1.0, 1.0, 1.0])
    num_bins = np.array([10, 10, 10])

    uniform_discretizer = UniformDiscretizer(low, high, num_bins)
    bins = uniform_discretizer.discretize(values)
    decoded_values = uniform_discretizer.decode(bins)
    
    bins2 = uniform_discretizer.discretize(decoded_values)
    decoded_values2 = uniform_discretizer.decode(bins2)   

    np.allclose(decoded_values, decoded_values2)
    # print("均匀离散化结果:", bins)
    # print("均匀解码后的值:", decoded_values)

def test_quantile_discretizer():
    values = np.array([0.2, 0.5, 0.8])
    data = np.random.rand(1000)  # 随机生成数据
    quantile_discretizer = QuantileDiscretizer(num_bins=10)

    # 使用fit方法拟合数据
    quantile_discretizer.fit(data)

    # 离散化值
    quantile_bins = quantile_discretizer.discretize(values)
    quantile_decoded = quantile_discretizer.decode(quantile_bins)
    
    quantile_bins2 = quantile_discretizer.discretize(quantile_decoded)
    quantile_decoded2 = quantile_discretizer.decode(quantile_bins2)

    np.allclose(quantile_decoded, quantile_decoded2)

    # print("分位数离散化结果:", quantile_bins)
    # print("分位数解码后的值:", quantile_decoded)
    # print("分位数编码后的值:", quantile_bins2)
    # print("分位数解码后的值:", quantile_decoded2)

if __name__ == "__main__":
    if 0:
        test_uniform_discretizer()
    if 1:
        test_quantile_discretizer()
