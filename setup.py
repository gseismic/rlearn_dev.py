from setuptools import setup, find_packages


setup(
    name='rlearn_dev', 
    version='0.0.2', 
    packages=find_packages(),
    description='Reinforcement Learning Algorithms [dev]',
    install_requires = ['torch', 'numpy', 'loguru', 'sklearn'],
    scripts=[],
    python_requires = '>=3.8',
    include_package_data=True,
    author='Liu Shengli',
    url='http://github.com/gseismic/rlearn_dev.py',
    zip_safe=False,
    author_email='liushengli203@163.com'
)
