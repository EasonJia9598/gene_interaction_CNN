# setup.py
from setuptools import setup

setup(
    name='permutated_CNN',
    version='0.1.0',
    packages=['permutated_CNN'],
    author='Zesheng Jia',
    author_email='will.jia.sheng@gmail.com',
    description='A Python package for permutated CNN models',
    #  Add torch in the future with sepcific version in requirements.txt
    install_requires=['numpy', 'pandas', 
                      'tqdm', 'matplotlib', 'seaborn'],
    # url='https://github.com/yourusername/permutated_CNN',
    # long_description='Read more about the package here...',
)
