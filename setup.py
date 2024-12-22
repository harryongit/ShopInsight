from setuptools import setup, find_packages

setup(
    name="ecom_data_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.2',
        'matplotlib>=3.4.2',
        'seaborn>=0.11.1',
        'plotly>=5.1.0',
        'statsmodels>=0.12.2',
        'pytest>=6.2.4',
        'jupyter>=1.0.0'
    ]
)
