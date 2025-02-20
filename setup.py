from setuptools import setup, find_packages

setup(
    name="spin_package",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'numpy>=1.18.0',
        'numba>=0.50.0',
        'scipy>=1.4.1',
        'tqdm>=4.46.0',
    ],
    python_requires=">=3.7",  # Specify your minimum Python version
)