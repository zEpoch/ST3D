from setuptools import setup, find_packages

setup(
    name='3dst',
    packages=find_packages(),
    version = '0.0.1',
    install_requires=[
        'anndata',
        'itertools',
        'math',
        'pyvista',
        'numpy',
        'pandas',
        'k3d',
        'scanpy',
        'matplotlib',
        'scipy',
        'PyMCubes',
    ]
)

