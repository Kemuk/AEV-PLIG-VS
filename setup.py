"""
Setup script for AEV-PLIG package.

Install the package in development mode with:
    pip install -e .

Install in production mode with:
    pip install .
"""

import os
from setuptools import setup, find_packages

# Read requirements from requirements.txt if it exists
try:
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    # Fallback to manual requirements list
    requirements = [
        'torch>=2.0.0',
        'torch-geometric>=2.3.0',
        'torch-scatter>=2.1.0',
        'rdkit>=2023.0.0',
        'torchani>=2.2.0',
        'biopandas>=0.4.0',
        'qcelemental>=0.25.0',
        'scikit-learn>=1.0.0',
        'pandas>=1.5.0',
        'numpy>=1.23.0',
        'scipy>=1.9.0',
        'tqdm>=4.65.0',
    ]

setup(
    name='aev-plig',
    version='2.0.0',
    description='Graph Neural Network-based Scoring Function for Protein-Ligand Binding Affinity Prediction',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    author='AEV-PLIG Development Team',
    url='https://github.com/Jnelen/AEV-PLIG',
    packages=find_packages(exclude=['scripts', 'tests', 'data', 'output']),
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        'console_scripts': [
            'aev-plig-train=scripts.train:main',
            'aev-plig-predict=scripts.predict:main',
            'aev-plig-generate-graphs=scripts.generate_pdbbind_graphs:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
