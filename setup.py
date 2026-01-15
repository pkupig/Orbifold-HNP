from setuptools import setup, find_packages
import os

# 读取README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="orbifold-coloring-system",
    version="1.0.0",
    author="Mathematics Institute, University of Warwick",
    author_email="",
    description="Iterative counter-example search system for unit distance graph coloring on orbifolds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/orbifold-coloring",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "viz": [
            "seaborn>=0.12.0",
            "plotly>=5.0.0",
            "networkx>=3.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "orbifold-coloring=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "data/*.pkl"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/orbifold-coloring/issues",
        "Source": "https://github.com/yourusername/orbifold-coloring",
    },
)