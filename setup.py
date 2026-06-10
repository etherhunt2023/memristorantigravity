from setuptools import find_packages, setup

setup(
    name="comsol2neuromorphic",
    version="0.1.0",
    description=(
        "A Python framework to convert COMSOL memristor simulations "
        "into hardware-aware neuromorphic networks"
    ),
    author="Antigravity AI",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.12",
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "pandas>=2.2.0",
        "torch>=2.2.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0.1",
        "pyside6>=6.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "black>=24.1.0",
            "ruff>=0.2.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
