# COMSOL2Neuromorphic

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

An open-source, research-grade Python framework designed to bridge finite element method (FEM) COMSOL simulations of memristors with PyTorch-based hardware-aware neuromorphic architectures.

This project is equivalent to a professional, production-quality repository suitable for publication in leading neuromorphic engineering journals such as *IEEE*, *Nature Electronics*, or *Advanced Electronic Materials*.

---

## Roadmap

1. **Module 1: Framework Setup & COMSOL Parser** (Current)
2. **Module 2: Electrical Parameter Extraction**
3. **Module 3: Physics-based Compact Models**
4. **Module 4: Parameter Fitting Engine**
5. **Module 5: Hardware-Aware Memristor Device**
6. **Module 6: Crossbar Array Simulator**
7. **Module 7: Synaptic Plasticity & PyTorch Integration**
8. **Module 8: Spiking Neurons & SNN Simulator**
9. **Module 9: SNN Training & Evaluation (MNIST/FashionMNIST/CIFAR)**
10. **Module 10: Publication-Quality Visualization & Tables**
11. **Module 11: Windows GUI Interface (PySide6)**

---

## Installation

To set up the development environment, clone the repository and run:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install in editable development mode
pip install -e .[dev]
```

## Running Tests

Verify the installation by running the test suite:

```bash
pytest tests/
```
