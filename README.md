# AutoDiff.jl

A Julia library for training convolutional neural networks using automatic differentiation.

## Overview

AutoDiff.jl provides a lightweight, educational implementation of a convolutional neural network (CNN) framework in Julia. It demonstrates core machine learning concepts including:

- **Automatic Differentiation**: Efficiently compute gradients for backpropagation
- **Convolutional Layers**: Implement convolution operations with learnable filters
- **Neural Network Training**: Build and train models end-to-end
- **Modular Architecture**: Clean, composable components for building custom networks

This library is designed for learning and experimentation, making it ideal for understanding how CNNs work at a fundamental level.

## Features

- Pure Julia implementation with no heavy external dependencies
- Automatic differentiation for gradient computation
- Flexible layer-based architecture
- Built-in data loading utilities
- Example notebooks demonstrating usage

## Quick Start

```julia
using AutoDiff

# Load your data
X_train, y_train = load_data("path/to/data")

# Define a network
model = build_network([
    ConvLayer(32, (3, 3)),
    ReLU(),
    FlattenLayer(),
    DenseLayer(128),
    ReLU(),
    DenseLayer(10)
])

# Train
train_model!(model, X_train, y_train, epochs=20, learning_rate=0.01)
```

## Documentation

- **Full Documentation**: See `docs/report.pdf` for detailed technical documentation
- **Example Usage**: Interactive notebook at `src/example.ipynb` with complete examples and results
- **Source Code**: Implementation files in `src/`

## Project Structure

```
├── src/
│   ├── example.ipynb          # Interactive examples and usage
│   ├── structures.jl          # Core data structures
│   ├── operators.jl           # Mathematical operations
│   ├── graph_operations.jl    # Computation graph management
│   ├── layers.jl              # Neural network layers
│   ├── network.jl             # Network construction and training
│   ├── load_data.jl           # Data loading utilities
│   └── utils.jl               # Helper functions
├── docs/
│   └── report.pdf             # Technical documentation
└── README.md
```

## Implementation Details

### Core Components

1. **Automatic Differentiation**: Builds computation graphs to automatically compute gradients
2. **Layers**: Modular layer implementations (Conv, Dense, ReLU, etc.)
3. **Network**: High-level API for model construction and training
4. **Data Loading**: Utilities for loading and preprocessing datasets

### Design Philosophy

The library emphasizes clarity and educational value, with straightforward implementations that make the underlying mathematics transparent.

## Usage Examples

See `src/example.ipynb` for:
- Dataset loading
- Model construction
- Training loops
- Evaluation and visualization
- Detailed results and analysis

## Technical Reference

For detailed information about the implementation, algorithms, and mathematical foundations, refer to `docs/report.pdf`.

## License

[Add your license here]

## Contributing

Contributions and feedback are welcome! Please refer to the documentation for code style and conventions.

---

**Author**: @jasiu648  
**Language**: Julia  
**Type**: Educational Machine Learning Library
