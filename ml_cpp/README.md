# EchoSurf ML C++ Framework

High-performance ML inference engine for gaming applications, optimized for real-time response.

## Performance Targets

| Model | Target Latency | Use Case |
|-------|----------------|----------|
| Reflex Model | <10ms | Lightning-fast gaming responses |
| Tactical Model | <50ms | Strategic decision making |
| Echo Value Model | <5ms | Content importance evaluation |

## Features

- **SIMD Optimization**: AVX2/SSE4 vectorized operations
- **Zero-Copy Tensors**: Aligned memory for optimal performance
- **Minimal Allocations**: Pre-allocated buffers for inference
- **Python Bindings**: Seamless integration with existing Python code
- **Multiple Weight Formats**: Binary (.esml), NumPy (.npz), JSON

## Architecture

```
ml_cpp/
├── include/           # Header files
│   ├── tensor.h       # Tensor operations
│   ├── layers.h       # Neural network layers
│   ├── model.h        # Model definitions
│   └── model_loader.h # Weight loading
├── src/               # Implementation
├── python/            # Python bindings (pybind11)
├── tests/             # Unit tests
├── benchmarks/        # Performance benchmarks
└── scripts/           # Utility scripts
```

## Building

### Prerequisites

- CMake 3.14+
- C++17 compiler (GCC 7+, Clang 5+, MSVC 2019+)
- Python 3.8+ (for bindings)
- pybind11 (auto-downloaded if not found)

### Build Commands

```bash
cd ml_cpp
mkdir build && cd build

# Standard build
cmake ..
make -j$(nproc)

# Release build with optimizations
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Run benchmarks
./benchmark_inference
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_PYTHON_BINDINGS` | ON | Build Python module |
| `BUILD_TESTS` | ON | Build unit tests |
| `BUILD_BENCHMARKS` | ON | Build benchmarks |

## Usage

### C++ Usage

```cpp
#include "model.h"

using namespace echosurf::ml;

// Create reflex model
ReflexModel reflex;

// Make prediction
ReflexModel::ReflexInput input{
    0.8f,  // threat_proximity
    0.5f,  // threat_direction
    0.9f,  // player_state
    0.3f,  // movement_momentum
    0.7f,  // time_pressure
    0.2f,  // cover_availability
    0.6f,  // aim_confidence
    0.8f   // situation_clarity
};

auto action = reflex.predict(input);
// action: Dodge, CounterAttack, TakeCover, or HoldPosition
```

### Python Usage

```python
import numpy as np
from ml_cpp.python import ReflexModel, TacticalModel, is_available

# Check if C++ acceleration is available
if is_available():
    # Create model
    reflex = ReflexModel()

    # Predict action
    action = reflex.predict_action(
        threat_proximity=0.8,
        threat_direction=0.5,
        player_state=0.9,
        movement_momentum=0.3,
        time_pressure=0.7,
        cover_availability=0.2,
        aim_confidence=0.6,
        situation_clarity=0.8
    )
    print(f"Action: {action}")

    # Or use numpy arrays directly
    input_array = np.array([[0.8, 0.5, 0.9, 0.3, 0.7, 0.2, 0.6, 0.8]], dtype=np.float32)
    probs = reflex.predict(input_array)
```

### Loading Pre-trained Weights

```cpp
// C++
ReflexModel model;
ModelLoader::load_reflex_model("weights/reflex_model.esml", model);
```

```python
# Python - Export weights first
python scripts/export_weights.py

# Then load in C++
```

## Model Architectures

### Reflex Model
```
Input(8) → Dense(128, ReLU) → Dropout(0.1) →
Dense(64, ReLU) → Dropout(0.1) →
Dense(32, ReLU) → Dense(4, Softmax)
```

**Input Features:**
- threat_proximity: How close is the threat (0-1)
- threat_direction: Direction of threat (-1 to 1)
- player_state: Health/shield status (0-1)
- movement_momentum: Current momentum
- time_pressure: Urgency factor
- cover_availability: Nearby cover options
- aim_confidence: Aim lock quality
- situation_clarity: Visibility/information

**Output Actions:**
- 0: Dodge
- 1: Counter Attack
- 2: Take Cover
- 3: Hold Position

### Tactical Model
```
Input(16) → Dense(128, ReLU) → Dropout(0.2) →
Dense(64, ReLU) → Dropout(0.2) →
Dense(32, ReLU) → Dense(16, ReLU) → Dense(8, Softmax)
```

**Output Actions:**
- 0: Attack
- 1: Defend
- 2: Flank
- 3: Retreat
- 4: Heal
- 5: Resupply
- 6: Support
- 7: Objective

## Benchmarks

Run benchmarks to verify performance on your system:

```bash
./benchmark_inference 10000
```

Expected output (on modern CPU with AVX2):
```
REFLEX MODEL BENCHMARK
  Mean:    0.015 ms
  P95:     0.020 ms
  Target:  10.000 ms
  Status:  PASS

TACTICAL MODEL BENCHMARK
  Mean:    0.025 ms
  P95:     0.035 ms
  Target:  50.000 ms
  Status:  PASS
```

## Integration with Python ML System

The C++ framework is designed to work alongside the existing Python ML system:

1. **Training**: Use Python TensorFlow/Keras for training
2. **Export**: Use `scripts/export_weights.py` to export weights
3. **Inference**: Use C++ for low-latency inference

```python
# In your Python code
from ml_cpp.python import ReflexModel, is_available

class HybridMLSystem:
    def __init__(self):
        if is_available():
            # Use C++ for fast inference
            self.reflex = ReflexModel()
            # Load trained weights
            ModelLoader.load_reflex_model("weights/reflex_model.esml", self.reflex)
        else:
            # Fallback to Python
            from ml_system import MLSystem
            self.ml = MLSystem()
            self.reflex = self.ml.reflex_model
```

## License

Part of the EchoSurf project.
