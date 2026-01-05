#!/usr/bin/env python3
"""
EchoSurf Model Weight Exporter

Exports TensorFlow/Keras model weights to C++ loadable format.
Supports binary (.esml), NumPy (.npz), and JSON formats.
"""

import numpy as np
import struct
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

MAGIC = 0x4C4D5345  # "ESML"
VERSION = 1


def export_weights_binary(weights_dict: dict, output_path: str) -> None:
    """
    Export weights to EchoSurf binary format (.esml).

    Binary format:
    - Header: MAGIC (4), VERSION (4), NUM_WEIGHTS (4)
    - Per weight: NAME_LEN (4), NAME, NUM_DIMS (4), DIMS, DATA
    """
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', MAGIC))
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<I', len(weights_dict)))

        # Weights
        for name, tensor in weights_dict.items():
            # Ensure float32
            tensor = np.asarray(tensor, dtype=np.float32)

            # Name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            # Shape
            f.write(struct.pack('<I', len(tensor.shape)))
            for dim in tensor.shape:
                f.write(struct.pack('<I', dim))

            # Data
            f.write(tensor.tobytes())

    print(f"Exported {len(weights_dict)} weight tensors to {output_path}")


def export_weights_npz(weights_dict: dict, output_path: str) -> None:
    """Export weights to NumPy NPZ format."""
    # Convert names to valid npz keys (replace / with _)
    clean_weights = {
        name.replace('/', '_'): np.asarray(tensor, dtype=np.float32)
        for name, tensor in weights_dict.items()
    }
    np.savez(output_path, **clean_weights)
    print(f"Exported weights to {output_path}")


def export_weights_json(weights_dict: dict, output_path: str) -> None:
    """Export weights to JSON format (human readable)."""
    json_data = {}
    for name, tensor in weights_dict.items():
        tensor = np.asarray(tensor, dtype=np.float32)
        json_data[name] = {
            "shape": list(tensor.shape),
            "data": tensor.flatten().tolist()
        }

    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"Exported weights to {output_path}")


def extract_keras_weights(model) -> dict:
    """Extract weights from a Keras model."""
    weights_dict = {}
    dense_idx = 0

    for layer in model.layers:
        layer_weights = layer.get_weights()

        if hasattr(layer, 'kernel') and len(layer_weights) > 0:
            # Dense layer
            weights_dict[f'dense_{dense_idx}/kernel'] = layer_weights[0]

            if len(layer_weights) > 1 and layer.use_bias:
                weights_dict[f'dense_{dense_idx}/bias'] = layer_weights[1]

            dense_idx += 1

        elif hasattr(layer, 'gamma') and len(layer_weights) >= 4:
            # BatchNorm layer
            bn_idx = dense_idx  # Reuse counter for simplicity
            weights_dict[f'batch_norm_{bn_idx}/gamma'] = layer_weights[0]
            weights_dict[f'batch_norm_{bn_idx}/beta'] = layer_weights[1]
            weights_dict[f'batch_norm_{bn_idx}/moving_mean'] = layer_weights[2]
            weights_dict[f'batch_norm_{bn_idx}/moving_variance'] = layer_weights[3]

    return weights_dict


def export_model(model, output_dir: str, model_name: str) -> None:
    """Export a Keras model to all formats."""
    os.makedirs(output_dir, exist_ok=True)

    weights = extract_keras_weights(model)

    # Export to all formats
    export_weights_binary(weights, os.path.join(output_dir, f'{model_name}.esml'))
    export_weights_npz(weights, os.path.join(output_dir, f'{model_name}.npz'))
    export_weights_json(weights, os.path.join(output_dir, f'{model_name}.json'))

    print(f"\nExported {model_name}:")
    print(f"  Total weights: {len(weights)}")
    print(f"  Total params: {sum(np.prod(w.shape) for w in weights.values())}")


def main():
    """Main export function."""
    try:
        from ml_system import MLSystem
    except ImportError:
        print("Error: Could not import ml_system. Make sure you're in the echosurf directory.")
        sys.exit(1)

    # Initialize ML system
    print("Loading ML System...")
    ml = MLSystem()

    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'weights')

    # Export each model
    print("\n=== Exporting Models ===\n")

    if hasattr(ml, 'reflex_model') and ml.reflex_model is not None:
        export_model(ml.reflex_model, output_dir, 'reflex_model')

    if hasattr(ml, 'tactical_model') and ml.tactical_model is not None:
        export_model(ml.tactical_model, output_dir, 'tactical_model')

    if hasattr(ml, 'echo_value_model') and ml.echo_value_model is not None:
        export_model(ml.echo_value_model, output_dir, 'echo_value_model')

    if hasattr(ml, 'visual_model') and ml.visual_model is not None:
        export_model(ml.visual_model, output_dir, 'visual_model')

    if hasattr(ml, 'behavior_model') and ml.behavior_model is not None:
        export_model(ml.behavior_model, output_dir, 'behavior_model')

    if hasattr(ml, 'pattern_model') and ml.pattern_model is not None:
        export_model(ml.pattern_model, output_dir, 'pattern_model')

    print(f"\n=== Export Complete ===")
    print(f"Weights saved to: {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    main()
