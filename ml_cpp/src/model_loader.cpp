/**
 * EchoSurf ML Framework - Model Loader Implementation
 */

#include "model_loader.h"
#include <cstring>
#include <sstream>
#include <regex>
#include <stdexcept>

namespace echosurf {
namespace ml {

// ============================================================================
// File I/O Helpers
// ============================================================================

std::vector<char> ModelLoader::read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Error reading file: " + path);
    }

    return buffer;
}

// ============================================================================
// Binary Format
// ============================================================================

bool BinaryWeightFormat::write(std::ostream& out,
                              const std::unordered_map<std::string, Tensor>& weights) {
    // Write header
    uint32_t magic = MAGIC;
    uint32_t version = VERSION;
    uint32_t num_weights = static_cast<uint32_t>(weights.size());

    out.write(reinterpret_cast<char*>(&magic), 4);
    out.write(reinterpret_cast<char*>(&version), 4);
    out.write(reinterpret_cast<char*>(&num_weights), 4);

    // Write each weight
    for (const auto& [name, tensor] : weights) {
        // Name
        uint32_t name_len = static_cast<uint32_t>(name.size());
        out.write(reinterpret_cast<char*>(&name_len), 4);
        out.write(name.c_str(), name_len);

        // Shape
        uint32_t num_dims = static_cast<uint32_t>(tensor.shape().ndim());
        out.write(reinterpret_cast<char*>(&num_dims), 4);
        for (size_t i = 0; i < num_dims; ++i) {
            uint32_t dim = static_cast<uint32_t>(tensor.shape()[i]);
            out.write(reinterpret_cast<char*>(&dim), 4);
        }

        // Data
        out.write(reinterpret_cast<const char*>(tensor.data()),
                 tensor.size() * sizeof(float));
    }

    return out.good();
}

bool BinaryWeightFormat::read(std::istream& in,
                             std::unordered_map<std::string, Tensor>& weights) {
    // Read header
    uint32_t magic, version, num_weights;
    in.read(reinterpret_cast<char*>(&magic), 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&num_weights), 4);

    if (magic != MAGIC) {
        return false;  // Invalid format
    }

    if (version != VERSION) {
        return false;  // Unsupported version
    }

    // Read each weight
    for (uint32_t w = 0; w < num_weights; ++w) {
        // Name
        uint32_t name_len;
        in.read(reinterpret_cast<char*>(&name_len), 4);
        std::string name(name_len, '\0');
        in.read(&name[0], name_len);

        // Shape
        uint32_t num_dims;
        in.read(reinterpret_cast<char*>(&num_dims), 4);
        std::vector<size_t> dims(num_dims);
        for (uint32_t d = 0; d < num_dims; ++d) {
            uint32_t dim;
            in.read(reinterpret_cast<char*>(&dim), 4);
            dims[d] = dim;
        }

        TensorShape shape(dims);
        Tensor tensor(shape);

        // Data
        in.read(reinterpret_cast<char*>(tensor.data()),
               tensor.size() * sizeof(float));

        weights[name] = std::move(tensor);
    }

    return in.good();
}

// ============================================================================
// ModelLoader Implementation
// ============================================================================

bool ModelLoader::load_binary(const std::string& path,
                             std::unordered_map<std::string, Tensor>& weights) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    return BinaryWeightFormat::read(file, weights);
}

bool ModelLoader::save_binary(const std::string& path,
                             const std::unordered_map<std::string, Tensor>& weights) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    return BinaryWeightFormat::write(file, weights);
}

Tensor ModelLoader::load_npy(const std::string& path) {
    std::vector<char> data = read_file(path);

    // Check magic number
    if (data.size() < 10 || data[0] != '\x93' ||
        data[1] != 'N' || data[2] != 'U' ||
        data[3] != 'M' || data[4] != 'P' || data[5] != 'Y') {
        throw std::runtime_error("Invalid NPY file: " + path);
    }

    // Parse version
    uint8_t major = static_cast<uint8_t>(data[6]);
    uint8_t minor = static_cast<uint8_t>(data[7]);
    (void)minor;  // Unused

    // Header length
    size_t header_len;
    size_t offset;
    if (major == 1) {
        uint16_t len;
        std::memcpy(&len, &data[8], 2);
        header_len = len;
        offset = 10 + header_len;
    } else {
        uint32_t len;
        std::memcpy(&len, &data[8], 4);
        header_len = len;
        offset = 12 + header_len;
    }

    // Parse header (Python dict-like format)
    std::string header(data.begin() + (major == 1 ? 10 : 12),
                      data.begin() + offset);

    // Extract shape from header
    std::regex shape_regex(R"(\('shape'\s*:\s*\(([^\)]*)\))");
    std::smatch shape_match;
    std::vector<size_t> dims;

    if (std::regex_search(header, shape_match, shape_regex)) {
        std::string shape_str = shape_match[1].str();
        std::istringstream ss(shape_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            // Remove whitespace
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t,") + 1);
            if (!token.empty()) {
                dims.push_back(std::stoull(token));
            }
        }
    }

    // Default to 1D if shape parsing failed
    if (dims.empty()) {
        size_t total_floats = (data.size() - offset) / sizeof(float);
        dims.push_back(total_floats);
    }

    TensorShape shape(dims);
    Tensor tensor(shape);

    // Copy data
    std::memcpy(tensor.data(), &data[offset], tensor.size() * sizeof(float));

    return tensor;
}

bool ModelLoader::load_npz(const std::string& path,
                          std::unordered_map<std::string, Tensor>& weights) {
    // NPZ is a ZIP file containing .npy files
    // For simplicity, we'll use a basic implementation
    // In production, use a proper ZIP library

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read ZIP local file headers
    while (file.good()) {
        uint32_t signature;
        file.read(reinterpret_cast<char*>(&signature), 4);

        if (signature != 0x04034b50) {  // Local file header
            break;
        }

        // Skip version, flags, compression, time, date, crc32
        file.seekg(22, std::ios::cur);

        uint16_t name_len, extra_len;
        file.read(reinterpret_cast<char*>(&name_len), 2);
        file.read(reinterpret_cast<char*>(&extra_len), 2);

        // Read filename
        std::string filename(name_len, '\0');
        file.read(&filename[0], name_len);

        // Skip extra field
        file.seekg(extra_len, std::ios::cur);

        // Remove .npy extension for weight name
        std::string weight_name = filename;
        if (weight_name.size() > 4 &&
            weight_name.substr(weight_name.size() - 4) == ".npy") {
            weight_name = weight_name.substr(0, weight_name.size() - 4);
        }

        // Read NPY content
        // Note: This assumes no compression (store method)
        // Production code should handle deflate compression

        // Read NPY header
        char npy_magic[6];
        file.read(npy_magic, 6);

        if (npy_magic[0] != '\x93' || npy_magic[1] != 'N') {
            continue;  // Skip non-NPY entries
        }

        uint8_t major = static_cast<uint8_t>(file.get());
        file.get();  // minor version

        size_t header_len;
        if (major == 1) {
            uint16_t len;
            file.read(reinterpret_cast<char*>(&len), 2);
            header_len = len;
        } else {
            uint32_t len;
            file.read(reinterpret_cast<char*>(&len), 4);
            header_len = len;
        }

        std::string header(header_len, '\0');
        file.read(&header[0], header_len);

        // Parse shape
        std::vector<size_t> dims;
        size_t pos = header.find("'shape'");
        if (pos != std::string::npos) {
            size_t start = header.find('(', pos);
            size_t end = header.find(')', start);
            if (start != std::string::npos && end != std::string::npos) {
                std::string shape_str = header.substr(start + 1, end - start - 1);
                std::istringstream ss(shape_str);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    token.erase(0, token.find_first_not_of(" \t"));
                    token.erase(token.find_last_not_of(" \t,") + 1);
                    if (!token.empty()) {
                        dims.push_back(std::stoull(token));
                    }
                }
            }
        }

        if (dims.empty()) {
            continue;
        }

        TensorShape shape(dims);
        Tensor tensor(shape);
        file.read(reinterpret_cast<char*>(tensor.data()),
                 tensor.size() * sizeof(float));

        weights[weight_name] = std::move(tensor);
    }

    return !weights.empty();
}

bool ModelLoader::load_json(const std::string& path,
                           std::unordered_map<std::string, Tensor>& weights) {
    std::vector<char> data = read_file(path);
    std::string json_str(data.begin(), data.end());
    return parse_json_weights(json_str, weights);
}

bool ModelLoader::parse_json_weights(const std::string& json_str,
                                     std::unordered_map<std::string, Tensor>& weights) {
    // Simple JSON parser for weight format
    // Format: {"name": {"shape": [d1, d2], "data": [f1, f2, ...]}, ...}

    // This is a minimal implementation
    // Production code should use a proper JSON library

    size_t pos = 0;
    while ((pos = json_str.find("\"shape\"", pos)) != std::string::npos) {
        // Find the weight name (look backwards for previous key)
        size_t name_end = json_str.rfind("\":", pos);
        if (name_end == std::string::npos) break;
        size_t name_start = json_str.rfind("\"", name_end - 1);
        if (name_start == std::string::npos) break;

        std::string name = json_str.substr(name_start + 1, name_end - name_start - 2);

        // Parse shape
        size_t shape_start = json_str.find('[', pos);
        size_t shape_end = json_str.find(']', shape_start);
        std::string shape_str = json_str.substr(shape_start + 1, shape_end - shape_start - 1);

        std::vector<size_t> dims;
        std::istringstream shape_ss(shape_str);
        std::string dim_token;
        while (std::getline(shape_ss, dim_token, ',')) {
            dim_token.erase(0, dim_token.find_first_not_of(" \t"));
            dim_token.erase(dim_token.find_last_not_of(" \t") + 1);
            if (!dim_token.empty()) {
                dims.push_back(std::stoull(dim_token));
            }
        }

        // Parse data
        size_t data_pos = json_str.find("\"data\"", shape_end);
        size_t data_start = json_str.find('[', data_pos);
        size_t data_end = json_str.find(']', data_start);
        std::string data_str = json_str.substr(data_start + 1, data_end - data_start - 1);

        std::vector<float> values;
        std::istringstream data_ss(data_str);
        std::string val_token;
        while (std::getline(data_ss, val_token, ',')) {
            val_token.erase(0, val_token.find_first_not_of(" \t"));
            val_token.erase(val_token.find_last_not_of(" \t") + 1);
            if (!val_token.empty()) {
                values.push_back(std::stof(val_token));
            }
        }

        TensorShape shape(dims);
        weights[name] = Tensor(shape, values);

        pos = data_end;
    }

    return !weights.empty();
}

bool ModelLoader::apply_weights(SequentialModel& model,
                               const std::unordered_map<std::string, Tensor>& weights) {
    // Map weights to layers by naming convention:
    // dense_0/kernel, dense_0/bias, dense_1/kernel, etc.

    size_t dense_idx = 0;
    for (size_t i = 0; i < model.num_layers(); ++i) {
        Layer& layer = model.layer(i);

        if (layer.name() == "dense") {
            std::string kernel_key = "dense_" + std::to_string(dense_idx) + "/kernel";
            std::string bias_key = "dense_" + std::to_string(dense_idx) + "/bias";

            auto kernel_it = weights.find(kernel_key);
            auto bias_it = weights.find(bias_key);

            std::unordered_map<std::string, Tensor> layer_weights;
            if (kernel_it != weights.end()) {
                layer_weights["kernel"] = kernel_it->second;
            }
            if (bias_it != weights.end()) {
                layer_weights["bias"] = bias_it->second;
            }

            layer.set_weights(layer_weights);
            dense_idx++;
        }
    }

    return true;
}

bool ModelLoader::load_model(const std::string& path, SequentialModel& model,
                            WeightFormat format) {
    std::unordered_map<std::string, Tensor> weights;

    bool loaded = false;
    switch (format) {
        case WeightFormat::Binary:
            loaded = load_binary(path, weights);
            break;
        case WeightFormat::NumPy:
            loaded = load_npz(path, weights);
            break;
        case WeightFormat::JSON:
            loaded = load_json(path, weights);
            break;
    }

    if (!loaded) {
        return false;
    }

    return apply_weights(model, weights);
}

ModelCheckpoint ModelLoader::load_checkpoint_info(const std::string& path) {
    ModelCheckpoint info;

    std::unordered_map<std::string, Tensor> weights;
    if (load_binary(path, weights)) {
        for (const auto& [name, tensor] : weights) {
            info.layer_names.push_back(name);
            info.weight_shapes[name] = tensor.shape();
            info.total_params += tensor.size();
        }
        info.num_layers = weights.size() / 2;  // Assuming kernel + bias per layer
    }

    return info;
}

bool ModelLoader::load_reflex_model(const std::string& path, ReflexModel& model) {
    return load_model(path, model.model());
}

bool ModelLoader::load_tactical_model(const std::string& path, TacticalModel& model) {
    return load_model(path, model.model());
}

bool ModelLoader::load_echo_value_model(const std::string& path, EchoValueModel& model) {
    return load_model(path, model.model());
}

// ============================================================================
// WeightExporterGenerator Implementation
// ============================================================================

std::string WeightExporterGenerator::generate_export_script(const std::string& model_type) {
    std::ostringstream script;

    script << R"(#!/usr/bin/env python3
"""
EchoSurf Model Weight Exporter

Exports TensorFlow/Keras model weights to C++ loadable format.
Generated by EchoSurf ML Framework.
"""

import numpy as np
import struct
import os

MAGIC = 0x4C4D5345  # "ESML"
VERSION = 1

def export_weights_binary(model, output_path):
    """Export model weights to binary format."""
    weights = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'kernel'):
            weights[f'dense_{i}//kernel'] = layer.kernel.numpy()
        if hasattr(layer, 'bias') and layer.bias is not None:
            weights[f'dense_{i}//bias'] = layer.bias.numpy()

    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', MAGIC))
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<I', len(weights)))

        # Weights
        for name, tensor in weights.items():
            # Name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            # Shape
            f.write(struct.pack('<I', len(tensor.shape)))
            for dim in tensor.shape:
                f.write(struct.pack('<I', dim))

            # Data (float32)
            f.write(tensor.astype(np.float32).tobytes())

    print(f"Exported {len(weights)} weight tensors to {output_path}")

def export_weights_npz(model, output_path):
    """Export model weights to NumPy NPZ format."""
    weights = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'kernel'):
            weights[f'dense_{i}_kernel'] = layer.kernel.numpy()
        if hasattr(layer, 'bias') and layer.bias is not None:
            weights[f'dense_{i}_bias'] = layer.bias.numpy()

    np.savez(output_path, **weights)
    print(f"Exported weights to {output_path}")

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from ml_system import MLSystem

    ml = MLSystem()
)";

    // Add model-specific export code
    if (model_type == "reflex") {
        script << R"(
    model = ml.reflex_model
    export_weights_binary(model, 'reflex_model.esml')
    export_weights_npz(model, 'reflex_model.npz')
)";
    } else if (model_type == "tactical") {
        script << R"(
    model = ml.tactical_model
    export_weights_binary(model, 'tactical_model.esml')
    export_weights_npz(model, 'tactical_model.npz')
)";
    } else if (model_type == "echo_value") {
        script << R"(
    model = ml.echo_value_model
    export_weights_binary(model, 'echo_value_model.esml')
    export_weights_npz(model, 'echo_value_model.npz')
)";
    } else {
        script << R"(
    # Export all models
    export_weights_binary(ml.reflex_model, 'reflex_model.esml')
    export_weights_binary(ml.tactical_model, 'tactical_model.esml')
    export_weights_binary(ml.echo_value_model, 'echo_value_model.esml')
)";
    }

    return script.str();
}

} // namespace ml
} // namespace echosurf
