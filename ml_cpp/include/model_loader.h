/**
 * EchoSurf ML Framework - Model Loader
 *
 * Loads model weights from various formats:
 * - NumPy .npz files (exported from Python)
 * - Raw binary format (for minimal overhead)
 * - JSON format (for human-readable inspection)
 */

#ifndef ECHOSURF_MODEL_LOADER_H
#define ECHOSURF_MODEL_LOADER_H

#include "tensor.h"
#include "model.h"
#include <string>
#include <unordered_map>
#include <fstream>
#include <vector>

namespace echosurf {
namespace ml {

/**
 * Weight format for serialization
 */
enum class WeightFormat {
    Binary,     // Raw float32 binary
    NumPy,      // NumPy .npy/.npz format
    JSON        // JSON with base64-encoded data
};

/**
 * Model checkpoint information
 */
struct ModelCheckpoint {
    std::string model_name;
    std::string version;
    size_t num_layers;
    size_t total_params;
    std::vector<std::string> layer_names;
    std::unordered_map<std::string, TensorShape> weight_shapes;
};

/**
 * Model Loader
 *
 * Handles loading model weights from disk to C++ model objects.
 */
class ModelLoader {
public:
    ModelLoader() = default;

    // Load weights from binary file
    static bool load_binary(const std::string& path,
                           std::unordered_map<std::string, Tensor>& weights);

    // Load weights from NumPy .npz file
    static bool load_npz(const std::string& path,
                        std::unordered_map<std::string, Tensor>& weights);

    // Load weights from JSON file
    static bool load_json(const std::string& path,
                         std::unordered_map<std::string, Tensor>& weights);

    // Load single tensor from .npy file
    static Tensor load_npy(const std::string& path);

    // Save weights to binary file
    static bool save_binary(const std::string& path,
                           const std::unordered_map<std::string, Tensor>& weights);

    // Apply loaded weights to a model
    static bool apply_weights(SequentialModel& model,
                             const std::unordered_map<std::string, Tensor>& weights);

    // Load and apply weights in one step
    static bool load_model(const std::string& path, SequentialModel& model,
                          WeightFormat format = WeightFormat::Binary);

    // Load checkpoint info without loading weights
    static ModelCheckpoint load_checkpoint_info(const std::string& path);

    // Convenience functions for specific models
    static bool load_reflex_model(const std::string& path, ReflexModel& model);
    static bool load_tactical_model(const std::string& path, TacticalModel& model);
    static bool load_echo_value_model(const std::string& path, EchoValueModel& model);

private:
    // Helper to read raw bytes
    static std::vector<char> read_file(const std::string& path);

    // Parse NumPy header
    static bool parse_npy_header(const std::vector<char>& data, size_t& offset,
                                 TensorShape& shape, std::string& dtype);

    // Parse JSON
    static bool parse_json_weights(const std::string& json_str,
                                   std::unordered_map<std::string, Tensor>& weights);
};

/**
 * Python Weight Exporter Script Generator
 *
 * Generates a Python script that can export TensorFlow/Keras
 * model weights to a format loadable by this C++ framework.
 */
class WeightExporterGenerator {
public:
    static std::string generate_export_script(const std::string& model_type);
};

/**
 * Binary Weight Format
 *
 * Header:
 *   - Magic number (4 bytes): "ESML"
 *   - Version (4 bytes): uint32
 *   - Num weights (4 bytes): uint32
 *
 * For each weight:
 *   - Name length (4 bytes): uint32
 *   - Name (variable): string
 *   - Num dims (4 bytes): uint32
 *   - Dims (4 * num_dims bytes): uint32[]
 *   - Data (4 * total_size bytes): float32[]
 */
class BinaryWeightFormat {
public:
    static constexpr uint32_t MAGIC = 0x4C4D5345;  // "ESML"
    static constexpr uint32_t VERSION = 1;

    static bool write(std::ostream& out,
                     const std::unordered_map<std::string, Tensor>& weights);

    static bool read(std::istream& in,
                    std::unordered_map<std::string, Tensor>& weights);
};

} // namespace ml
} // namespace echosurf

#endif // ECHOSURF_MODEL_LOADER_H
