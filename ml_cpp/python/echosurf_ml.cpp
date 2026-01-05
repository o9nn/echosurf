/**
 * EchoSurf ML Framework - Python Bindings
 *
 * Provides Python interface to C++ inference engine.
 * Uses pybind11 for seamless integration.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../include/tensor.h"
#include "../include/layers.h"
#include "../include/model.h"
#include "../include/model_loader.h"

namespace py = pybind11;
using namespace echosurf::ml;

/**
 * Convert NumPy array to Tensor
 */
Tensor numpy_to_tensor(py::array_t<float> arr) {
    py::buffer_info buf = arr.request();

    std::vector<size_t> dims;
    for (auto& d : buf.shape) {
        dims.push_back(static_cast<size_t>(d));
    }

    TensorShape shape(dims);
    return Tensor(shape, static_cast<float*>(buf.ptr));
}

/**
 * Convert Tensor to NumPy array
 */
py::array_t<float> tensor_to_numpy(const Tensor& tensor) {
    std::vector<ssize_t> shape;
    for (size_t i = 0; i < tensor.shape().ndim(); ++i) {
        shape.push_back(static_cast<ssize_t>(tensor.shape()[i]));
    }

    py::array_t<float> result(shape);
    py::buffer_info buf = result.request();
    std::memcpy(buf.ptr, tensor.data(), tensor.size() * sizeof(float));

    return result;
}

PYBIND11_MODULE(echosurf_ml_cpp, m) {
    m.doc() = "EchoSurf ML C++ Inference Engine";

    // ========================================================================
    // TensorShape
    // ========================================================================
    py::class_<TensorShape>(m, "TensorShape")
        .def(py::init<>())
        .def(py::init<std::vector<size_t>>())
        .def("ndim", &TensorShape::ndim)
        .def("size", &TensorShape::size)
        .def("__getitem__", [](const TensorShape& s, size_t i) { return s[i]; })
        .def("__repr__", &TensorShape::to_string);

    // ========================================================================
    // Tensor
    // ========================================================================
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const TensorShape&>())
        .def(py::init<const TensorShape&, float>())
        .def_static("from_numpy", &numpy_to_tensor)
        .def("to_numpy", &tensor_to_numpy)
        .def("shape", &Tensor::shape)
        .def("size", &Tensor::size)
        .def("ndim", &Tensor::ndim)
        .def("__getitem__", [](const Tensor& t, size_t i) { return t[i]; })
        .def("__setitem__", [](Tensor& t, size_t i, float v) { t[i] = v; })
        .def("zero", &Tensor::zero)
        .def("fill", &Tensor::fill)
        .def("copy", &Tensor::copy)
        .def("reshape", &Tensor::reshape);

    // ========================================================================
    // Activation enum
    // ========================================================================
    py::enum_<Activation>(m, "Activation")
        .value("None_", Activation::None)
        .value("ReLU", Activation::ReLU)
        .value("Sigmoid", Activation::Sigmoid)
        .value("Tanh", Activation::Tanh)
        .value("Softmax", Activation::Softmax)
        .value("LeakyReLU", Activation::LeakyReLU);

    // ========================================================================
    // DenseLayer
    // ========================================================================
    py::class_<DenseLayer>(m, "DenseLayer")
        .def(py::init<size_t, size_t, Activation, bool>(),
             py::arg("input_features"),
             py::arg("output_features"),
             py::arg("activation") = Activation::None,
             py::arg("use_bias") = true)
        .def("forward", [](DenseLayer& layer, py::array_t<float> input) {
            Tensor in_tensor = numpy_to_tensor(input);
            Tensor out_tensor;
            layer.forward(in_tensor, out_tensor);
            return tensor_to_numpy(out_tensor);
        })
        .def("set_kernel", [](DenseLayer& layer, py::array_t<float> kernel) {
            layer.set_kernel(numpy_to_tensor(kernel));
        })
        .def("set_bias", [](DenseLayer& layer, py::array_t<float> bias) {
            layer.set_bias(numpy_to_tensor(bias));
        })
        .def("param_count", &DenseLayer::param_count);

    // ========================================================================
    // InferenceMetrics
    // ========================================================================
    py::class_<InferenceMetrics>(m, "InferenceMetrics")
        .def(py::init<>())
        .def_readonly("last_inference_ms", &InferenceMetrics::last_inference_ms)
        .def_readonly("avg_inference_ms", &InferenceMetrics::avg_inference_ms)
        .def_readonly("min_inference_ms", &InferenceMetrics::min_inference_ms)
        .def_readonly("max_inference_ms", &InferenceMetrics::max_inference_ms)
        .def_readonly("total_inferences", &InferenceMetrics::total_inferences)
        .def("reset", &InferenceMetrics::reset);

    // ========================================================================
    // SequentialModel
    // ========================================================================
    py::class_<SequentialModel>(m, "SequentialModel")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def("add_dense", [](SequentialModel& model, size_t in_features,
                            size_t out_features, Activation activation) {
            model.add<DenseLayer>(in_features, out_features, activation);
        }, py::arg("in_features"), py::arg("out_features"),
           py::arg("activation") = Activation::None)
        .def("add_dropout", [](SequentialModel& model, float rate) {
            model.add<DropoutLayer>(rate);
        })
        .def("forward", [](SequentialModel& model, py::array_t<float> input) {
            Tensor in_tensor = numpy_to_tensor(input);
            Tensor out_tensor = model.forward(in_tensor);
            return tensor_to_numpy(out_tensor);
        })
        .def("predict", [](SequentialModel& model, py::array_t<float> input) {
            Tensor in_tensor = numpy_to_tensor(input);
            Tensor out_tensor = model.predict(in_tensor);
            return tensor_to_numpy(out_tensor);
        })
        .def("name", &SequentialModel::name)
        .def("num_layers", &SequentialModel::num_layers)
        .def("total_params", &SequentialModel::total_params)
        .def("metrics", &SequentialModel::metrics)
        .def("reset_metrics", &SequentialModel::reset_metrics)
        .def("summary", &SequentialModel::summary);

    // ========================================================================
    // ReflexModel
    // ========================================================================
    py::class_<ReflexModel>(m, "ReflexModel")
        .def(py::init<>())
        .def("predict", [](ReflexModel& model, py::array_t<float> input) {
            Tensor in_tensor = numpy_to_tensor(input);
            Tensor out_tensor = model.forward(in_tensor);
            return tensor_to_numpy(out_tensor);
        })
        .def("predict_action", [](ReflexModel& model,
                                  float threat_proximity,
                                  float threat_direction,
                                  float player_state,
                                  float movement_momentum,
                                  float time_pressure,
                                  float cover_availability,
                                  float aim_confidence,
                                  float situation_clarity) {
            ReflexModel::ReflexInput input{
                threat_proximity, threat_direction, player_state,
                movement_momentum, time_pressure, cover_availability,
                aim_confidence, situation_clarity
            };
            return static_cast<int>(model.predict(input));
        })
        .def("model", py::overload_cast<>(&ReflexModel::model),
             py::return_value_policy::reference)
        .def("meets_latency_target", &ReflexModel::meets_latency_target)
        .def_property_readonly_static("LATENCY_TARGET_MS",
            [](py::object) { return ReflexModel::LATENCY_TARGET_MS; });

    // ========================================================================
    // TacticalModel
    // ========================================================================
    py::class_<TacticalModel>(m, "TacticalModel")
        .def(py::init<>())
        .def("predict", [](TacticalModel& model, py::array_t<float> input) {
            Tensor in_tensor = numpy_to_tensor(input);
            Tensor out_tensor = model.forward(in_tensor);
            return tensor_to_numpy(out_tensor);
        })
        .def("predict_action", [](TacticalModel& model,
                                  float threat_level, float health,
                                  float ammo, float armor,
                                  float items, float currency,
                                  float pos_x, float pos_y, float pos_z,
                                  float ally_strength, float ally_distance,
                                  float ally_health, float enemy_strength,
                                  float enemy_distance, float enemy_count,
                                  float objective_distance) {
            TacticalModel::TacticalInput input{
                threat_level, health, ammo, armor, items, currency,
                pos_x, pos_y, pos_z,
                ally_strength, ally_distance, ally_health,
                enemy_strength, enemy_distance, enemy_count,
                objective_distance
            };
            return static_cast<int>(model.predict(input));
        })
        .def("model", py::overload_cast<>(&TacticalModel::model),
             py::return_value_policy::reference)
        .def("meets_latency_target", &TacticalModel::meets_latency_target)
        .def_property_readonly_static("LATENCY_TARGET_MS",
            [](py::object) { return TacticalModel::LATENCY_TARGET_MS; });

    // ========================================================================
    // EchoValueModel
    // ========================================================================
    py::class_<EchoValueModel>(m, "EchoValueModel")
        .def(py::init<>())
        .def("predict", [](EchoValueModel& model, py::array_t<float> input) {
            Tensor in_tensor = numpy_to_tensor(input);
            return model.forward(in_tensor)[0];
        })
        .def("predict_value", [](EchoValueModel& model,
                                 float content_length, float complexity,
                                 float depth, float child_count,
                                 float sibling_count, float historical_value) {
            EchoValueModel::EchoInput input{
                content_length, complexity, depth,
                child_count, sibling_count, historical_value
            };
            return model.predict(input);
        })
        .def("model", py::overload_cast<>(&EchoValueModel::model),
             py::return_value_policy::reference);

    // ========================================================================
    // ModelLoader
    // ========================================================================
    py::class_<ModelLoader>(m, "ModelLoader")
        .def_static("load_binary", &ModelLoader::load_binary)
        .def_static("load_npz", &ModelLoader::load_npz)
        .def_static("save_binary", &ModelLoader::save_binary)
        .def_static("load_reflex_model", &ModelLoader::load_reflex_model)
        .def_static("load_tactical_model", &ModelLoader::load_tactical_model)
        .def_static("load_echo_value_model", &ModelLoader::load_echo_value_model);

    // ========================================================================
    // Tensor operations module
    // ========================================================================
    py::module_ ops_m = m.def_submodule("ops", "Tensor operations");

    ops_m.def("matmul", [](py::array_t<float> a, py::array_t<float> b) {
        Tensor ta = numpy_to_tensor(a);
        Tensor tb = numpy_to_tensor(b);
        Tensor tc;
        ops::matmul(ta, tb, tc);
        return tensor_to_numpy(tc);
    });

    ops_m.def("relu", [](py::array_t<float> x) {
        Tensor tx = numpy_to_tensor(x);
        Tensor out;
        ops::relu(tx, out);
        return tensor_to_numpy(out);
    });

    ops_m.def("sigmoid", [](py::array_t<float> x) {
        Tensor tx = numpy_to_tensor(x);
        Tensor out;
        ops::sigmoid(tx, out);
        return tensor_to_numpy(out);
    });

    ops_m.def("softmax", [](py::array_t<float> x) {
        Tensor tx = numpy_to_tensor(x);
        Tensor out;
        ops::softmax(tx, out);
        return tensor_to_numpy(out);
    });

    ops_m.def("sum", [](py::array_t<float> x) {
        return ops::sum(numpy_to_tensor(x));
    });

    ops_m.def("mean", [](py::array_t<float> x) {
        return ops::mean(numpy_to_tensor(x));
    });

    ops_m.def("max", [](py::array_t<float> x) {
        return ops::max(numpy_to_tensor(x));
    });

    ops_m.def("min", [](py::array_t<float> x) {
        return ops::min(numpy_to_tensor(x));
    });

    // ========================================================================
    // Version info
    // ========================================================================
    m.attr("__version__") = "1.0.0";
    m.attr("SIMD_ENABLED") =
#ifdef ECHOSURF_SIMD_AVX2
        true;
#else
        false;
#endif
}
