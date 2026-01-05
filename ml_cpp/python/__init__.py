"""
EchoSurf ML C++ Python Bindings

High-performance ML inference for gaming applications.
Provides <10ms reflex inference and <50ms tactical inference.
"""

try:
    from .echosurf_ml_cpp import (
        # Core types
        TensorShape,
        Tensor,
        Activation,
        InferenceMetrics,

        # Layers
        DenseLayer,

        # Models
        SequentialModel,
        ReflexModel,
        TacticalModel,
        EchoValueModel,

        # Loader
        ModelLoader,

        # Operations submodule
        ops,

        # Version info
        __version__,
        SIMD_ENABLED,
    )

    CPP_AVAILABLE = True

except ImportError:
    # C++ module not built - provide fallback info
    CPP_AVAILABLE = False
    __version__ = "1.0.0-python"
    SIMD_ENABLED = False

    class NotBuiltError(Exception):
        """Raised when C++ module is not available."""
        pass

    def _not_available(*args, **kwargs):
        raise NotBuiltError(
            "EchoSurf ML C++ module not built. "
            "Run 'cmake .. && make' in ml_cpp/build directory."
        )

    # Placeholder classes
    TensorShape = _not_available
    Tensor = _not_available
    DenseLayer = _not_available
    SequentialModel = _not_available
    ReflexModel = _not_available
    TacticalModel = _not_available
    EchoValueModel = _not_available
    ModelLoader = _not_available


def is_available():
    """Check if C++ acceleration is available."""
    return CPP_AVAILABLE


def get_info():
    """Get information about the C++ module."""
    return {
        "available": CPP_AVAILABLE,
        "version": __version__,
        "simd_enabled": SIMD_ENABLED,
    }


__all__ = [
    "TensorShape",
    "Tensor",
    "Activation",
    "InferenceMetrics",
    "DenseLayer",
    "SequentialModel",
    "ReflexModel",
    "TacticalModel",
    "EchoValueModel",
    "ModelLoader",
    "ops",
    "is_available",
    "get_info",
    "CPP_AVAILABLE",
    "SIMD_ENABLED",
    "__version__",
]
