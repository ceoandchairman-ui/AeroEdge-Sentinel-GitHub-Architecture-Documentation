"""Edge inference engine for anomaly detection in spacecraft telemetry.

This module exposes `InferenceEngine`, which can run inference against an ONNX
model (if available) or a deterministic mock model for development/HIL testing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Deque, Dict, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    """Inference output for a telemetry frame."""

    reconstruction_error: float
    threshold: float
    is_anomaly: bool


class MockQuantizedLSTMAutoencoder:
    """Development fallback that mimics a quantized LSTM autoencoder output.

    The mock model performs a simple normalization and denormalization pass that
    produces non-zero reconstruction error under drifted/anomalous inputs.
    """

    def __init__(self, feature_dim: int) -> None:
        self._feature_dim = feature_dim
        self._running_mean = np.zeros(feature_dim, dtype=np.float32)
        self._initialized = False

    def reconstruct(self, features: np.ndarray) -> np.ndarray:
        if features.shape[-1] != self._feature_dim:
            raise ValueError(
                f"Expected feature dimension {self._feature_dim}, got {features.shape[-1]}"
            )

        if not self._initialized:
            self._running_mean = features.astype(np.float32)
            self._initialized = True
        else:
            self._running_mean = 0.98 * self._running_mean + 0.02 * features

        # Simulate quantization effects and imperfect reconstruction.
        normalized = features - self._running_mean
        quantized = np.round(normalized * 32.0) / 32.0
        reconstructed = self._running_mean + 0.94 * quantized
        return reconstructed.astype(np.float32)


class InferenceEngine:
    """Runs anomaly detection using an ONNX model with adaptive thresholding."""

    def __init__(
        self,
        model_path: str | Path = "models/lstm_autoencoder_int8.onnx",
        warmup_samples: int = 40,
        sigma_multiplier: float = 3.0,
    ) -> None:
        self._model_path = Path(model_path)
        self._warmup_samples = warmup_samples
        self._sigma_multiplier = sigma_multiplier
        self._feature_dim = 7
        self._errors: Deque[float] = deque(maxlen=max(200, warmup_samples * 4))

        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._mock_model = MockQuantizedLSTMAutoencoder(feature_dim=self._feature_dim)

        self._initialize_runtime()

    def _initialize_runtime(self) -> None:
        """Initialize ONNX Runtime session when a model artifact is available."""
        if not self._model_path.exists():
            LOGGER.warning(
                "ONNX model not found at %s; using mock quantized autoencoder.",
                self._model_path,
            )
            return

        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self._model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        LOGGER.info("Initialized ONNX Runtime with model: %s", self._model_path)

    def infer(self, telemetry_vector: Sequence[float]) -> InferenceResult:
        """Infer anomaly status for a single telemetry vector.

        Expected telemetry order:
        [imu_x, imu_y, imu_z, thermal_board, thermal_battery, bus_voltage, bus_current]
        """
        features = np.asarray(telemetry_vector, dtype=np.float32)
        if features.shape != (self._feature_dim,):
            raise ValueError(
                f"Expected telemetry vector shape ({self._feature_dim},), got {features.shape}"
            )

        reconstructed = self._run_model(features)
        error = float(np.mean(np.square(features - reconstructed)))

        self._errors.append(error)
        threshold = self._compute_threshold()
        is_anomaly = len(self._errors) >= self._warmup_samples and error > threshold

        return InferenceResult(
            reconstruction_error=error,
            threshold=threshold,
            is_anomaly=is_anomaly,
        )

    def _run_model(self, features: np.ndarray) -> np.ndarray:
        """Run ONNX inference if available, otherwise use mock reconstruction."""
        if self._session is None or self._input_name is None or self._output_name is None:
            return self._mock_model.reconstruct(features)

        # Assume model input shape [batch, time, features] and output with matching features.
        model_input = features.reshape(1, 1, self._feature_dim).astype(np.float32)
        outputs = self._session.run([self._output_name], {self._input_name: model_input})
        reconstructed = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if reconstructed.shape[0] != self._feature_dim:
            raise ValueError(
                f"Unexpected model output shape {reconstructed.shape}; expected feature dimension {self._feature_dim}"
            )
        return reconstructed

    def _compute_threshold(self) -> float:
        if not self._errors:
            return float("inf")
        values = np.asarray(self._errors, dtype=np.float32)
        mean = float(np.mean(values))
        std = float(np.std(values))
        if len(values) < self._warmup_samples:
            return mean + (self._sigma_multiplier + 1.0) * max(std, 1e-6)
        return mean + self._sigma_multiplier * max(std, 1e-6)

    def metrics(self) -> Dict[str, float]:
        """Expose current detector metrics for observability hooks."""
        if not self._errors:
            return {"mean_error": 0.0, "std_error": 0.0, "samples": 0.0}

        values = np.asarray(self._errors, dtype=np.float32)
        return {
            "mean_error": float(np.mean(values)),
            "std_error": float(np.std(values)),
            "samples": float(len(values)),
        }


def default_logger() -> None:
    """Set module-level default logging format for quick starts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


if __name__ == "__main__":
    default_logger()
    engine = InferenceEngine()
    sample = [0.01, -0.02, 0.98, 42.0, 36.0, 28.5, 1.4]
    result = engine.infer(sample)
    LOGGER.info(
        "error=%.6f threshold=%.6f anomaly=%s",
        result.reconstruction_error,
        result.threshold,
        result.is_anomaly,
    )
