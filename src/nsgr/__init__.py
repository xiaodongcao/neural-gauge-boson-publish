"""NSGR: neural stochastic-gauge representations for quantum many-body systems."""

__version__ = "0.1.0"

from .analytical_gauge import (
    ANALYTICAL_GAUGE_INFOS,
    ANALYTICAL_GAUGE_MODES,
    AnalyticalGaugeInfo,
    describe_analytical_gauge,
)
from .lattice import Lattice
from .model import Model
from .postprocess import (
    centered_exponentiated_weights,
    MeasurementSnapshot,
    MeasurementHistory,
    WeightedComplexHistogram2D,
    WeightedMagnitudeHistogram,
    WindowEndDataset,
    WindowState,
    alpha_site_contributions,
    compute_equal_time_measurements,
    compute_first_moment_observables,
    compute_measurement_history,
    histogram_alpha_contribution,
    load_window_end_dataset,
    self_normalized_weight_ratio,
    weighted_complex_histogram_2d,
    weighted_magnitude_histogram,
    weighted_ratio_mean_complex_with_error,
)
from .simulation import GaugeSimulator, SimulationArtifacts, SimulationPhysics
from .training import GaugeTrainer, TrainingArtifacts
from .training_plot import plot_training_history
from .utility import load_array_archive, load_config


__all__ = [
    "__version__",
    "ANALYTICAL_GAUGE_INFOS",
    "ANALYTICAL_GAUGE_MODES",
    "AnalyticalGaugeInfo",
    "GaugeSimulator",
    "GaugeTrainer",
    "Lattice",
    "MeasurementSnapshot",
    "MeasurementHistory",
    "WeightedComplexHistogram2D",
    "Model",
    "SimulationArtifacts",
    "SimulationPhysics",
    "TrainingArtifacts",
    "WeightedMagnitudeHistogram",
    "WindowEndDataset",
    "WindowState",
    "alpha_site_contributions",
    "centered_exponentiated_weights",
    "compute_equal_time_measurements",
    "compute_first_moment_observables",
    "compute_measurement_history",
    "describe_analytical_gauge",
    "histogram_alpha_contribution",
    "load_array_archive",
    "load_config",
    "load_window_end_dataset",
    "plot_training_history",
    "self_normalized_weight_ratio",
    "weighted_complex_histogram_2d",
    "weighted_magnitude_histogram",
    "weighted_ratio_mean_complex_with_error",
]
