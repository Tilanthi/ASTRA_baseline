# Copyright 2026 Glenn J. White
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Next-Generation Astrophysics Capabilities

This module extends STAN with advanced capabilities for next-generation
telescope data analysis and theoretical modeling.

Modules:
- archive_query: VO/TAP interfaces to major astronomical archives
- transient_science: Light curve fitting, classification, transient physics
- astrochemistry: Extended chemical networks, COMs, isotopologue analysis
- disk_physics: Protoplanetary disk structure, planet-disk interaction
- galactic_dynamics: Orbit fitting, stellar streams, chemical evolution
- ml_survey: Machine learning for large survey analysis
- atmospheric_retrieval: Exoplanet atmosphere modeling
- cosmological_context: Galaxy-halo connection, environment effects
- alert_processing: Real-time transient alert stream handling
- radio_astronomy: Radio data processing (ALMA, VLA, LOFAR, MWA)

Date: 2025-12-15
Version: 1.1
"""


def _degraded_warn(_module: str, _exc: BaseException) -> None:
    """Log why an optional import degraded instead of failing silently."""
    import logging
    logging.getLogger(__name__).warning(
        "%s unavailable (%s: %s) - dependent names set to None",
        _module, type(_exc).__name__, _exc,
    )


try:
    from .archive_query import (
    VOQueryEngine,
    TAP_Client,
    AstroqueryInterface,
    CrossMatchEngine,
    ArchiveDataManager,
    # Radio archives
    ALMAArchive,
    NRAOArchive,
    LOFARArchive,
    MWAArchive,
    ESOArchive,
    RadioArchiveManager,
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".archive_query", _exc)
    VOQueryEngine = TAP_Client = AstroqueryInterface = CrossMatchEngine = ArchiveDataManager = ALMAArchive = NRAOArchive = LOFARArchive = MWAArchive = ESOArchive = RadioArchiveManager = None  # degraded: unavailable

try:
    from .transient_science import (
    TransientClassifier,
    LightCurveFitter,
    SupernovaModels,
    GRBAfterglowModel,
    KilonovaModel,
    TransientAlertBroker
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".transient_science", _exc)
    TransientClassifier = LightCurveFitter = SupernovaModels = GRBAfterglowModel = KilonovaModel = TransientAlertBroker = None  # degraded: unavailable

try:
    from .astrochemistry import (
    ChemicalNetwork,
    UMISTNetwork,
    KIDANetwork,
    GrainSurfaceChemistry,
    IsotopologueAnalyzer,
    COMFormationModel,
    DeuteriumFractionation
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".astrochemistry", _exc)
    ChemicalNetwork = UMISTNetwork = KIDANetwork = GrainSurfaceChemistry = IsotopologueAnalyzer = COMFormationModel = DeuteriumFractionation = None  # degraded: unavailable

try:
    from .disk_physics import (
    ProtoplanetaryDisk,
    DiskEvolutionModel,
    GapOpeningCriteria,
    DustGrainEvolution,
    DiskDispersalModel,
    PlanetDiskInteraction
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".disk_physics", _exc)
    ProtoplanetaryDisk = DiskEvolutionModel = GapOpeningCriteria = DustGrainEvolution = DiskDispersalModel = PlanetDiskInteraction = None  # degraded: unavailable

try:
    from .galactic_dynamics import (
    GalacticPotential,
    OrbitIntegrator,
    StellarStreamFinder,
    ChemicalEvolutionModel,
    ActionAngleCalculator,
    ClusterDissolutionModel
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".galactic_dynamics", _exc)
    GalacticPotential = OrbitIntegrator = StellarStreamFinder = ChemicalEvolutionModel = ActionAngleCalculator = ClusterDissolutionModel = None  # degraded: unavailable

try:
    from .ml_survey import (
    AnomalyDetector,
    PhotometricRedshiftEstimator,
    SourceClassifier,
    SpectralAutoencoder,
    ActiveLearningSelector
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".ml_survey", _exc)
    AnomalyDetector = PhotometricRedshiftEstimator = SourceClassifier = SpectralAutoencoder = ActiveLearningSelector = None  # degraded: unavailable

try:
    from .atmospheric_retrieval import (
    AtmosphericRetrieval,
    TransmissionSpectrum,
    EmissionSpectrum,
    CloudModel,
    ChemicalEquilibrium
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".atmospheric_retrieval", _exc)
    AtmosphericRetrieval = TransmissionSpectrum = EmissionSpectrum = CloudModel = ChemicalEquilibrium = None  # degraded: unavailable

try:
    from .cosmological_context import (
    HaloMassFunction,
    GalaxyHaloConnection,
    EnvironmentalMetrics,
    CGMModel,
    ReionizationModel
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".cosmological_context", _exc)
    HaloMassFunction = GalaxyHaloConnection = EnvironmentalMetrics = CGMModel = ReionizationModel = None  # degraded: unavailable

try:
    from .alert_processing import (
    AlertStreamProcessor,
    ZTFAlertHandler,
    RubinAlertHandler,
    AlertFilterPipeline,
    FollowUpPrioritizer
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".alert_processing", _exc)
    AlertStreamProcessor = ZTFAlertHandler = RubinAlertHandler = AlertFilterPipeline = FollowUpPrioritizer = None  # degraded: unavailable

try:
    from .radio_astronomy import (
    RadioFacility,
    ObservingBand,
    RadioObservation,
    Visibility,
    RadioSource,
    FacilitySpecs,
    RadioContinuumAnalysis,
    RadioSpectralLine,
    RadioInterferometry,
    LowFrequencyRadio,
    RadioPolarization,
    RadioSourcePhysics,
    RadioArchiveInterface,
    jy_to_kelvin,
    kelvin_to_jy,
    freq_to_wavelength,
    wavelength_to_freq,
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".radio_astronomy", _exc)
    RadioFacility = ObservingBand = RadioObservation = Visibility = RadioSource = FacilitySpecs = RadioContinuumAnalysis = RadioSpectralLine = RadioInterferometry = LowFrequencyRadio = RadioPolarization = RadioSourcePhysics = RadioArchiveInterface = jy_to_kelvin = kelvin_to_jy = freq_to_wavelength = wavelength_to_freq = None  # degraded: unavailable

__all__ = [
    # Archive Query
    'VOQueryEngine', 'TAP_Client', 'AstroqueryInterface',
    'CrossMatchEngine', 'ArchiveDataManager',
    'ALMAArchive', 'NRAOArchive', 'LOFARArchive', 'MWAArchive',
    'ESOArchive', 'RadioArchiveManager',

    # Transient Science
    'TransientClassifier', 'LightCurveFitter', 'SupernovaModels',
    'GRBAfterglowModel', 'KilonovaModel', 'TransientAlertBroker',

    # Astrochemistry
    'ChemicalNetwork', 'UMISTNetwork', 'KIDANetwork',
    'GrainSurfaceChemistry', 'IsotopologueAnalyzer',
    'COMFormationModel', 'DeuteriumFractionation',

    # Disk Physics
    'ProtoplanetaryDisk', 'DiskEvolutionModel', 'GapOpeningCriteria',
    'DustGrainEvolution', 'DiskDispersalModel', 'PlanetDiskInteraction',

    # Galactic Dynamics
    'GalacticPotential', 'OrbitIntegrator', 'StellarStreamFinder',
    'ChemicalEvolutionModel', 'ActionAngleCalculator', 'ClusterDissolutionModel',

    # ML Survey
    'AnomalyDetector', 'PhotometricRedshiftEstimator', 'SourceClassifier',
    'SpectralAutoencoder', 'ActiveLearningSelector',

    # Atmospheric Retrieval
    'AtmosphericRetrieval', 'TransmissionSpectrum', 'EmissionSpectrum',
    'CloudModel', 'ChemicalEquilibrium',

    # Cosmological Context
    'HaloMassFunction', 'GalaxyHaloConnection', 'EnvironmentalMetrics',
    'CGMModel', 'ReionizationModel',

    # Alert Processing
    'AlertStreamProcessor', 'ZTFAlertHandler', 'RubinAlertHandler',
    'AlertFilterPipeline', 'FollowUpPrioritizer',

    # Radio Astronomy
    'RadioFacility', 'ObservingBand', 'RadioObservation', 'Visibility', 'RadioSource',
    'FacilitySpecs', 'RadioContinuumAnalysis', 'RadioSpectralLine',
    'RadioInterferometry', 'LowFrequencyRadio', 'RadioPolarization',
    'RadioSourcePhysics', 'RadioArchiveInterface',
    'jy_to_kelvin', 'kelvin_to_jy', 'freq_to_wavelength', 'wavelength_to_freq',
]

__version__ = '1.1'


