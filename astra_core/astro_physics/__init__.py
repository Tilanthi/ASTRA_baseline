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
ASTRO-SWARM: Astronomical Inference via Stigmergic Swarm Intelligence

A reconfiguration of the V36-Swarm-MORK-Graph system for astronomical
and astrophysical problem solving.

Key capabilities beyond conventional LLMs:
1. Physics-constrained causal inference
2. Multi-agent parallel hypothesis exploration
3. Persistent knowledge accumulation across sessions
4. Cross-domain analogical reasoning
5. Mechanism discovery from observational data

Extended capabilities (v2.0):
6. Radiative transfer and non-LTE excitation
7. FITS/spectral cube data handling
8. Bayesian inference and MCMC sampling
9. Multi-wavelength SED fitting
10. Astrochemical network modeling
11. Interferometric imaging (UV, CLEAN, self-cal)
12. Advanced gravitational lensing
13. MHD turbulence analysis
14. Spectroscopic database access (CDMS, JPL, LAMDA, Splatalogue)
15. Multi-scale simulation coupling

V43 additions:
16. Core ISM physics (gravitational collapse, shocks, HII regions, SNRs)
17. Data analysis infrastructure (line fitting, source extraction, kinematics)

V44 additions:
18. Radio astronomy surveys and source detection
19. Star formation and stellar evolution (IMF, supernovae, feedback)
20. SPH gas dynamics and molecular cloud formation
21. Infrared and submillimeter astronomy
22. Time series and power spectrum analysis
23. Data cube and spectrum visualization

Date: 2025-12-23
Version: 3.1.0 (V44)
"""

# Core components (using relative imports)

def _degraded_warn(_module: str, _exc: BaseException) -> None:
    """Log why an optional import degraded instead of failing silently."""
    import logging
    logging.getLogger(__name__).warning(
        "%s unavailable (%s: %s) - dependent names set to None",
        _module, type(_exc).__name__, _exc,
    )


try:
    from .core import AstroSwarmSystem
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".core", _exc)
    AstroSwarmSystem = None  # degraded: unavailable
try:
    from .physics import PhysicsEngine, AstrophysicalConstraints
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".physics", _exc)
    PhysicsEngine = AstrophysicalConstraints = None  # degraded: unavailable
try:
    from .knowledge_graph import AstronomicalKnowledgeGraph
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".knowledge_graph", _exc)
    AstronomicalKnowledgeGraph = None  # degraded: unavailable
try:
    from .inference import BayesianSwarmInference
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".inference", _exc)
    BayesianSwarmInference = None  # degraded: unavailable
try:
    from .agents import AstroAgent, SpectroscopicAgent, PhotometricAgent, DynamicalAgent
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".agents", _exc)
    AstroAgent = SpectroscopicAgent = PhotometricAgent = DynamicalAgent = None  # degraded: unavailable

# Extended capability modules
try:
    from .radiative_transfer import (
    StatisticalEquilibriumSolver, LineProfileSynthesizer,
    DustContinuumRT, PDRInterface
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".radiative_transfer", _exc)
    StatisticalEquilibriumSolver = LineProfileSynthesizer = DustContinuumRT = PDRInterface = None  # degraded: unavailable
try:
    from .data_interface import (
    FITSHandler, SpectralCubeHandler, VOTableHandler,
    RegionHandler, CASAInterface
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".data_interface", _exc)
    FITSHandler = SpectralCubeHandler = VOTableHandler = RegionHandler = CASAInterface = None  # degraded: unavailable
try:
    from .uncertainty_quantification import (
    PriorSet, GaussianLikelihood, MetropolisHastings,
    EnsembleSampler, NestedSampler, FisherMatrix
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".uncertainty_quantification", _exc)
    PriorSet = GaussianLikelihood = MetropolisHastings = EnsembleSampler = NestedSampler = FisherMatrix = None  # degraded: unavailable
try:
    from .sed_fitting import (
    FilterLibrary, ModifiedBlackbody, StellarPopulation,
    AGNTemplate, CompositeSED, SEDFitter
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".sed_fitting", _exc)
    FilterLibrary = ModifiedBlackbody = StellarPopulation = AGNTemplate = CompositeSED = SEDFitter = None  # degraded: unavailable
try:
    from .chemical_networks import (
    ReactionNetwork, ChemistrySolver, PDRChemistry,
    GrainChemistry, HotCoreChemistry
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".chemical_networks", _exc)
    ReactionNetwork = ChemistrySolver = PDRChemistry = GrainChemistry = HotCoreChemistry = None  # degraded: unavailable
try:
    from .interferometry import (
    ArrayConfiguration, UVSimulator, Imager,
    CLEANDeconvolver, SelfCalibrator, VisibilityModeler
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".interferometry", _exc)
    ArrayConfiguration = UVSimulator = Imager = CLEANDeconvolver = SelfCalibrator = VisibilityModeler = None  # degraded: unavailable
try:
    from .advanced_lensing import (
    Cosmology, SIEProfile, NFWProfile, CompositeLensModel,
    TimeDelayCosmography, SubstructureDetector
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".advanced_lensing", _exc)
    Cosmology = SIEProfile = NFWProfile = CompositeLensModel = TimeDelayCosmography = SubstructureDetector = None  # degraded: unavailable
try:
    from .turbulence_analysis import (
    StructureFunctionAnalysis, PowerSpectrumAnalysis,
    VelocityAnalysis, SpectralPCA, DavisChandrasekharFermi,
    HistogramRelativeOrientations, TurbulenceStatistics
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".turbulence_analysis", _exc)
    StructureFunctionAnalysis = PowerSpectrumAnalysis = VelocityAnalysis = SpectralPCA = DavisChandrasekharFermi = HistogramRelativeOrientations = TurbulenceStatistics = None  # degraded: unavailable
try:
    from .spectroscopic_databases import (
    CDMSDatabase, JPLDatabase, LAMDADatabase,
    SplatalogueInterface, HITRANDatabase, UnifiedSpectroscopyQuery,
    SpectralLine, MoleculeData, CollisionPartner
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".spectroscopic_databases", _exc)
    CDMSDatabase = JPLDatabase = LAMDADatabase = SplatalogueInterface = HITRANDatabase = UnifiedSpectroscopyQuery = SpectralLine = MoleculeData = CollisionPartner = None  # degraded: unavailable
try:
    from .multiscale_coupling import (
    MultiScaleSimulation, ZoomRegion, ScaleCoupler,
    TurbulentPressureModel, StarFormationModel,
    StellarFeedbackModel, AGNFeedbackModel,
    CoolingFunction, HierarchicalRefinement
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".multiscale_coupling", _exc)
    MultiScaleSimulation = ZoomRegion = ScaleCoupler = TurbulentPressureModel = StarFormationModel = StellarFeedbackModel = AGNFeedbackModel = CoolingFunction = HierarchicalRefinement = None  # degraded: unavailable

# V43: Core ISM Physics
try:
    from .gravitational_collapse import (
    JeansAnalysis, VirialAnalysis, FreefallCollapse,
    FragmentationCriterion, AccretionRates,
    get_jeans_analyzer, get_virial_analyzer
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".gravitational_collapse", _exc)
    JeansAnalysis = VirialAnalysis = FreefallCollapse = FragmentationCriterion = AccretionRates = get_jeans_analyzer = get_virial_analyzer = None  # degraded: unavailable
try:
    from .shock_physics import (
    RankineHugoniot, JShock, CShock, ShockChemistry,
    OutflowShockAnalysis, get_shock_chemistry
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".shock_physics", _exc)
    RankineHugoniot = JShock = CShock = ShockChemistry = OutflowShockAnalysis = get_shock_chemistry = None  # degraded: unavailable
try:
    from .hii_region_physics import (
    StromgrenSphere, NebularDiagnosticsCalculator,
    RecombinationLines, FreeFreeEmission,
    stromgren_radius, get_diagnostics_calculator
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".hii_region_physics", _exc)
    StromgrenSphere = NebularDiagnosticsCalculator = RecombinationLines = FreeFreeEmission = stromgren_radius = get_diagnostics_calculator = None  # degraded: unavailable
try:
    from .supernova_remnant_physics import (
    SedovTaylorBlastwave, SNREvolution, SynchrotronEmission,
    XRayThermalEmission, get_snr_evolution
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".supernova_remnant_physics", _exc)
    SedovTaylorBlastwave = SNREvolution = SynchrotronEmission = XRayThermalEmission = get_snr_evolution = None  # degraded: unavailable

# V43: Data Analysis Infrastructure
try:
    from .spectral_line_analysis import (
    GaussianLineFitter, VoigtProfileFitter, HyperfineStructureFitter,
    LineIdentifier, OpticalDepthCorrector, ColumnDensityCalculator,
    fit_gaussian_line, identify_line
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".spectral_line_analysis", _exc)
    GaussianLineFitter = VoigtProfileFitter = HyperfineStructureFitter = LineIdentifier = OpticalDepthCorrector = ColumnDensityCalculator = fit_gaussian_line = identify_line = None  # degraded: unavailable
try:
    from .source_extraction import (
    SourceDetector, AperturePhotometry, PSFPhotometry,
    DendrogramExtractor, FilamentFinder, CoreCatalogBuilder,
    detect_sources, extract_dendrogram, find_filaments
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".source_extraction", _exc)
    SourceDetector = AperturePhotometry = PSFPhotometry = DendrogramExtractor = FilamentFinder = CoreCatalogBuilder = detect_sources = extract_dendrogram = find_filaments = None  # degraded: unavailable
try:
    from .kinematic_analysis import (
    MomentMapGenerator, PVDiagramExtractor, RotationCurveAnalyzer,
    InfallSignatureDetector, OutflowAnalyzer, TurbulentFieldDecomposer,
    make_moment_maps, extract_pv_diagram, detect_infall
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".kinematic_analysis", _exc)
    MomentMapGenerator = PVDiagramExtractor = RotationCurveAnalyzer = InfallSignatureDetector = OutflowAnalyzer = TurbulentFieldDecomposer = make_moment_maps = extract_pv_diagram = detect_infall = None  # degraded: unavailable

# V44: Extended Astrophysics Capabilities
try:
    from .radio_surveys import (
    RadioSource, SurveyCatalog, RadioSourceType, SurveyType,
    RadioSurveyAnalyzer, VariabilityAnalyzer,
    create_analyzer, get_cross_match_tolerance, load_survey_catalog, estimate_luminosity
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".radio_surveys", _exc)
    RadioSource = SurveyCatalog = RadioSourceType = SurveyType = RadioSurveyAnalyzer = VariabilityAnalyzer = create_analyzer = get_cross_match_tolerance = load_survey_catalog = estimate_luminosity = None  # degraded: unavailable
try:
    from .star_formation import (
    StellarPhase, RemnantType, SFTRindicator,
    StellarPopulation, Star,
    InitialMassFunction, StarFormationLaw, StarFormationRateTracer,
    StellarEvolution, SupernovaFeedback,
    create_stellar_population, sample_masses_from_imf, calculate_sfr_from_luminosity
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".star_formation", _exc)
    StellarPhase = RemnantType = SFTRindicator = StellarPopulation = Star = InitialMassFunction = StarFormationLaw = StarFormationRateTracer = StellarEvolution = SupernovaFeedback = create_stellar_population = sample_masses_from_imf = calculate_sfr_from_luminosity = None  # degraded: unavailable
try:
    from .sph_gas_dynamics import (
    SPHParticle, SPHKernel, KernelType, Filament,
    SPHSimulation, FilamentFinder, MolecularCloudFormation,
    TurbulentDriver, GravitySolver,
    create_sph_simulation, find_filaments_in_data, get_h2_fraction
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".sph_gas_dynamics", _exc)
    SPHParticle = SPHKernel = KernelType = Filament = SPHSimulation = FilamentFinder = MolecularCloudFormation = TurbulentDriver = GravitySolver = create_sph_simulation = find_filaments_in_data = get_h2_fraction = None  # degraded: unavailable
try:
    from .infrared_submm import (
    IRBand, PAHFeature,
    IRPhotometry, PAHSpectrum, DustProperties,
    ModifiedBlackbody, IRColorAnalysis, SubmillimeterAnalysis, LineCooling,
    fit_dust_sed, calculate_gas_mass, get_ir_color
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".infrared_submm", _exc)
    IRBand = PAHFeature = IRPhotometry = PAHSpectrum = DustProperties = ModifiedBlackbody = IRColorAnalysis = SubmillimeterAnalysis = LineCooling = fit_dust_sed = calculate_gas_mass = get_ir_color = None  # degraded: unavailable
try:
    from .time_series_analysis import (
    SignalType, TimeSeries, PeriodogramResult,
    PowerSpectrumAnalyzer, VariabilityDetector, WaveletAnalyzer,
    CrossCorrelationAnalyzer, BurstDetector,
    analyze_power_spectrum, detect_periodicity, compute_structure_function, cross_correlate_series
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".time_series_analysis", _exc)
    SignalType = TimeSeries = PeriodogramResult = PowerSpectrumAnalyzer = VariabilityDetector = WaveletAnalyzer = CrossCorrelationAnalyzer = BurstDetector = analyze_power_spectrum = detect_periodicity = compute_structure_function = cross_correlate_series = None  # degraded: unavailable
try:
    from .data_visualization import (
    VisualizationType, DataCube, Spectrum,
    CubeVisualizer, SpectrumVisualizer, MultiPanelFigure,
    create_moment_map_cube, plot_spectrum
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".data_visualization", _exc)
    VisualizationType = DataCube = Spectrum = CubeVisualizer = SpectrumVisualizer = MultiPanelFigure = create_moment_map_cube = plot_spectrum = None  # degraded: unavailable

# V45: Deep Learning Integration (Phase 1)
try:
    from .deep_learning import (
    DLConfig,
    GalaxyMorphologyCNN,
    ISMStructureCNN,
    SpectralAutoencoder,
    LightCurveAutoencoder,
    TimeSeriesTransformer,
    RadiativeTransferPINN,
    StellarStructurePINN,
    CrossModalMatcher,
    train_autoencoder,
    SpectralDataset,
    ImageDataset
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".deep_learning", _exc)
    DLConfig = GalaxyMorphologyCNN = ISMStructureCNN = SpectralAutoencoder = LightCurveAutoencoder = TimeSeriesTransformer = RadiativeTransferPINN = StellarStructurePINN = CrossModalMatcher = train_autoencoder = SpectralDataset = ImageDataset = None  # degraded: unavailable
try:
    from .deep_learning.filament_detector import (
    FilamentDetector,
    VelocityCoherentFilamentDetector,
    FilamentProperties,
    FilamentDetectionHead,
    FilamentEncoder,
    train_filament_detector
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".deep_learning.filament_detector", _exc)
    FilamentDetector = VelocityCoherentFilamentDetector = FilamentProperties = FilamentDetectionHead = FilamentEncoder = train_filament_detector = None  # degraded: unavailable
try:
    from .deep_learning.molecular_cloud_segmenter import (
    MolecularCloudSegmenter,
    VelocityCubeSegmenter,
    CloudProperties,
    CloudPropertyHead,
    MaskRCNNBackbone,
    train_cloud_segmenter
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".deep_learning.molecular_cloud_segmenter", _exc)
    MolecularCloudSegmenter = VelocityCubeSegmenter = CloudProperties = CloudPropertyHead = MaskRCNNBackbone = train_cloud_segmenter = None  # degraded: unavailable
try:
    from .deep_learning.shock_detector import (
    InterstellarShockDetector,
    SpectralLineShockDetector,
    TemporalShockDetector,
    ShockProperties,
    ShockTypeClassifier,
    ShockParameterRegressor,
    train_shock_detector
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".deep_learning.shock_detector", _exc)
    InterstellarShockDetector = SpectralLineShockDetector = TemporalShockDetector = ShockProperties = ShockTypeClassifier = ShockParameterRegressor = train_shock_detector = None  # degraded: unavailable

# V45: Real-Time Processing (Phase 2)
try:
    from .streaming.streaming_alert_processor import (
    StreamingAlertProcessor,
    AlertClassifier,
    AlertPrioritizer,
    AlertMetadata,
    ProcessedAlert,
    AlertSource,
    TransientType,
    create_alert_processor
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".streaming.streaming_alert_processor", _exc)
    StreamingAlertProcessor = AlertClassifier = AlertPrioritizer = AlertMetadata = ProcessedAlert = AlertSource = TransientType = create_alert_processor = None  # degraded: unavailable
try:
    from .streaming.real_time_anomaly_detection import (
    RealTimeAnomalyDetector,
    LightCurveAnomalyDetector,
    SpectralAnomalyDetector,
    AnomalyReport,
    IsolationForestOnline,
    OnlineStandardScaler,
    create_anomaly_detector
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".streaming.real_time_anomaly_detection", _exc)
    RealTimeAnomalyDetector = LightCurveAnomalyDetector = SpectralAnomalyDetector = AnomalyReport = IsolationForestOnline = OnlineStandardScaler = create_anomaly_detector = None  # degraded: unavailable

# V45: Multi-Messenger Joint Inference (Phase 3)
try:
    from .multi_messenger.gw_em_correlation import (
    GWEMCorrelator,
    TemporalCorrelation,
    SpatialCorrelation,
    DistanceConsistency,
    KilonovaModel,
    MultiEpochCorrelation,
    GWTrigger,
    EMCounterpart,
    JointGWEMDetection,
    create_gw_em_correlator
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".multi_messenger.gw_em_correlation", _exc)
    GWEMCorrelator = TemporalCorrelation = SpatialCorrelation = DistanceConsistency = KilonovaModel = MultiEpochCorrelation = GWTrigger = EMCounterpart = JointGWEMDetection = create_gw_em_correlator = None  # degraded: unavailable
try:
    from .multi_messenger.joint_lightcurve_modeling import (
    JointLightCurveFitter,
    GWStrainModel,
    KilonovaLightCurveModel,
    GRBAfterglowModel,
    NeutrinoFluenceModel,
    JointLikelihood,
    MultiMessengerData,
    PhysicalParameters,
    create_joint_fitter
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".multi_messenger.joint_lightcurve_modeling", _exc)
    JointLightCurveFitter = GWStrainModel = KilonovaLightCurveModel = GRBAfterglowModel = NeutrinoFluenceModel = JointLikelihood = MultiMessengerData = PhysicalParameters = create_joint_fitter = None  # degraded: unavailable

# V45: Causal Discovery for Astronomy (Phase 4)
try:
    from ..causal.discovery.astro_causal_discovery import (
    AstroFCI,
    TemporalCausalDiscovery,
    AstronomicalConditionalIndependence,
    CausalGraph,
    create_astro_fci,
    create_temporal_discovery
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("..causal.discovery.astro_causal_discovery", _exc)
    AstroFCI = TemporalCausalDiscovery = AstronomicalConditionalIndependence = CausalGraph = create_astro_fci = create_temporal_discovery = None  # degraded: unavailable

__version__ = "4.0.0"  # V45 - Deep Learning & Multi-Messenger Integration
__all__ = [
    # Core components
    'AstroSwarmSystem',
    'PhysicsEngine',
    'AstrophysicalConstraints',
    'AstronomicalKnowledgeGraph',
    'BayesianSwarmInference',
    'AstroAgent',
    'SpectroscopicAgent',
    'PhotometricAgent',
    'DynamicalAgent',
    # Radiative transfer
    'StatisticalEquilibriumSolver',
    'LineProfileSynthesizer',
    'DustContinuumRT',
    'PDRInterface',
    # Data interface
    'FITSHandler',
    'SpectralCubeHandler',
    'VOTableHandler',
    'RegionHandler',
    'CASAInterface',
    # Uncertainty quantification
    'PriorSet',
    'GaussianLikelihood',
    'MetropolisHastings',
    'EnsembleSampler',
    'NestedSampler',
    'FisherMatrix',
    # SED fitting
    'FilterLibrary',
    'ModifiedBlackbody',
    'StellarPopulation',
    'AGNTemplate',
    'CompositeSED',
    'SEDFitter',
    # Chemical networks
    'ReactionNetwork',
    'ChemistrySolver',
    'PDRChemistry',
    'GrainChemistry',
    'HotCoreChemistry',
    # Interferometry
    'ArrayConfiguration',
    'UVSimulator',
    'Imager',
    'CLEANDeconvolver',
    'SelfCalibrator',
    'VisibilityModeler',
    # Advanced lensing
    'Cosmology',
    'SIEProfile',
    'NFWProfile',
    'CompositeLensModel',
    'TimeDelayCosmography',
    'SubstructureDetector',
    # Turbulence analysis
    'StructureFunctionAnalysis',
    'PowerSpectrumAnalysis',
    'VelocityAnalysis',
    'SpectralPCA',
    'DavisChandrasekharFermi',
    'HistogramRelativeOrientations',
    'TurbulenceStatistics',
    # Spectroscopic databases
    'CDMSDatabase',
    'JPLDatabase',
    'LAMDADatabase',
    'SplatalogueInterface',
    'HITRANDatabase',
    'UnifiedSpectroscopyQuery',
    'SpectralLine',
    'MoleculeData',
    'CollisionPartner',
    # Multi-scale coupling
    'MultiScaleSimulation',
    'ZoomRegion',
    'ScaleCoupler',
    'TurbulentPressureModel',
    'StarFormationModel',
    'StellarFeedbackModel',
    'AGNFeedbackModel',
    'CoolingFunction',
    'HierarchicalRefinement',
    # V43: Core ISM Physics
    'JeansAnalysis',
    'VirialAnalysis',
    'FreefallCollapse',
    'FragmentationCriterion',
    'AccretionRateCalculator',
    'get_jeans_analysis',
    'get_virial_analysis',
    'RankineHugoniot',
    'JShock',
    'CShock',
    'ShockChemistry',
    'OutflowShock',
    'get_shock_chemistry',
    'StromgrenSphere',
    'NebularDiagnosticsCalculator',
    'RecombinationLines',
    'FreeFreeEmission',
    'calculate_stromgren_radius',
    'get_nebular_diagnostics',
    'SedovTaylorBlastwave',
    'SNREvolution',
    'SynchrotronEmission',
    'XRayThermalEmission',
    'get_snr_evolution',
    # V43: Data Analysis Infrastructure
    'GaussianLineFitter',
    'VoigtProfileFitter',
    'HyperfineStructureFitter',
    'LineIdentifier',
    'OpticalDepthCorrector',
    'ColumnDensityCalculator',
    'fit_gaussian',
    'fit_hyperfine',
    'identify_lines',
    'SourceDetector',
    'AperturePhotometry',
    'PSFPhotometry',
    'DendrogramExtractor',
    'FilamentFinder',
    'CoreCatalogBuilder',
    'detect_sources',
    'extract_dendrogram',
    'find_filaments',
    'MomentMapGenerator',
    'PVDiagramExtractor',
    'RotationCurveAnalyzer',
    'InfallSignatureDetector',
    'OutflowAnalyzer',
    'TurbulentFieldDecomposer',
    'make_moment_maps',
    'extract_pv_diagram',
    'detect_infall',
    # V44: Extended Astrophysics Capabilities
    # Radio surveys
    'RadioSource',
    'SurveyCatalog',
    'RadioSourceType',
    'SurveyType',
    'RadioSurveyAnalyzer',
    'VariabilityAnalyzer',
    'create_analyzer',
    'get_cross_match_tolerance',
    'load_survey_catalog',
    'estimate_luminosity',
    # Star formation
    'StellarPhase',
    'RemnantType',
    'SFTRindicator',
    'StellarPopulation',
    'Star',
    'InitialMassFunction',
    'StarFormationLaw',
    'StarFormationRateTracer',
    'StellarEvolution',
    'SupernovaFeedback',
    'create_stellar_population',
    'sample_masses_from_imf',
    'calculate_sfr_from_luminosity',
    # SPH gas dynamics
    'SPHParticle',
    'SPHKernel',
    'KernelType',
    'Filament',
    'SPHSimulation',
    'FilamentFinder',
    'MolecularCloudFormation',
    'TurbulentDriver',
    'GravitySolver',
    'create_sph_simulation',
    'find_filaments_in_data',
    'get_h2_fraction',
    # Infrared/submillimeter
    'IRBand',
    'PAHFeature',
    'IRPhotometry',
    'PAHSpectrum',
    'DustProperties',
    'ModifiedBlackbody',
    'IRColorAnalysis',
    'SubmillimeterAnalysis',
    'LineCooling',
    'fit_dust_sed',
    'calculate_gas_mass',
    'get_ir_color',
    # Time series analysis
    'SignalType',
    'TimeSeries',
    'PeriodogramResult',
    'PowerSpectrumAnalyzer',
    'VariabilityDetector',
    'WaveletAnalyzer',
    'CrossCorrelationAnalyzer',
    'BurstDetector',
    'analyze_power_spectrum',
    'detect_periodicity',
    'compute_structure_function',
    'cross_correlate_series',
    # Data visualization
    'VisualizationType',
    'DataCube',
    'Spectrum',
    'CubeVisualizer',
    'SpectrumVisualizer',
    'MultiPanelFigure',
    'create_moment_map_cube',
    'plot_spectrum',
    # V45: Deep Learning Integration (Phase 1)
    'DLConfig',
    'GalaxyMorphologyCNN',
    'ISMStructureCNN',
    'SpectralAutoencoder',
    'LightCurveAutoencoder',
    'TimeSeriesTransformer',
    'RadiativeTransferPINN',
    'StellarStructurePINN',
    'CrossModalMatcher',
    'train_autoencoder',
    'SpectralDataset',
    'ImageDataset',
    'FilamentDetector',
    'VelocityCoherentFilamentDetector',
    'FilamentProperties',
    'FilamentDetectionHead',
    'FilamentEncoder',
    'train_filament_detector',
    'MolecularCloudSegmenter',
    'VelocityCubeSegmenter',
    'CloudProperties',
    'CloudPropertyHead',
    'MaskRCNNBackbone',
    'train_cloud_segmenter',
    'InterstellarShockDetector',
    'SpectralLineShockDetector',
    'TemporalShockDetector',
    'ShockProperties',
    'ShockTypeClassifier',
    'ShockParameterRegressor',
    'train_shock_detector',
    # V45: Real-Time Processing (Phase 2)
    'StreamingAlertProcessor',
    'AlertClassifier',
    'AlertPrioritizer',
    'AlertMetadata',
    'ProcessedAlert',
    'AlertSource',
    'TransientType',
    'create_alert_processor',
    'RealTimeAnomalyDetector',
    'LightCurveAnomalyDetector',
    'SpectralAnomalyDetector',
    'AnomalyReport',
    'IsolationForestOnline',
    'OnlineStandardScaler',
    'create_anomaly_detector',
    # V45: Multi-Messenger Joint Inference (Phase 3)
    'GWEMCorrelator',
    'TemporalCorrelation',
    'SpatialCorrelation',
    'DistanceConsistency',
    'KilonovaModel',
    'MultiEpochCorrelation',
    'GWTrigger',
    'EMCounterpart',
    'JointGWEMDetection',
    'create_gw_em_correlator',
    'JointLightCurveFitter',
    'GWStrainModel',
    'KilonovaLightCurveModel',
    'GRBAfterglowModel',
    'NeutrinoFluenceModel',
    'JointLikelihood',
    'MultiMessengerData',
    'PhysicalParameters',
    'create_joint_fitter',
    # V45: Causal Discovery for Astronomy (Phase 4)
    'AstroFCI',
    'TemporalCausalDiscovery',
    'AstronomicalConditionalIndependence',
    'CausalGraph',
    'create_astro_fci',
    'create_temporal_discovery',
]


