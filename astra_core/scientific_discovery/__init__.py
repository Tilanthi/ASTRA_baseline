"""
STAN Scientific Discovery Module
================================

Comprehensive scientific discovery system for autonomous research in astronomy
and astrophysics. Integrates literature mining, data analysis, theoretical
modeling, and autonomous experiment design.

Modules:
--------
- research_papers: PDF processing, citation networks, literature mining
- astro_databases: Access to Vizier, SIMBAD, ADS, and other catalogs
- data_repositories: Access to ALMA, NASA, ESO, CADC, arXiv datasets
- advanced_analysis: ML photometry, galaxy classification, phot-z
- theoretical_physics: MHD solvers, plasma physics, radiation-hydro
- discovery_orchestrator: Central autonomous discovery coordinator

Version: 1.0.0-Discovery
Date: 2025-12-27
"""

# =============================================================================
# Research Paper Processing
# =============================================================================

def _degraded_warn(_module: str, _exc: BaseException) -> None:
    """Log why an optional import degraded instead of failing silently."""
    import logging
    logging.getLogger(__name__).warning(
        "%s unavailable (%s: %s) - dependent names set to None",
        _module, type(_exc).__name__, _exc,
    )


try:
    from .research_papers import (
    PDFProcessor,
    CitationNetwork,
    LiteratureMiner,
    PaperAnalyzer,
    Paper,
    CitationGraph,
    extract_paper_metadata,
    build_citation_network,
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".research_papers", _exc)
    PDFProcessor = CitationNetwork = LiteratureMiner = PaperAnalyzer = Paper = CitationGraph = extract_paper_metadata = build_citation_network = None  # degraded: unavailable

# =============================================================================
# Astronomical Database Access
# =============================================================================
try:
    from .astro_databases import (
    AstroDatabaseConnector,
    VizierClient,
    SIMBADClient,
    ADSClient,
    CatalogQuery,
    SourceInfo,
    query_catalog,
    cross_match_catalogs,
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".astro_databases", _exc)
    AstroDatabaseConnector = VizierClient = SIMBADClient = ADSClient = CatalogQuery = SourceInfo = query_catalog = cross_match_catalogs = None  # degraded: unavailable

# =============================================================================
# Data Repository Access
# =============================================================================
try:
    from .data_repositories import (
    DataRepositoryManager,
    ALMAArchive,
    NASAArchive,
    ESOArchive,
    CADCArchive,
    ArxivClient,
    DatasetDownloader,
    download_observation,
    query_archive,
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".data_repositories", _exc)
    DataRepositoryManager = ALMAArchive = NASAArchive = ESOArchive = CADCArchive = ArxivClient = DatasetDownloader = download_observation = query_archive = None  # degraded: unavailable

# =============================================================================
# Advanced Data Analysis
# =============================================================================
try:
    from .advanced_analysis import (
    AdvancedAnalyzer,
    GalaxyClassifier,
    PhotometricRedshiftEstimator,
    SEDFitter,
    SourceExtractor,
    LineIdentifier,
    classify_galaxy,
    estimate_photoz,
    fit_sed,
    identify_lines,
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".advanced_analysis", _exc)
    AdvancedAnalyzer = GalaxyClassifier = PhotometricRedshiftEstimator = SEDFitter = SourceExtractor = LineIdentifier = classify_galaxy = estimate_photoz = fit_sed = identify_lines = None  # degraded: unavailable

# =============================================================================
# Theoretical Physics
# =============================================================================
try:
    from .theoretical_physics import (
    TheoreticalPhysicsEngine,
    MHDSolver,
    PlasmaPhysicsModule,
    RadiationHydrodynamics,
    GRMHDModule,
    CosmicRayTransport,
    MagneticReconnection,
    solve_mhd,
    run_radiation_hydro,
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".theoretical_physics", _exc)
    TheoreticalPhysicsEngine = MHDSolver = PlasmaPhysicsModule = RadiationHydrodynamics = GRMHDModule = CosmicRayTransport = MagneticReconnection = solve_mhd = run_radiation_hydro = None  # degraded: unavailable

# =============================================================================
# Discovery Orchestrator (Main Entry Point)
# =============================================================================
try:
    from .discovery_orchestrator import (
    ScientificDiscoveryOrchestrator,
    DiscoveryTask,
    DiscoveryResult,
    Hypothesis,
    ExperimentProposal,
    LiteratureReview,
    create_discovery_system,
    autonomous_discovery,
    review_literature,
    propose_experiment,
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".discovery_orchestrator", _exc)
    ScientificDiscoveryOrchestrator = DiscoveryTask = DiscoveryResult = Hypothesis = ExperimentProposal = LiteratureReview = create_discovery_system = autonomous_discovery = review_literature = propose_experiment = None  # degraded: unavailable

__all__ = [
    # Research Papers
    'PDFProcessor',
    'CitationNetwork',
    'LiteratureMiner',
    'PaperAnalyzer',
    'Paper',
    'CitationGraph',
    'extract_paper_metadata',
    'build_citation_network',

    # Astro Databases
    'AstroDatabaseConnector',
    'VizierClient',
    'SIMBADClient',
    'ADSClient',
    'CatalogQuery',
    'SourceInfo',
    'query_catalog',
    'cross_match_catalogs',

    # Data Repositories
    'DataRepositoryManager',
    'ALMAArchive',
    'NASAArchive',
    'ESOArchive',
    'CADCArchive',
    'ArxivClient',
    'DatasetDownloader',
    'download_observation',
    'query_archive',

    # Advanced Analysis
    'AdvancedAnalyzer',
    'GalaxyClassifier',
    'PhotometricRedshiftEstimator',
    'SEDFitter',
    'SourceExtractor',
    'LineIdentifier',
    'classify_galaxy',
    'estimate_photoz',
    'fit_sed',
    'identify_lines',

    # Theoretical Physics
    'TheoreticalPhysicsEngine',
    'MHDSolver',
    'PlasmaPhysicsModule',
    'RadiationHydrodynamics',
    'GRMHDModule',
    'CosmicRayTransport',
    'MagneticReconnection',
    'solve_mhd',
    'run_radiation_hydro',

    # Discovery Orchestrator
    'ScientificDiscoveryOrchestrator',
    'DiscoveryTask',
    'DiscoveryResult',
    'Hypothesis',
    'ExperimentProposal',
    'LiteratureReview',
    'create_discovery_system',
    'autonomous_discovery',
    'review_literature',
    'propose_experiment',
]

__version__ = '1.0.0-Discovery'
