"""
Astronomical Knowledge Graph

A specialized knowledge graph for representing astronomical objects,
physical relationships, and observational constraints.

Key features:
1. Nodes represent physical objects (stars, galaxies, black holes)
2. Edges represent physical relationships (orbits, lenses, radiates)
3. Includes hierarchical scales (stellar → galactic → cosmological)
4. Supports uncertainty and measurement errors
5. Integrates with physics constraints
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path


# =============================================================================
# NODE TYPES
# =============================================================================

class AstroNodeType(Enum):
    """Astronomical object and concept types"""

    # Physical Objects
    STAR = "star"
    GALAXY = "galaxy"
    BLACK_HOLE = "black_hole"
    NEUTRON_STAR = "neutron_star"
    WHITE_DWARF = "white_dwarf"
    PLANET = "planet"
    ASTEROID = "asteroid"
    COMET = "comet"
    NEBULA = "nebula"
    CLUSTER = "cluster"  # Star cluster or galaxy cluster
    QUASAR = "quasar"
    DARK_MATTER_HALO = "dark_matter_halo"

    # Molecular Cloud / ISM Objects
    MOLECULAR_CLOUD = "molecular_cloud"
    GIANT_MOLECULAR_CLOUD = "giant_molecular_cloud"
    CLOUD_COMPLEX = "cloud_complex"
    FILAMENT = "filament"
    DENSE_CORE = "dense_core"
    PROTOSTAR = "protostar"
    HII_REGION = "hii_region"
    PHOTODISSOCIATION_REGION = "photodissociation_region"
    SUPERNOVA_REMNANT = "supernova_remnant"

    # ISM Phases
    WARM_NEUTRAL_MEDIUM = "warm_neutral_medium"
    COLD_NEUTRAL_MEDIUM = "cold_neutral_medium"
    WARM_IONIZED_MEDIUM = "warm_ionized_medium"
    HOT_IONIZED_MEDIUM = "hot_ionized_medium"

    # Dust Components
    DUST_GRAIN = "dust_grain"
    PAH = "pah"  # Polycyclic Aromatic Hydrocarbons
    ICE_MANTLE = "ice_mantle"
    SILICATE_GRAIN = "silicate_grain"
    CARBONACEOUS_GRAIN = "carbonaceous_grain"

    # Observational
    OBSERVATION = "observation"
    SPECTRUM = "spectrum"
    IMAGE = "image"
    LIGHT_CURVE = "light_curve"
    MEASUREMENT = "measurement"

    # Physical Properties
    PHYSICAL_PROPERTY = "physical_property"
    DERIVED_QUANTITY = "derived_quantity"

    # Theoretical
    MODEL = "model"
    HYPOTHESIS = "hypothesis"
    MECHANISM = "mechanism"
    CONSTRAINT = "constraint"

    # Spatial/Temporal
    POSITION = "position"
    EPOCH = "epoch"
    EVENT = "event"


class RelationType(Enum):
    """Types of relationships between astronomical entities"""

    # Physical relationships
    ORBITS = "orbits"
    HOSTS = "hosts"  # Galaxy hosts star, star hosts planet
    LENSES = "lenses"  # Gravitational lensing
    ACCRETES_FROM = "accretes_from"
    MERGES_WITH = "merges_with"
    EJECTS = "ejects"
    IRRADIATES = "irradiates"

    # Observational relationships
    OBSERVED_BY = "observed_by"
    MEASURED_AS = "measured_as"
    CONSTRAINS = "constrains"
    DERIVED_FROM = "derived_from"

    # Structural relationships
    CONTAINS = "contains"
    PART_OF = "part_of"
    ASSOCIATED_WITH = "associated_with"

    # Causal relationships
    CAUSES = "causes"
    PRODUCES = "produces"
    REQUIRES = "requires"
    EXCLUDES = "excludes"

    # Similarity/Analogy
    ANALOGOUS_TO = "analogous_to"
    SIMILAR_TO = "similar_to"
    SAME_CLASS_AS = "same_class_as"

    # Molecular Cloud / ISM relationships
    FRAGMENTS_INTO = "fragments_into"           # Cloud → cores/filaments
    COLLAPSES_TO = "collapses_to"              # Core → protostar
    SHIELDS = "shields"                         # Cloud shields interior from UV
    HEATS = "heats"                            # Radiation heats gas/dust
    COOLS = "cools"                            # Molecules cool gas
    SUPPORTS = "supports"                       # Turbulence/B-field supports against gravity
    DRIVES_TURBULENCE = "drives_turbulence"    # Feedback → turbulence
    OUTFLOWS_INTO = "outflows_into"            # Protostar → outflow → cloud
    PHOTOIONIZES = "photoionizes"              # O/B star → HII region
    PHOTODISSOCIATES = "photodissociates"      # UV → PDR
    TRACES = "traces"                          # Molecule traces physical conditions
    DEPLETES_ONTO = "depletes_onto"            # Gas species → dust grains
    EVAPORATES_FROM = "evaporates_from"        # Ice mantle → gas phase
    EMITS_AT = "emits_at"                      # Source → spectral line/wavelength
    ABSORBS_AT = "absorbs_at"                  # Medium → absorption feature


# =============================================================================
# NODE DEFINITIONS
# =============================================================================

@dataclass
class UncertainValue:
    """A value with associated uncertainty"""
    value: float
    uncertainty: float
    unit: str
    confidence: float = 0.68  # Default 1-sigma

    def __str__(self):
        return f"{self.value:.4g} ± {self.uncertainty:.2g} {self.unit}"

    def to_dict(self) -> Dict:
        return {
            'value': self.value,
            'uncertainty': self.uncertainty,
            'unit': self.unit,
            'confidence': self.confidence
        }


@dataclass
class AstroNode:
    """Base class for astronomical knowledge graph nodes"""
    node_id: str
    node_type: AstroNodeType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    uncertainties: Dict[str, UncertainValue] = field(default_factory=dict)
    provenance: str = ""  # Source of information
    created: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0  # Confidence in node's existence/validity

    def add_property(self, key: str, value: Any, uncertainty: float = None,
                     unit: str = "", confidence: float = 0.68):
        """Add a property with optional uncertainty"""
        self.properties[key] = value
        if uncertainty is not None:
            self.uncertainties[key] = UncertainValue(value, uncertainty, unit, confidence)

    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'name': self.name,
            'properties': self.properties,
            'uncertainties': {k: v.to_dict() for k, v in self.uncertainties.items()},
            'provenance': self.provenance,
            'confidence': self.confidence
        }


@dataclass
class AstroEdge:
    """Relationship between astronomical entities"""
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    bidirectional: bool = False

    def to_dict(self) -> Dict:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type.value,
            'properties': self.properties,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'bidirectional': self.bidirectional
        }


def _json_default(obj):
    """JSON fallback: enums by value, datetimes as ISO strings."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


# =============================================================================
# SPECIALIZED NODE TYPES
# =============================================================================

@dataclass
class StarNode(AstroNode):
    """Node representing a star"""
    spectral_type: str = ""
    luminosity_class: str = ""

    def __post_init__(self):
        self.node_type = AstroNodeType.STAR


@dataclass
class GalaxyNode(AstroNode):
    """Node representing a galaxy"""
    morphology: str = ""  # Hubble type
    redshift: float = 0.0

    def __post_init__(self):
        self.node_type = AstroNodeType.GALAXY


@dataclass
class ObservationNode(AstroNode):
    """Node representing an observation"""
    instrument: str = ""
    wavelength_range: Tuple[float, float] = (0, 0)  # nm
    observation_date: datetime = None
    exposure_time: float = 0.0  # seconds

    def __post_init__(self):
        self.node_type = AstroNodeType.OBSERVATION


@dataclass
class MechanismNode(AstroNode):
    """Node representing a physical mechanism (from V36 mechanism discovery)"""
    equation: str = ""
    functional_form: str = ""
    domain: str = ""
    is_novel: bool = False

    def __post_init__(self):
        self.node_type = AstroNodeType.MECHANISM


@dataclass
class HypothesisNode(AstroNode):
    """Node representing a scientific hypothesis"""
    parameters: Dict[str, Any] = field(default_factory=dict)
    predictions: List[str] = field(default_factory=list)
    falsifiable: bool = True
    status: str = "untested"  # untested, supported, falsified

    def __post_init__(self):
        self.node_type = AstroNodeType.HYPOTHESIS


# =============================================================================
# MOLECULAR CLOUD & ISM NODE TYPES
# =============================================================================

@dataclass
class MolecularCloudNode(AstroNode):
    """Node representing a molecular cloud or cloud structure"""
    cloud_type: str = "molecular_cloud"  # GMC, filament, core, etc.
    mass_msun: float = 0.0
    radius_pc: float = 0.0
    mean_column_density: float = 0.0  # N_H2 in cm⁻²
    mean_volume_density: float = 0.0  # n_H2 in cm⁻³
    velocity_dispersion: float = 0.0  # km/s
    virial_parameter: float = 0.0
    mach_number: float = 0.0
    T_kin: float = 15.0  # K
    T_dust: float = 15.0  # K
    distance_pc: float = 0.0
    galactic_l: float = 0.0  # deg
    galactic_b: float = 0.0  # deg

    def __post_init__(self):
        self.node_type = AstroNodeType.MOLECULAR_CLOUD


@dataclass
class FilamentNode(AstroNode):
    """Node representing a filamentary structure"""
    length_pc: float = 0.0
    width_pc: float = 0.0  # FWHM
    aspect_ratio: float = 0.0
    line_mass: float = 0.0  # M_sun/pc
    critical_line_mass: float = 0.0  # M_sun/pc
    is_supercritical: bool = False
    n_cores: int = 0
    parent_cloud: str = ""

    def __post_init__(self):
        self.node_type = AstroNodeType.FILAMENT


@dataclass
class DenseCoreNode(AstroNode):
    """Node representing a dense core"""
    core_type: str = "starless"  # starless, prestellar, protostellar
    mass_msun: float = 0.0
    radius_pc: float = 0.0
    peak_density: float = 0.0  # cm⁻³
    T_kin: float = 10.0  # K
    is_bound: bool = False
    is_collapsing: bool = False
    has_infall_signature: bool = False
    has_outflow: bool = False
    parent_filament: str = ""
    parent_cloud: str = ""

    def __post_init__(self):
        self.node_type = AstroNodeType.DENSE_CORE


@dataclass
class DustNode(AstroNode):
    """Node representing dust properties in a region"""
    T_dust: float = 20.0  # K
    T_dust_uncertainty: float = 2.0
    beta: float = 1.8  # Emissivity spectral index
    beta_uncertainty: float = 0.1
    kappa_850: float = 1.85  # Opacity at 850μm (cm²/g)
    gas_to_dust: float = 100.0
    A_V: float = 0.0  # Visual extinction
    dust_model: str = "MRN"  # MRN, WD01, Ossenkopf, etc.
    has_ice_mantles: bool = False
    has_grain_growth: bool = False

    def __post_init__(self):
        self.node_type = AstroNodeType.DUST_GRAIN


@dataclass
class MolecularLineNode(AstroNode):
    """Node representing a molecular line observation"""
    molecule: str = ""
    transition: str = ""  # e.g., "J=1-0", "(1,1)"
    rest_frequency_ghz: float = 0.0
    integrated_intensity: float = 0.0  # K km/s
    peak_temperature: float = 0.0  # K
    line_width: float = 0.0  # km/s
    v_lsr: float = 0.0  # km/s
    optical_depth: float = 0.0
    excitation_temp: float = 0.0  # K
    column_density: float = 0.0  # cm⁻²
    abundance: float = 0.0  # relative to H2

    def __post_init__(self):
        self.node_type = AstroNodeType.SPECTRUM


@dataclass
class ISMPhaseNode(AstroNode):
    """Node representing an ISM phase"""
    phase: str = "CNM"  # WNM, CNM, WIM, HIM
    temperature: float = 0.0  # K
    density: float = 0.0  # cm⁻³
    ionization_fraction: float = 0.0
    filling_factor: float = 0.0
    pressure: float = 0.0  # K cm⁻³

    def __post_init__(self):
        if self.phase == "WNM":
            self.node_type = AstroNodeType.WARM_NEUTRAL_MEDIUM
        elif self.phase == "CNM":
            self.node_type = AstroNodeType.COLD_NEUTRAL_MEDIUM
        elif self.phase == "WIM":
            self.node_type = AstroNodeType.WARM_IONIZED_MEDIUM
        else:
            self.node_type = AstroNodeType.HOT_IONIZED_MEDIUM


# =============================================================================
# ASTRONOMICAL KNOWLEDGE GRAPH
# =============================================================================

class AstronomicalKnowledgeGraph:
    """
    Knowledge graph for astronomical inference

    Key features:
    1. Physical constraint validation on edges
    2. Hierarchical scale organization
    3. Uncertainty propagation
    4. Provenance tracking
    5. Hypothesis management
    """

    def __init__(self, name: str = "astro_kg"):
        self.name = name
        self.nodes: Dict[str, AstroNode] = {}
        self.edges: List[AstroEdge] = []

        # Indices for fast lookup
        self.type_index: Dict[AstroNodeType, Set[str]] = {}
        self.name_index: Dict[str, str] = {}  # name -> node_id
        self.adjacency: Dict[str, List[str]] = {}  # node_id -> [connected_node_ids]

        # Hypothesis tracking
        self.hypotheses: Dict[str, HypothesisNode] = {}
        self.active_hypothesis: Optional[str] = None

        # Scale hierarchy
        self.scale_hierarchy = [
            'subatomic', 'atomic', 'molecular', 'planetary', 'stellar',
            'galactic', 'cluster', 'cosmological'
        ]

    # =========================================================================
    # NODE OPERATIONS
    # =========================================================================

    def add_node(self, node: AstroNode) -> str:
        """Add a node to the graph"""
        self.nodes[node.node_id] = node

        # Update type index
        if node.node_type not in self.type_index:
            self.type_index[node.node_type] = set()
        self.type_index[node.node_type].add(node.node_id)

        # Update name index
        self.name_index[node.name.lower()] = node.node_id

        # Initialize adjacency
        if node.node_id not in self.adjacency:
            self.adjacency[node.node_id] = []

        return node.node_id

    def get_node(self, node_id: str) -> Optional[AstroNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)

    def find_by_name(self, name: str) -> Optional[AstroNode]:
        """Find a node by name"""
        node_id = self.name_index.get(name.lower())
        if node_id:
            return self.nodes.get(node_id)
        return None

    def find_by_type(self, node_type: AstroNodeType) -> List[AstroNode]:
        """Find all nodes of a given type"""
        node_ids = self.type_index.get(node_type, set())
        return [self.nodes[nid] for nid in node_ids]

    # =========================================================================
    # EDGE OPERATIONS
    # =========================================================================

    def add_edge(self, edge: AstroEdge) -> bool:
        """Add an edge to the graph"""
        # Validate nodes exist
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            return False

        self.edges.append(edge)

        # Update adjacency
        self.adjacency[edge.source_id].append(edge.target_id)
        if edge.bidirectional:
            self.adjacency[edge.target_id].append(edge.source_id)
        return True

    # =========================================================================
    # QUERIES / PERSISTENCE / STATISTICS
    # (re-implemented 2026-08; bodies lost to file truncation before the
    #  audit. API reconstructed from the surviving call sites in
    #  astro_physics/core.py: load(path), save(path), get_statistics().)
    # ==========================================================================

    def get_neighbors(self, node_id: str) -> List[str]:
        """Node IDs directly connected to the given node."""
        return list(self.adjacency.get(node_id, []))

    def get_edges(self, node_id: Optional[str] = None) -> List[AstroEdge]:
        """All edges, or only those touching the given node."""
        if node_id is None:
            return list(self.edges)
        return [e for e in self.edges
                if node_id in (e.source_id, e.target_id)]

    def save(self, path) -> None:
        """Save the graph to JSON (full node/edge fidelity)."""
        import dataclasses

        def _node_payload(node: AstroNode) -> Dict:
            payload = dataclasses.asdict(node)
            payload['node_class'] = type(node).__name__
            return payload

        data = {
            'name': self.name,
            'nodes': [_node_payload(n) for n in self.nodes.values()],
            'edges': [dataclasses.asdict(e) for e in self.edges],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, default=_json_default, indent=2))

    def load(self, path) -> None:
        """Load a graph from JSON saved by :meth:`save`."""
        import dataclasses

        _classes = {
            cls.__name__: cls for cls in (
                AstroNode, StarNode, GalaxyNode, ObservationNode,
                MechanismNode, HypothesisNode, MolecularCloudNode,
                FilamentNode, DenseCoreNode,
            )
        }

        data = json.loads(Path(path).read_text())
        self.name = data.get('name', self.name)
        self.nodes.clear()
        self.edges.clear()
        self.type_index.clear()
        self.name_index.clear()
        self.adjacency.clear()
        self.hypotheses.clear()

        # Nodes: new format is a list of payloads with 'node_class';
        # the older format was a dict keyed by node_id of plain to_dict()
        # payloads (base AstroNode fields only).
        raw_nodes = data.get('nodes', [])
        if isinstance(raw_nodes, dict):
            raw_nodes = list(raw_nodes.values())

        for payload in raw_nodes:
            if not isinstance(payload, dict):
                continue
            cls = _classes.get(payload.pop('node_class', ''), AstroNode)
            field_names = {f.name for f in dataclasses.fields(cls)}
            kwargs = {k: v for k, v in payload.items() if k in field_names}
            if isinstance(kwargs.get('node_type'), str):
                try:
                    kwargs['node_type'] = AstroNodeType(kwargs['node_type'])
                except ValueError:
                    kwargs['node_type'] = AstroNodeType.PHYSICAL_PROPERTY
            try:
                node = cls(**kwargs)
            except TypeError:
                node = AstroNode(**kwargs)
            self.add_node(node)
            if isinstance(node, HypothesisNode):
                self.hypotheses[node.node_id] = node

        for payload in data.get('edges', []):
            if isinstance(payload.get('relation_type'), str):
                payload['relation_type'] = RelationType(payload['relation_type'])
            field_names = {f.name for f in dataclasses.fields(AstroEdge)}
            self.add_edge(AstroEdge(
                **{k: v for k, v in payload.items() if k in field_names}))

    def get_statistics(self) -> Dict[str, Any]:
        """Graph summary statistics."""
        type_counts: Dict[str, int] = {}
        for node_type, ids in self.type_index.items():
            type_counts[node_type.value] = len(ids)
        return {
            'name': self.name,
            'n_nodes': len(self.nodes),
            'n_edges': len(self.edges),
            'n_hypotheses': len(self.hypotheses),
            'node_types': type_counts,
        }
