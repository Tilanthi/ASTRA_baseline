"""
Astronomical Swarm Agents

Specialized agents for different types of astronomical analysis:
1. SpectroscopicAgent - Analyzes spectra, identifies lines, measures redshifts
2. PhotometricAgent - Analyzes light curves, SEDs, magnitudes
3. DynamicalAgent - Analyzes orbits, rotation curves, velocity fields
4. ImagingAgent - Analyzes images, morphology, source detection

Each agent type has domain-specific expertise but communicates via
stigmergic trails (pheromones) following Gordon's biological principles.

(Re-implemented 2026-08: the original agent bodies were lost to file
truncation before the August 2026 audit. The base-class API below is
reconstructed exactly from the surviving call sites - core.py's
create_agent() and the MolecularLineAgent / DustContinuumAgent /
CloudStructureAgent / StarFormationAgent subclasses in
molecular_cloud_agents.py.)
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import json
from pathlib import Path

from .physics import PhysicsEngine, PhysicalConstants
from .knowledge_graph import (
    AstronomicalKnowledgeGraph, AstroNode, AstroEdge,
    AstroNodeType, RelationType, MechanismNode, HypothesisNode
)


# =============================================================================
# STIGMERGIC COMMUNICATION STRUCTURES
# =============================================================================

@dataclass
class PheromoneTrail:
    """A discovery deposited into the shared stigmergic memory.

    Subclasses (e.g. CloudPheromoneTrail) extend this with domain-specific
    fields; the base fields below are exactly those set by the surviving
    trail-construction call sites.
    """
    trail_id: str
    agent_id: str
    agent_type: str
    timestamp: datetime
    discovery_type: str                      # e.g. 'column_density', 'temperature'
    content: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5                  # 0-1 subjective confidence
    strength: float = 5.0                    # decays via apply_evaporation()
    evidence_quality: float = 0.5            # 0-1 quality of underlying data


class StigmergicMemory:
    """Shared trail store through which agents communicate indirectly.

    Gordon-style stigmergy: agents never message each other; they leave
    trails and read the trails of others. Trails evaporate over analysis
    cycles so that stale discoveries fade and repeatedly confirmed ones
    dominate the consensus.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.trails: List[PheromoneTrail] = []
        if self.storage_path is not None:
            try:
                self.storage_path.mkdir(parents=True, exist_ok=True)
                self._load()
            except Exception:
                pass  # degraded: persistence unavailable, memory stays volatile

    # ------------------------------------------------------------------ API
    def leave_trail(self, trail: PheromoneTrail) -> None:
        """Deposit a trail (called by agents after each analysis)."""
        self.trails.append(trail)
        if len(self.trails) > 10000:
            # Keep the strongest: drop the weakest quarter
            self.trails.sort(key=lambda t: t.strength)
            del self.trails[:len(self.trails) // 4]
        self._save()

    def get_strong_trails(self, threshold: float = 5.0) -> List[PheromoneTrail]:
        """Return trails with strength >= threshold, strongest first."""
        return sorted(
            (t for t in self.trails if t.strength >= threshold),
            key=lambda t: t.strength, reverse=True)

    def apply_evaporation(self, rate: float = 0.05) -> None:
        """Decay all trail strengths by the given fraction; drop dead ones."""
        for trail in self.trails:
            trail.strength *= (1.0 - rate)
            trail.confidence *= (1.0 - 0.5 * rate)
        self.trails = [t for t in self.trails if t.strength > 0.1]
        self._save()

    # ----------------------------------------------------------- persistence
    def _save(self) -> None:
        if self.storage_path is None:
            return
        try:
            payload = [{
                'trail_id': t.trail_id,
                'agent_id': t.agent_id,
                'agent_type': t.agent_type,
                'timestamp': t.timestamp.isoformat()
                if isinstance(t.timestamp, datetime) else str(t.timestamp),
                'discovery_type': t.discovery_type,
                'content': t.content,
                'confidence': t.confidence,
                'strength': t.strength,
                'evidence_quality': t.evidence_quality,
            } for t in self.trails]
            (self.storage_path / 'trails.json').write_text(json.dumps(payload))
        except Exception:
            pass  # degraded: persistence unavailable

    def _load(self) -> None:
        f = self.storage_path / 'trails.json'
        if not f.exists():
            return
        try:
            for d in json.loads(f.read_text()):
                ts = d.get('timestamp')
                try:
                    ts = datetime.fromisoformat(ts)
                except (TypeError, ValueError):
                    ts = datetime.now()
                self.trails.append(PheromoneTrail(
                    trail_id=d['trail_id'], agent_id=d.get('agent_id', ''),
                    agent_type=d.get('agent_type', ''), timestamp=ts,
                    discovery_type=d.get('discovery_type', 'unknown'),
                    content=d.get('content', {}), confidence=d.get('confidence', 0.5),
                    strength=d.get('strength', 5.0),
                    evidence_quality=d.get('evidence_quality', 0.5)))
        except Exception:
            pass  # degraded: unreadable persistence, start empty


# =============================================================================
# BASE AGENT
# =============================================================================

class AstroAgent(ABC):
    """Base class for all astronomical swarm agents.

    An agent owns a physics engine (for quantitative work), a knowledge
    graph (to register discoveries) and a stigmergic memory (to share
    discoveries with other agents indirectly).
    """

    def __init__(self, agent_id: str,
                 physics_engine: Optional[PhysicsEngine],
                 knowledge_graph: Optional[AstronomicalKnowledgeGraph],
                 stigmergic_memory: Optional[StigmergicMemory]):
        self.agent_id = agent_id
        self.physics_engine = physics_engine
        self.knowledge_graph = knowledge_graph
        self.memory = stigmergic_memory

    @property
    def agent_type(self) -> str:
        return "AstroAgent"

    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze domain-specific data and return a result dict."""

    def leave_trail(self, discovery_type: str, content: Dict[str, Any],
                    confidence: float, strength: float,
                    evidence_quality: float) -> None:
        """Convenience wrapper for depositing a base-class trail."""
        if self.memory is None:
            return
        self.memory.leave_trail(PheromoneTrail(
            trail_id=f"trail_{self.agent_id}_{int(datetime.now().timestamp() * 1000) % 10**10}",
            agent_id=self.agent_id, agent_type=self.agent_type,
            timestamp=datetime.now(), discovery_type=discovery_type,
            content=content, confidence=confidence, strength=strength,
            evidence_quality=evidence_quality))


# =============================================================================
# SPECIALIZED AGENTS
# =============================================================================

def _detect_peaks(flux: np.ndarray, wavelengths: Optional[np.ndarray] = None,
                  sigma_threshold: float = 3.0) -> List[Dict[str, float]]:
    """Simple local-maximum line detector.

    A point is a peak if it is the maximum of its 5-point neighbourhood and
    stands more than sigma_threshold * robust_sigma above the continuum
    (robust sigma estimated from the median absolute deviation).
    """
    flux = np.asarray(flux, dtype=float)
    n = len(flux)
    if n < 5:
        return []
    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) * 1.4826
    sigma = mad if mad > 0 else max(flux.std(), 1e-30)
    peaks = []
    for i in range(2, n - 2):
        window = flux[i - 2:i + 3]
        if flux[i] == window.max() and flux[i] > med + sigma_threshold * sigma:
            # linear sub-pixel centroid across the local window
            w = np.clip(window - med, 0, None)
            offsets = np.arange(-2, 3)
            centre = float((w * offsets).sum() / max(w.sum(), 1e-30))
            idx = i + centre
            peaks.append({
                'index': i,
                'wavelength': float(wavelengths[int(round(idx))])
                if wavelengths is not None and 0 <= int(round(idx)) < n
                else float(idx),
                'peak_flux': float(flux[i]),
                'snr': float((flux[i] - med) / sigma),
            })
    return peaks


class SpectroscopicAgent(AstroAgent):
    """Spectrum analysis: emission-line detection and redshift estimation."""

    @property
    def agent_type(self) -> str:
        return "SpectroscopicAgent"

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {'agent_type': self.agent_type}
        flux = data.get('flux')
        if flux is None:
            result['message'] = 'No flux array provided'
            return result
        wave = data.get('wavelength')
        flux = np.asarray(flux, dtype=float)
        wave = np.asarray(wave, dtype=float) if wave is not None else None

        lines = _detect_peaks(flux, wave, sigma_threshold=data.get('sigma_threshold', 3.0))
        result['n_lines'] = len(lines)
        result['lines'] = lines

        # Redshift from matched rest wavelengths (nearest-rest matching)
        rest = data.get('rest_wavelengths')
        if rest is not None and lines:
            rest = np.asarray(rest, dtype=float)
            zs = []
            for line in lines:
                d = np.abs(rest - line['wavelength'])
                j = int(np.argmin(d))
                if d[j] < 0.05 * rest[j]:          # loose match window
                    zs.append((line['wavelength'] - rest[j]) / rest[j])
            if zs:
                result['redshift'] = float(np.median(zs))
                result['redshift_n_lines'] = len(zs)

        if lines:
            self.leave_trail(
                discovery_type='spectral_lines',
                content={'n_lines': len(lines),
                         'strongest_snr': max(l['snr'] for l in lines),
                         **({'redshift': result['redshift']}
                            if 'redshift' in result else {})},
                confidence=0.7, strength=6.0, evidence_quality=0.7)
        return result


class PhotometricAgent(AstroAgent):
    """Photometry analysis: SED peak temperature, colours, variability."""

    # Wien displacement constant [um * K]
    b_wien = 2897.771955

    @property
    def agent_type(self) -> str:
        return "PhotometricAgent"

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {'agent_type': self.agent_type}

        sed = data.get('sed')
        if isinstance(sed, dict) and sed.get('wavelength') is not None:
            wave = np.asarray(sed['wavelength'], dtype=float)
            flux = np.asarray(sed['flux'], dtype=float)
            i_peak = int(np.nanargmax(flux))
            result['sed_peak_wavelength'] = float(wave[i_peak])
            result['brightness_temperature_K'] = float(
                self.b_wien / wave[i_peak])          # Wien's law
            result['integrated_flux'] = float(np.trapezoid(flux, wave)) \
                if len(wave) > 1 else float(flux.sum())
            d = data.get('distance_m')
            if d:
                result['luminosity_W'] = float(
                    4 * np.pi * d ** 2 * result['integrated_flux'])

        mags = data.get('magnitudes')
        if isinstance(mags, dict) and len(mags) >= 2:
            bands = sorted(mags)
            result['colors'] = {
                f'{a}-{b}': float(mags[a] - mags[b])
                for a, b in zip(bands[:-1], bands[1:])}

        lc = data.get('lightcurve')
        if lc is not None:
            lc = np.asarray(lc, dtype=float)
            if lc.size > 1:
                result['mean_mag'] = float(lc.mean())
                result['std_mag'] = float(lc.std())
                result['variability_index'] = float(lc.std() / max(abs(lc.mean()), 1e-30))

        if not any(k in result for k in
                   ('brightness_temperature_K', 'colors', 'variability_index')):
            result['message'] = 'No SED, magnitudes or lightcurve provided'
            return result

        self.leave_trail(
            discovery_type='photometry',
            content={k: v for k, v in result.items() if k != 'agent_type'},
            confidence=0.7, strength=6.0, evidence_quality=0.75)
        return result


class DynamicalAgent(AstroAgent):
    """Kinematics analysis: velocity statistics, virial estimates."""

    @property
    def agent_type(self) -> str:
        return "DynamicalAgent"

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {'agent_type': self.agent_type}
        v = data.get('velocities')
        if v is None:
            result['message'] = 'No velocities array provided'
            return result
        v = np.asarray(v, dtype=float)
        result['v_mean'] = float(v.mean())
        result['v_std'] = float(v.std())

        # Virial mass M ~ 5 sigma^2 R / G for a uniform sphere
        # (velocities conventionally in km/s; converted to m/s here)
        radius = data.get('radius')
        if radius:
            G = PhysicalConstants.G
            radius_m = radius * PhysicalConstants.pc \
                if radius < 1e6 else radius      # pc -> m heuristic
            sigma_ms = v.std() * 1e3             # km/s -> m/s
            result['virial_mass_kg'] = float(5.0 * sigma_ms ** 2 * radius_m / G)
            result['virial_mass_Msun'] = float(
                result['virial_mass_kg'] / PhysicalConstants.M_sun)

        self.leave_trail(
            discovery_type='kinematics',
            content={k: v for k, v in result.items() if k != 'agent_type'},
            confidence=0.8, strength=6.0, evidence_quality=0.8)
        return result


class ImagingAgent(AstroAgent):
    """Image analysis: source detection and morphology statistics."""

    @property
    def agent_type(self) -> str:
        return "ImagingAgent"

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {'agent_type': self.agent_type}
        image = data.get('image')
        if image is None:
            result['message'] = 'No image array provided'
            return result
        img = np.asarray(image, dtype=float)
        if img.ndim != 2:
            result['message'] = 'Image must be 2-D'
            return result

        med = np.median(img)
        mad = np.median(np.abs(img - med)) * 1.4826
        sigma = mad if mad > 0 else max(float(img.std()), 1e-30)
        result['background'] = float(med)
        result['noise_sigma'] = float(sigma)

        # Local maxima brighter than threshold * sigma
        threshold = data.get('threshold_sigma', 5.0)
        padded = np.pad(img, 1, mode='edge')
        neighbourhood_max = np.full(img.shape, -np.inf)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbourhood_max = np.maximum(
                    neighbourhood_max, padded[1 + dy:1 + dy + img.shape[0],
                                              1 + dx:1 + dx + img.shape[1]])
        mask = (img == neighbourhood_max) & (img > med + threshold * sigma)
        ys, xs = np.nonzero(mask)
        sources = [{'x': int(x), 'y': int(y),
                    'peak_flux': float(img[y, x]),
                    'snr': float((img[y, x] - med) / sigma)}
                   for y, x in zip(ys, xs)]
        sources.sort(key=lambda s: s['snr'], reverse=True)
        result['n_sources'] = len(sources)
        result['sources'] = sources
        result['peak_snr'] = max((s['snr'] for s in sources), default=0.0)

        if sources:
            self.leave_trail(
                discovery_type='sources',
                content={'n_sources': len(sources),
                         'peak_snr': result['peak_snr']},
                confidence=0.75, strength=6.0, evidence_quality=0.7)
        return result
