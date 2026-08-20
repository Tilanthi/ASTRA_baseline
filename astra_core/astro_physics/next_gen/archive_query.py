"""
Archive Query Interfaces

Provides unified access to major astronomical archives via VO/TAP protocols
and astroquery interfaces.

Supported Archives:
- MAST (HST, JWST, TESS, Kepler)
- ESO Archive (VLT, ALMA, VISTA)
- IRSA (Spitzer, WISE, 2MASS, Herschel)
- CDS/VizieR (catalogs, cross-matching)
- Gaia Archive (astrometry, photometry, RVS)
- CADC (CFHT, JCMT, Gemini)
- NRAO Archive (VLA, ALMA, GBT)
- NED (extragalactic cross-IDs)
- Simbad (object queries)

Date: 2025-12-15
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings

try:
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astropy.table import Table, vstack
    from astropy.time import Time
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

try:
    from astroquery.vizier import Vizier
    from astroquery.simbad import Simbad
    from astroquery.ned import Ned
    from astroquery.gaia import Gaia
    from astroquery.mast import Observations as MASTObs
    from astroquery.ipac.irsa import Irsa
    from astroquery.eso import Eso
    ASTROQUERY_AVAILABLE = True
except ImportError:
    ASTROQUERY_AVAILABLE = False


class ArchiveType(Enum):
    """Supported astronomical archives"""
    VIZIER = "vizier"
    SIMBAD = "simbad"
    NED = "ned"
    GAIA = "gaia"
    MAST = "mast"
    IRSA = "irsa"
    ESO = "eso"
    CADC = "cadc"
    NRAO = "nrao"


@dataclass
class QueryResult:
    """Container for archive query results"""
    archive: str
    query_type: str
    n_results: int
    table: Any  # astropy Table
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def __repr__(self):
        return f"QueryResult({self.archive}, {self.n_results} results)"


@dataclass
class ConeSearchParams:
    """Parameters for cone search queries"""
    ra: float  # degrees
    dec: float  # degrees
    radius: float  # arcmin
    catalog: Optional[str] = None
    columns: Optional[List[str]] = None
    row_limit: int = 10000


@dataclass
class CrossMatchParams:
    """Parameters for cross-matching queries"""
    ra_col: str = "ra"
    dec_col: str = "dec"
    radius: float = 1.0  # arcsec
    join_type: str = "best"  # 'best', 'all', or distance threshold


class TAP_Client:
    """
    Table Access Protocol (TAP) client for VO-compliant archives.

    Supports ADQL queries to any TAP endpoint.
    """

    # Standard TAP endpoints
    TAP_ENDPOINTS = {
        'gaia': 'https://gea.esac.esa.int/tap-server/tap',
        'vizier': 'http://tapvizier.u-strasbg.fr/TAPVizieR/tap',
        'cadc': 'https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/tap',
        'eso': 'http://archive.eso.org/tap_obs',
        'mast': 'https://mast.stsci.edu/vo-tap/api/v0.1',
        'ned': 'https://ned.ipac.caltech.edu/tap',
    }

    def __init__(self, endpoint: str = None, archive: str = None):
        """
        Initialize TAP client.

        Args:
            endpoint: Direct TAP endpoint URL
            archive: Archive name (uses predefined endpoint)
        """
        if endpoint:
            self.endpoint = endpoint
        elif archive and archive.lower() in self.TAP_ENDPOINTS:
            self.endpoint = self.TAP_ENDPOINTS[archive.lower()]
        else:
            raise ValueError(f"Specify endpoint URL or archive from: {list(self.TAP_ENDPOINTS.keys())}")

        self.archive = archive or "custom"

    def query(self, adql: str, maxrec: int = 10000) -> QueryResult:
        """
        Execute ADQL query.

        Args:
            adql: ADQL query string
            maxrec: Maximum records to return

        Returns:
            QueryResult with table
        """
        if not ASTROQUERY_AVAILABLE:
            raise ImportError("astroquery required for TAP queries")

        from astroquery.utils.tap.core import TapPlus

        tap = TapPlus(url=self.endpoint)
        job = tap.launch_job(adql, maxrec=maxrec)
        result_table = job.get_results()

        return QueryResult(
            archive=self.archive,
            query_type="TAP/ADQL",
            n_results=len(result_table),
            table=result_table,
            metadata={'adql': adql, 'endpoint': self.endpoint}
        )

    def cone_search(self, ra: float, dec: float, radius: float,
                    table: str, columns: str = "*") -> QueryResult:
        """
        Cone search via TAP.

        Args:
            ra, dec: Center coordinates (degrees)
            radius: Search radius (arcmin)
            table: Table name to query
            columns: Columns to return

        Returns:
            QueryResult
        """
        adql = f"""
        SELECT {columns}
        FROM {table}
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius/60.0})
        )
        """
        return self.query(adql)


class VOQueryEngine:
    """
    High-level Virtual Observatory query engine.

    Provides unified interface to multiple archives with automatic
    protocol selection.
    """

    def __init__(self):
        """Initialize VO query engine with available services"""
        self.available_services = []

        if ASTROQUERY_AVAILABLE:
            self.available_services.extend([
                'vizier', 'simbad', 'ned', 'gaia', 'mast', 'irsa'
            ])

        self._cache = {}

    def resolve_name(self, name: str) -> Tuple[float, float]:
        """
        Resolve object name to coordinates.

        Args:
            name: Object name (e.g., "M31", "NGC 1068")

        Returns:
            (ra, dec) in degrees
        """
        if not ASTROQUERY_AVAILABLE:
            raise ImportError("astroquery required for name resolution")

        result = Simbad.query_object(name)
        if result is None:
            # Try NED
            result = Ned.query_object(name)
            if result is None:
                raise ValueError(f"Could not resolve: {name}")
            ra = result['RA'][0]
            dec = result['DEC'][0]
        else:
            coord = SkyCoord(
                result['RA'][0], result['DEC'][0],
                unit=(u.hourangle, u.deg)
            )
            ra, dec = coord.ra.deg, coord.dec.deg

        return ra, dec

    def query_region(self, ra: float, dec: float, radius: float,
                     archives: List[str] = None,
                     catalogs: Dict[str, List[str]] = None) -> Dict[str, QueryResult]:
        """
        Query multiple archives for a sky region.

        Args:
            ra, dec: Center coordinates (degrees)
            radius: Search radius (arcmin)
            archives: List of archives to query (default: all available)
            catalogs: Dict mapping archive to specific catalogs

        Returns:
            Dict mapping archive name to QueryResult
        """
        if not ASTROPY_AVAILABLE:
            raise ImportError("astropy required")

        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        radius_u = radius * u.arcmin
        if archives is None:
            archives = self.available_services

        results: Dict[str, QueryResult] = {}
        for archive in archives:
            warnings_list: List[str] = []
            try:
                if archive == 'vizier':
                    tables = Vizier.query_region(coord, radius=radius_u,
                                                  catalog=catalogs.get('vizier') if catalogs else None)
                    table = vstack(list(tables)) if len(tables) else Table()
                elif archive == 'simbad':
                    table = Simbad.query_region(coord, radius=radius_u) or Table()
                elif archive == 'ned':
                    table = Ned.query_region(coord, radius=radius_u) or Table()
                elif archive == 'gaia':
                    adql = ("SELECT * FROM gaiadr3.gaia_source "
                            "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
                            f"CIRCLE('ICRS', {ra}, {dec}, {radius/60.0})))")
                    table = Gaia.launch_job(adql).get_results()
                elif archive == 'mast':
                    table = MASTObs.query_region(coord, radius=radius_u) or Table()
                elif archive == 'irsa':
                    table = Irsa.query_region(coord, radius=radius_u,
                                              catalog=catalogs.get('irsa') if catalogs else None)
                else:
                    warnings_list.append(f"unknown archive: {archive}")
                    continue
                results[archive] = QueryResult(
                    archive=archive, query_type="region",
                    n_results=len(table), table=table,
                    metadata={'ra': ra, 'dec': dec, 'radius_arcmin': radius},
                    warnings=warnings_list)
            except Exception as exc:
                results[archive] = QueryResult(
                    archive=archive, query_type="region", n_results=0,
                    table=Table(),
                    metadata={'ra': ra, 'dec': dec, 'radius_arcmin': radius},
                    warnings=[f"query failed: {exc}"])
        return results


# =============================================================================
# UNIFIED ASTROQUERY INTERFACE
# (re-implemented 2026-08; the original bodies were lost to file truncation
#  before the August 2026 audit)
# =============================================================================

class AstroqueryInterface:
    """
    Unified wrapper around the astroquery modules for the major archives.

    Every method returns a :class:`QueryResult`.  Methods raise ImportError
    when astroquery is not installed and pass through archive errors
    unchanged so callers can handle transient network failures.
    """

    def __init__(self):
        if not ASTROQUERY_AVAILABLE:
            raise ImportError("astroquery is required for AstroqueryInterface")

    # ------------------------------------------------------------- catalogs
    def query_vizier(self, catalog: str, ra: float, dec: float,
                     radius: float, row_limit: int = 10000) -> QueryResult:
        """Cone search a VizieR catalog (radius in arcmin)."""
        v = Vizier(row_limit=row_limit)
        tables = v.query_region(
            SkyCoord(ra=ra*u.deg, dec=dec*u.deg), radius=radius*u.arcmin,
            catalog=catalog)
        table = vstack(list(tables)) if len(tables) else Table()
        return QueryResult('vizier', 'cone', len(table), table,
                           {'catalog': catalog})

    def query_gaia(self, ra: float, dec: float, radius: float,
                   row_limit: int = 10000) -> QueryResult:
        """Gaia DR3 cone search (radius in arcmin) via ADQL."""
        adql = ("SELECT source_id, ra, dec, parallax, pmra, pmdec, "
                "phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag "
                "FROM gaiadr3.gaia_source "
                "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
                f"CIRCLE('ICRS', {ra}, {dec}, {radius/60.0})))")
        job = Gaia.launch_job(adql)
        table = job.get_results()
        return QueryResult('gaia', 'cone', len(table), table, {'adql': adql})

    # -------------------------------------------------------------- objects
    def query_simbad(self, name: str) -> QueryResult:
        """SIMBAD identifiers/measurements for a named object."""
        table = Simbad.query_object(name) or Table()
        return QueryResult('simbad', 'object', len(table), table, {'name': name})

    def query_ned(self, name: str) -> QueryResult:
        """NED object query (photometry + redshifts)."""
        table = Ned.query_object(name) or Table()
        return QueryResult('ned', 'object', len(table), table, {'name': name})

    # ---------------------------------------------------------- observatories
    def query_mast(self, target: Optional[str] = None,
                   ra: Optional[float] = None, dec: Optional[float] = None,
                   radius: float = 5.0, obs_collection: str = None) -> QueryResult:
        """MAST observation footprint query by target name or position."""
        kwargs = {'target': target} if target else {
            'coordinates': SkyCoord(ra=ra*u.deg, dec=dec*u.deg),
            'radius': radius * u.arcmin}
        if obs_collection:
            kwargs['obs_collection'] = obs_collection
        table = MASTObs.query_criteria(**kwargs) or Table()
        return QueryResult('mast', 'observations', len(table), table, kwargs)

    def query_irsa(self, catalog: str, ra: float, dec: float,
                   radius: float) -> QueryResult:
        """IRSA cone search (2MASS/WISE/Spitzer/...; radius in arcmin)."""
        table = Irsa.query_region(
            SkyCoord(ra=ra*u.deg, dec=dec*u.deg), radius=radius*u.arcmin,
            catalog=catalog) or Table()
        return QueryResult('irsa', 'cone', len(table), table, {'catalog': catalog})


class CrossMatchEngine:
    """
    Positional cross-matching between astropy Tables using SkyCoord.

    Uses vectorised ``match_to_catalog_sky`` and keeps the best match per
    source within the configured radius (CrossMatchParams.radius arcsec).
    """

    def __init__(self, params: Optional[CrossMatchParams] = None):
        if not ASTROPY_AVAILABLE:
            raise ImportError("astropy is required for CrossMatchEngine")
        self.params = params or CrossMatchParams()

    def match_tables(self, table1, table2, params: CrossMatchParams = None,
                     ra1_col: str = 'ra', dec1_col: str = 'dec',
                     ra2_col: str = 'ra', dec2_col: str = 'dec'):
        """
        Match table1 against table2.

        Returns (matched_rows, separations_arcsec) where matched_rows are the
        row indices into table2 (-1 for unmatched table1 sources).
        """
        p = params or self.params
        c1 = SkyCoord(table1[ra1_col] * u.deg, table1[dec1_col] * u.deg)
        c2 = SkyCoord(table2[ra2_col] * u.deg, table2[dec2_col] * u.deg)
        idx, sep2d, _ = c1.match_to_catalog_sky(c2)
        sep = sep2d.arcsec
        keep = sep <= p.radius
        matched_rows = np.where(keep, idx, -1)
        return matched_rows, sep

    def crossmatch(self, table1, table2, **kwargs):
        """Convenience: return table1 rows with a match column + separation."""
        matched, sep = self.match_tables(table1, table2, **kwargs)
        out = table1.copy()
        out['match_idx'] = matched
        out['separation_arcsec'] = sep
        return out


class ArchiveDataManager:
    """Local cache and bookkeeping for QueryResult tables (ECSV format)."""

    def __init__(self, cache_dir: str = "archive_cache"):
        from pathlib import Path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: QueryResult, name: Optional[str] = None) -> str:
        name = name or f"{result.archive}_{result.query_type}"
        path = self.cache_dir / f"{name}.ecsv"
        if result.n_results:
            result.table.write(str(path), format='ascii.ecsv', overwrite=True)
        return str(path)

    def load(self, name: str):
        path = self.cache_dir / f"{name}.ecsv"
        if not path.exists():
            raise FileNotFoundError(f"No cached result: {path}")
        return Table.read(str(path), format='ascii.ecsv')

    def list_cached(self) -> List[str]:
        return sorted(f.stem for f in self.cache_dir.glob('*.ecsv'))

    def clear_cache(self) -> None:
        for f in self.cache_dir.glob('*.ecsv'):
            f.unlink()


# =============================================================================
# RADIO ARCHIVES
# (thin wrappers over their public TAP services)
# =============================================================================

class ALMAArchive:
    """ALMA science archive via its public TAP service (ObsCore)."""

    TAP_ENDPOINT = "https://almascience.eso.org/tap"

    def __init__(self):
        self.client = TAP_Client(endpoint=self.TAP_ENDPOINT, archive="alma")

    def query_region(self, ra: float, dec: float, radius_arcmin: float,
                     columns: str = "*", table: str = "ivoa.obscore") -> QueryResult:
        """Cone search over ALMA ObsCore (projects, bands, angular resolution)."""
        return self.client.cone_search(ra, dec, radius_arcmin, table, columns)


class ESOArchive:
    """ESO observation archive (VLT, La Silla, ALMA PI data) via TAP."""

    def __init__(self):
        self.client = TAP_Client(archive="eso")

    def query_region(self, ra: float, dec: float, radius_arcmin: float,
                     columns: str = "*", table: str = "ivoa.ObsCore") -> QueryResult:
        return self.client.cone_search(ra, dec, radius_arcmin, table, columns)


class NRAOArchive:
    """
    NRAO data archive (VLA, VLBA, GBT) query interface.

    The NRAO archive exposes a GET-based query endpoint; results are
    returned as the raw response (no VO table normalisation).
    """

    QUERY_URL = "https://archive.nrao.edu/archive/ArchiveQuery"

    def query_region(self, ra: float, dec: float, radius_arcmin: float,
                     telescope: str = "VLA", bands: str = "all") -> QueryResult:
        import urllib.parse
        import urllib.request
        params = urllib.parse.urlencode({
            'SITE': 'NRAO', 'TELESCOPE': telescope,
            'RA': f"{ra:.6f}", 'DEC': f"{dec:.6f}",
            'SRADIUS': f"{radius_arcmin/60.0:.6f}", 'BANDS': bands,
        })
        with urllib.request.urlopen(f"{self.QUERY_URL}?{params}", timeout=60) as r:
            raw = r.read().decode('utf-8', errors='replace')
        return QueryResult('nrao', 'region', 0, None,
                           {'query_url': f"{self.QUERY_URL}?{params}",
                            'raw_length': len(raw)})


class LOFARArchive:
    """LOFAR long-term archive surveys via the ASTRON TAP service."""

    TAP_ENDPOINT = "https://tap.astron.nl"

    def __init__(self):
        self.client = TAP_Client(endpoint=self.TAP_ENDPOINT, archive="lofar")

    def query_region(self, ra: float, dec: float, radius_arcmin: float,
                     table: str, columns: str = "*") -> QueryResult:
        """Cone search; table names as listed by the ASTRON TAP service."""
        return self.client.cone_search(ra, dec, radius_arcmin, table, columns)


class MWAArchive:
    """MWA survey data (e.g. GLEAM) via the CSIRO ASKAP/MWA (CASDA) TAP service."""

    TAP_ENDPOINT = "https://casda.csiro.au/casda_vo_tools/tap"

    def __init__(self):
        self.client = TAP_Client(endpoint=self.TAP_ENDPOINT, archive="mwa")

    def query_region(self, ra: float, dec: float, radius_arcmin: float,
                     table: str, columns: str = "*") -> QueryResult:
        return self.client.cone_search(ra, dec, radius_arcmin, table, columns)


class RadioArchiveManager:
    """Facade running one cone search across all radio archives."""

    def __init__(self):
        self.archives = {
            'alma': ALMAArchive, 'eso': ESOArchive, 'nrao': NRAOArchive,
            'lofar': LOFARArchive, 'mwa': MWAArchive,
        }

    def query_all(self, ra: float, dec: float, radius_arcmin: float,
                  archives: Optional[List[str]] = None,
                  tables: Optional[Dict[str, str]] = None) -> Dict[str, QueryResult]:
        tables = tables or {}
        results = {}
        for name in (archives or self.archives):
            try:
                arch = self.archives[name]()
                if name in ('lofar', 'mwa') and name not in tables:
                    results[name] = QueryResult(
                        name, 'region', 0, None,
                        warnings=[f"no table name supplied for {name} TAP"])
                    continue
                results[name] = arch.query_region(
                    ra, dec, radius_arcmin, **(
                        {'table': tables[name]} if name in tables else {}))
            except Exception as exc:
                results[name] = QueryResult(name, 'region', 0, None,
                                            warnings=[f"query failed: {exc}"])
        return results
