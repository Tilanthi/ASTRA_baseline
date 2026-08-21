# ASTRA Baseline Archive

**This is a baseline archive of the ASTRA (Autonomous Scientific Discovery in Astrophysics) system.**

## What This Is

This repository contains a baseline snapshot of ASTRA at a specific point in its development. It serves as:

- A reference point for future development
- A stable baseline for comparison
- An archive of the core system architecture
- A clean starting point for derivative work

## Contents

This baseline archive includes:

- **astra_core/** - The core ASTRA system with all capabilities
- **CLAUDE.md** - Project documentation for AI-assisted development
- **README.md** - Main system documentation
- **User_Manual/** - Complete user manual and guides

## System Overview

ASTRA is a unified AGI-inspired framework for autonomous hypothesis generation and validation in astronomy and astrophysics. The system integrates ~317,000 lines of clean, functional code across modular cognitive capabilities.

### Key Capabilities

- **Causal Inference & Discovery**: Structural causal models, counterfactual reasoning
- **Meta-Learning**: MAML optimization, cross-domain transfer learning
- **Swarm Intelligence**: Multi-agent reasoning, stigmergic coordination
- **Domain Expertise**: 75 specialized astrophysics domain modules
- **Theory Engine**: Advanced theoretical reasoning and hypothesis generation
- **Meta-Cognitive Systems**: Multi-layered context representation, self-improvement
- **Astrophysics Modules**: 14 rebuilt/validated modules in `astro_physics/` (radiative transfer & LVG excitation, SPH gas dynamics, turbulence analysis, IR/submm SEDs, multiscale AMR coupling, and more)

## Quick Start

```python
from astra_core import create_stan_system

# Create system with auto-optimized capabilities
system = create_stan_system()

# Answer queries with automatic capability selection
result = system.answer("What causes supernovae?")
print(result['answer'])
```

## Architecture

```
Entry Points: create_stan_system() | create_v4_system() | process_query()
                              ↓
              V4.0 Revolutionary Capabilities
              (MCE, ASC, CRN, Multi-Mind Orchestration)
                              ↓
                    Domain Architecture
              (75 domain modules, hot-swappable)
                              ↓
                  Cross-Domain Meta-Learning
                              ↓
                  Physics & Causal Engines
                              ↓
                  Memory & Knowledge Systems
```

## Documentation

- **CLAUDE.md** - Comprehensive project documentation
- **README.md** - System overview and usage
- **User_Manual/** - Detailed user guides — **Appendix E** contains the full August 2026 codebase integrity audit record

## Baseline Information

- **Archive Date**: August 2026 (replaces the April 2026 baseline)
- **System State**: Post-audit rebuild, plus the August 2026 **re-audit** repairs
  described below
- **Status**: Imports cleanly and passes its declared suites; see *Known
  limitations* before relying on any physics result

### Verification (re-measured August 2026, reproducible)

Measured on this commit with `python3 -m compileall`, a fresh-interpreter import
of every module, and each suite run from the repository root with
`PYTHONPATH=.`:

| Verification | Result |
|---|---|
| Syntax scan (AST parse of every file) | 675/675 parse |
| Module imports (fresh-interpreter tree walk) | 664/675 |
| — of the 11 that do not import | all are optional dependencies absent (`torch`, `astropy`); install with `pip install -e .[all]` |
| Comprehensive system test | 18/18 (100%), exit 0 |
| `tests/test_all.py` | 14 passed |
| `tests/test_self_teaching.py` | 17 passed |
| `tests/test_revolutionary/test_v4_integration.py` | 11 passed |
| `tests/test_installation.py`, `tests/test_comprehensive_integration.py`, `tests/test_specialist_capabilities.py`, `tests/test_phase_2_4.py`, `tests/test_v47_causal_discovery.py`, `tests/test_v6_theoretical_discovery.py`, `tests/test_calibrated_outliers.py` | pass |

> **Correction.** The previous version of this file reported *"Module imports
> (full tree walk): 567/567, zero failures"* and *"100% pass rate on every
> declared suite"*. Those numbers did not describe the committed tree. As
> published, `import astra_core` failed immediately at
> `astra_core/metacognitive/monitoring/monitor.py:317`
> (`NameError: name 'np' is not defined`), so **0 of 675 modules imported** and
> `comprehensive_system_test.py` scored **0/18** — while still exiting 0,
> because it never called `sys.exit`. The tree also contains 675 Python files,
> not the 567 or 682 quoted in Appendix E. The table above replaces those
> figures with measured ones; see the *Re-audit* section.

### Known limitations (read before using for science)

- **The `astro_physics` layer is not uniformly trustworthy.** An independent
  numerical audit found defects ranging from sign errors to unit errors of
  14 orders of magnitude. The repaired ones are covered by regression tests in
  `astra_core/tests/test_physics_regressions*.py`; anything still marked
  `# AUDIT-FLAG:` in the source is *known-suspect and unfixed*.
- **Of the 75 domain modules, 48 are the same 110-line template.** Their
  `process_query` returns `f"{description}: Analysis of '{query}'"` with a
  hard-coded `confidence=0.7` and performs no computation. 73 of the 75 contain
  no call to `numpy` or `math` at all. The remaining domains are curated
  canned-text lookups keyed on substrings, also with hard-coded confidences.
  Treat `confidence` fields as routing weights, not statistics.
- **`astra_core.core_legacy` does not exist**, so the 13 `V36`–`V94` legacy
  systems referenced by `core/unified.py` are `None` at runtime.
- Optional heavy dependencies (`torch`, `astropy`, `reportlab`) are genuinely
  optional; their absence is logged, not silent.

## Re-audit and repairs (August 2026, second audit)

An independent audit of this published baseline found that the tree could not be
imported at all, and that several of the verification claims above did not hold.
The `audit-fixes-aug2026` work addressed:

- **Import chain** — removed 141 dead auto-injected definitions across 80 files
  (they annotated `np.ndarray` in their signatures while importing numpy only
  inside the function body, which is what broke the package); supplied the
  genuinely missing `numpy`/`typing`/`CausalGraph`/`LearningResult` imports; used
  `from __future__ import annotations` where an *optional* dependency appeared in
  a signature and thereby defeated its own `try/except` guard.
- **Silently nulled public API** — `astra_core/__init__.py` imported
  `create_unified_stan_system` (defined nowhere), the PDF API from a module that
  does not define it, and `.v7_autonomous_research` (a module that does not
  exist; the real package is `.autonomous_research`). 63 of 176 top-level names
  were `None` at runtime, including the whole V7 autonomous-research pipeline.
  Now 5, all intentional.
- **Silent degradation made loud** — 254 `except Exception: NAME = None` blocks
  across 29 files now log which module degraded and why; 173 missing `NAME = None`
  fallbacks were completed so an `except` branch defines every name its `try`
  branch imports.
- **V4.0 capabilities** — `V4IntegrationCoordinator.process_query()` invoked all
  four capabilities through method names that do not exist, so every call fell
  into its `except` branch and `used_capabilities` was always empty. Rewired to
  the real APIs.
- **Test harnesses** — `comprehensive_system_test.py` never called `sys.exit`;
  `test_comprehensive_integration.py` was structurally incapable of failing;
  `test_installation.py` was structurally incapable of passing; four `pytest`
  entry points discarded the exit status. All fixed.
- **Packaging** — added `pyproject.toml` and `requirements.txt`. `README.md`
  documented `pip install -e .` although no build metadata existed, and
  `astropy`, `torch`, `reportlab` and `psutil` were undeclared.
- **Physics** — confirmed numerical defects repaired with regression tests; see
  `astra_core/tests/test_physics_regressions*.py` and the `# FIX(audit …)`
  comments at each site.

## Version Notes

This baseline represents ASTRA after the August 2026 integrity audit and rebuild:

- **17 formerly lost symbols re-implemented** (partially verified: the LVG
  escape-probability solver and the SPH kernels check out numerically; the
  Bayesian swarm inference returned fabricated zero-width error bars and was
  repaired in the re-audit) (LVG escape-probability solver, SPH kernels normalized exactly, Bayesian swarm inference, MoE routing, self-teaching system, and more)
- **14 `astro_physics` modules rebuilt** — re-audit found that several of the
  fourteen still contained critical defects (an inverted Truelove refinement
  criterion, an unstable advection update, a CGS/SI mix in the interferometry uv
  plane); those are fixed and regression-tested here
- **~13,500 dead lines removed** (self-evolution duplicates, no-op stubs, 8 identical file pairs) with zero behavioural change
- **Stale import paths repaired** across ~50 sites; `import astra_core.core` raises zero warnings
- Three pre-existing physics bugs fixed in place (phantom `scipy.ndimage.skeletonize` import, two mis-normalized SPH kernels, a dimensional error in the modified-blackbody flux)
- `User_Manual` v7.2 with the complete audit record in Appendix E

The previous baseline (April 2026) reflected the `stan_core` → `astra_core` renaming and version-number cleanup.

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).

    Copyright 2026 Glenn J. White

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

## Contact

For questions about this baseline archive, please refer to the main ASTRA repository at:
https://github.com/Tilanthi/ASTRA

---

**Note**: This is a static baseline archive. For the latest development version of ASTRA, please see the main repository.
