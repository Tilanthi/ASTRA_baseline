# ASTRA User Manual
## Autonomous Scientific Discovery in Astrophysics

**Version**: 7.3 (correction pass, August 2026)
**Date**: August 2026
**Authors**: Glenn J. White, Open University and Rutherford Appleton Laboratory, England
**Repository**: https://github.com/Tilanthi/ASTRA

---

> ### Status of this document — please read first
>
> Versions 7.0–7.2 of this manual described an **aspirational** system. Eighteen documented
> methods and two documented import paths did not exist in the codebase at all, and several
> worked examples printed invented numbers. This edition is a **correction pass**: the prose
> is unchanged wherever it was already true, and every incorrect claim has been either
> repaired or explicitly marked as not implemented.
>
> **Every Python code block in this manual has been executed against the current tree.** The
> printed outputs are the real outputs. Blocks that merely display an API signature are
> fenced as `text` blocks so that they are not mistaken for runnable code.
>
> Three practical consequences:
>
> 1. **Run from the repository root**, or set `PYTHONPATH` to it. The package is not
>    installed on `sys.path` by the examples below:
>    `cd /path/to/ASTRA && PYTHONPATH=. python3 your_script.py`.
> 2. **The examples are cumulative.** A block may use `system`, `registry` or `scientist`
>    created in an earlier block of the same section chain.
> 3. **`astra_core.__version__` reports `4.0.0`**, not 7.x. The "V7.0" in the product name
>    refers to the autonomous-research subpackage, not to the package version string. Both
>    numbers are reported below by executable code rather than asserted here.
>
> Where a capability described in earlier editions has no implementation, this manual now
> says so in a **Not implemented** box and, wherever one exists, points at a real, tested
> routine that does the equivalent job. A short, honest example is more use than an
> impressive fictional one.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Installation and Setup](#3-installation-and-setup)
   - 3.1 System Requirements
   - 3.2 Installation Methods
   - 3.3 Configuration
   - 3.4 Driving ASTRA from an agentic coding CLI
4. [Getting Started](#4-getting-started)
5. [Core Capabilities Overview](#5-core-capabilities-overview)
6. [V5.0 Discovery Enhancement System](#6-v50-discovery-enhancement-system)
7. [V7.0 Autonomous Research Scientist](#7-v70-autonomous-research-scientist)
8. [Use Case Examples](#8-use-case-examples)
9. [Advanced Features](#9-advanced-features)
10. [Domain Modules](#10-domain-modules) — the 31 / 17 / 27 split and derived confidence
11. [API Reference](#11-api-reference) — signatures taken from live objects
12. [Best Practices](#12-best-practices)
13. [Troubleshooting](#13-troubleshooting)
14. [Appendices](#14-appendices) — **E.8 is the current audit status**

---

## 1. Introduction

### 1.1 What is ASTRA?

ASTRA (Autonomous Scientific Discovery in Astrophysics) is an integrated computational framework that combines numerical data analysis, causal reasoning, physical validation, and statistical inference to enable automated scientific discovery in astrophysics. Unlike traditional machine learning systems that detect patterns without understanding their physical meaning, or large language models that can explain concepts but cannot process numerical data, ASTRA integrates multiple analytical approaches to provide physically interpretable, validated scientific insights.

**Version 7.0** introduced the **Autonomous Research Scientist** subpackage
(`astra_core.autonomous_research`), which runs a seven-stage pipeline from question
generation to a draft manuscript object. Section 7 documents what that pipeline actually
produces, which is a *structured template* rather than a piece of original science: the
stages execute and return well-formed objects, but the scientific content is generated from
fixed templates and is not derived from data. It is a workflow skeleton, not a scientist.

### 1.2 Key Design Principles

These are the principles the codebase is *designed* around. The right-hand column records how
far each one is realised in the tree you have, as measured in the August 2026 audit
(Appendix E).

**Physics-Aware Reasoning**: numerical routines in `astra_core.astro_physics` are pinned by
regression tests against hand-derived values, and 121 of them are exposed as domain
capabilities that carry their formula and their verifying test (Section 10). This principle
is realised at the level of the individual routine. There is *no* global validator that
checks an arbitrary answer against conservation laws.

**Causal Understanding**: real. `capabilities/causal_discovery.py` implements PC, GES and a
hybrid over partial-correlation and mutual-information independence tests;
`reasoning/astrophysical_causal_discovery.py` adds Malmquist and Eddington bias detection.
Both are demonstrated in Section 5.

**Uncertainty Quantification**: partial. Fitting routines
(`sed_fitting`, `uncertainty_quantification`, `star_formation.fit_ks`) return parameter
covariances; the conversational layer returns a `confidence` scalar which, inside the domain
system, is *derived from provenance* and cannot be hard-coded (Section 10.3). Outside the
domain system, several `confidence` values are still defaults rather than measurements —
those are flagged where they appear.

**Reproducibility**: partial. Domain results carry `metadata['provenance']`,
`metadata['inputs']`, the formula and the name of the verifying test. There is no end-to-end
provenance graph from raw file to conclusion.

**Autonomous Research**: the V7.0 pipeline runs end to end, but its scientific content is
template-generated (Section 7). Treat it as scaffolding for a human-driven cycle.

### 1.3 Who Should Use This Manual?

This manual is written for expert users including:
- Research astronomers and astrophysicists
- Data scientists working with astronomical data
- Computational scientists requiring physics-aware analysis tools
- Graduate students and postdoctoral researchers in astrophysics

Users should have familiarity with:
- Python programming
- Basic statistical concepts
- Fundamental astrophysical principles
- Command-line operation

### 1.4 What's New in Version 7.0

The V7.0 feature list is given below with the state of each component in the current tree.
"Runs" means the code executes and returns a well-formed object; it does not mean the output
is scientifically meaningful.

**V7.0 Autonomous Research Scientist** (`astra_core.autonomous_research`):

| Component | State | Section |
|---|---|---|
| Question generation engine | Runs; question text comes from a fixed per-domain table and falls back to a cosmology list for most domains | 7.2.1 |
| Hypothesis formulation system | Runs; statements are templates keyed on the question's topic word | 7.2.2 |
| Experiment design and execution | Runs; returns three design dicts (observational / simulation / archival) with template costs and durations. No experiment is actually executed against data | 7.2.3–7.2.4 |
| Automated theory revision | Runs; returns a revision record | 7.2.6 |
| Publication generation | Runs; returns a `Publication` object with title, abstract, figures and tables. Note the `sections` the engine builds are **dropped** by the `Publication` dataclass | 7.2.7 |

**Enhanced Capabilities**:

| Component | State | Section |
|---|---|---|
| Multi-Mind Orchestration (7 specialised minds) | The seven minds exist and are selected by the arbitrator, but each returns the template string `"<Discipline> analysis of: <query>"` with a fixed confidence of 0.7 | 9.1 |
| Global coherence layer | `GlobalCoherenceLayer` exists with `maintain_consistency()` and `global_state_management()` | 9.2 |
| Analogical reasoning | Real, but the phenomenon database ships with **three** entries (`accretion_disk`, `stellar_oscillation`, `planetary_rings`) and matches on exact names | 9.3 |
| Continuous learning | `ContinuousLearning` class exists; `monitor_literature()` performs no network access | — |
| Scientific taste evaluation | `ScientificTaste` class exists with heuristic scoring | — |

---

## 2. System Architecture

### 2.1 Architectural Overview

ASTRA implements a layered architecture designed for astrophysical data analysis and inference:

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                    │
│  (Command line, Python API, Jupyter notebooks)             │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Orchestration Layer                       │
│  Query processing, module selection, result integration     │
└─────┬───────────┬───────────┬───────────┬───────────┬─────┘
      │           │           │           │           │
┌─────▼─────┐ ┌─▼──────┐ ┌─▼──────┐ ┌─▼──────┐ ┌─▼────────┐
│  Physics  │ │ Causal │ │Bayesian│ │ Data   │ │ Domain   │
│  Engine   │ │Reasoning│ │Inference│ │Processing│Knowledge│
└───────────┘ └────────┘ └────────┘ └────────┘ └──────────┘
      │           │           │           │           │
      └───────────┴───────────┴───────────┴───────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              V7.0 Autonomous Research Layer                 │
│  Question → Hypothesis → Experiment → Analysis → Theory   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Foundation Layer                         │
│  Memory systems, I/O handling, numerical libraries         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Core Components

#### 2.2.1 Physics Engine

`astra_core.physics.UnifiedPhysicsEngine` holds eight registered closed-form models
(`newtonian_gravity`, `schwarzschild_metric`, `orbital_velocity`, `blackbody`, `planck_law`,
`stefan_boltzmann`, `ideal_gas`, `virial_theorem`), reachable through
`system.compute_physics()` (Section 4.1, Example 3). **It works in CGS**; see Appendix C,
which also documents the second, SI, constant set in `astro_physics.physics`. The
`constraint_violations` dictionary it returns is populated by placeholder checks that return
0.0 unless the caller supplies `energy_initial`/`energy_final`-style keys, so a zero there
means "not checked", not "verified".

#### 2.2.2 Causal Reasoning Module

Real and tested. `capabilities/causal_discovery.py` (PC, GES, hybrid; partial-correlation and
mutual-information independence tests) and `reasoning/astrophysical_causal_discovery.py`
(Malmquist/Eddington bias detection, physics-constrained graphs, mechanism identification).
Worked examples in Sections 5.1.1 and 5.1.3.

#### 2.2.3 Bayesian Inference Engine

`astro_physics/uncertainty_quantification.py` provides priors, Gaussian likelihoods,
Metropolis–Hastings, an affine-invariant ensemble sampler, a nested sampler and a Fisher
matrix. `astro_physics/inference.py` provides `BayesianSwarmInference`. These are libraries;
they are not wired into the conversational layer.

#### 2.2.4 Data Processing Pipeline

`astro_physics/data_interface.py` (FITS, spectral cubes, VOTable, region files) and
`capabilities/safe_fits_reader.py`. There is no automatic ingestion or calibration pipeline:
you load and pass arrays yourself, as the examples in Sections 5 and 8 do.

#### 2.2.5 Domain Knowledge Systems

75 registered domain modules, of which 31 carry executable numerical capabilities, 27 return
curated reference text and 17 honestly report that they have no implementation. This is the
part of the system where "confidence" is derived rather than asserted; see Section 10.

---

## 3. Installation and Setup

### 3.1 System Requirements

**Minimum Requirements**:
- Python 3.8 or higher
- 8 GB RAM
- 2 GB free disk space
- Linux, macOS, or Windows with WSL2

**Recommended for Large Datasets**:
- Python 3.10 or higher
- 32 GB RAM
- 20 GB free disk space
- SSD storage for better I/O performance
- Multi-core processor (4+ cores)

**Hard requirements** (declared in `pyproject.toml`; the package will not import without
them): `numpy>=1.24`, `scipy>=1.10`, `scikit-learn>=1.2`, `networkx>=3.0`, `sympy>=1.12`,
`pandas>=2.0`, `psutil>=5.9`.

**Optional extras** (each maps to a real `[project.optional-dependencies]` group):

| Extra | Pulls in | Needed for |
|---|---|---|
| `astro` | `astropy` | FITS/VOTable I/O in `astro_physics.data_interface`, `safe_fits_reader` |
| `dl` | `torch` | deep-learning models in `astro_physics.deep_learning` |
| `pdf` | `reportlab`, `pillow` | `utils.pdf_generator` (without it you will see the `reportlab not available` warning printed by every example in this manual — it is harmless) |
| `science` | `matplotlib`, `emcee`, `corner`, `statsmodels` | plotting and MCMC helpers |
| `image` | `scikit-image` | morphological skeletonisation in `FilamentFinder`; without it a built-in Zhang–Suen thinning fallback is used |
| `dev` | `pytest`, `pyflakes` | running the test suite |
| `all` | all of the above | — |

### 3.2 Installation Methods

For a step-by-step install with a virtual environment — including macOS notes and the output
each command should print — see **`INSTALL.md`** in the repository root. The summary below is
enough if you already work in Python.

#### 3.2.1 Installation from GitHub

```bash
# Clone the repository
git clone https://github.com/Tilanthi/ASTRA.git
cd ASTRA

# Install in editable mode
pip install -e .

# Everything, including optional extras
pip install -e ".[all,dev]"
```

If you do not install the package, run from the repository root with `PYTHONPATH=.`; every
example in this manual was verified that way.

#### 3.2.2 Verification of Installation

```bash
python3 -c "import astra_core; print('ASTRA imports')"
```

The following block is executable and prints the real version string. Note that it reports
`4.0.0` (the value in `pyproject.toml`), **not** `7.x`: the package version and the "V7.0"
product name are different things, and earlier editions of this manual conflated them.

```python
import astra_core
print("ASTRA imported from:", astra_core.__file__)
print("astra_core.__version__ =", astra_core.__version__)
```

```text
astra_core.__version__ = 4.0.0
```

For a fuller check, run the two suites the audit uses (Appendix E.7):

```bash
PYTHONPATH=. python3 astra_core/comprehensive_system_test.py   # 18/18
PYTHONPATH=. python3 -m pytest -q                              # 333 passed
```

> **The `PYTHONPATH=.` is not optional.** Without it `comprehensive_system_test.py` reports
> `Passed: 0 (0.0%)` — and still exits 0. See Appendix E.8.

### 3.3 Configuration

#### 3.3.1 Basic Configuration

> **Not implemented.** Earlier editions instructed you to create `~/.astra/config.json` with
> keys `data_directory`, `memory_limit_gb`, `num_workers` and `log_level`. **No code in the
> tree reads that file, and none of those keys exists.** Creating it has no effect.

Configuration is done in Python, through the dataclass passed to the factory:

```text
from astra_core.core.unified_enhanced import EnhancedUnifiedConfig, create_enhanced_stan_system

config = EnhancedUnifiedConfig(enable_orchestration=False)   # real field
system = create_enhanced_stan_system(config)
```

The real fields include `enable_orchestration`, `orchestration_preload_threshold`,
`orchestration_max_preloaded` and `orchestration_optimization_interval`. The factory also
accepts a plain `dict`. Logging is configured with the standard library `logging` module;
ASTRA is very chatty at `INFO` level, so `logging.disable(logging.INFO)` is worth setting
before an interactive session.

### 3.4 Driving ASTRA from an agentic coding CLI

> **Correction.** Earlier editions of this section described a **Claude Code integration that
> does not exist**. Specifically, none of the following is real: a `claude-code --astra`
> flag; `claude-code config set astra.enabled true`; `astra.memory_limit`; an
> `npm install -g @anthropic/claude-code` package name; a Homebrew formula `claude-code`; an
> "ASTRA V7.0 initialized. Ready for autonomous research." banner; and the worked transcript
> that reported "0.103 ± 0.008 pc across 5,476 filaments" and "Confidence: 92%". **Those
> numbers were never computed by this software.** ASTRA ships **no console entry point at
> all**: `pyproject.toml` declares no `[project.scripts]`, and there is no `__main__.py`.
>
> What *is* true, and is worth keeping, is the underlying point: ASTRA is a Python library,
> so any agentic coding CLI that can run Python in your checkout can drive it. That is a
> property of the CLI, not an ASTRA feature. Nothing in ASTRA detects or configures one.

#### 3.4.1 The realistic pattern

Point the agent at the repository, tell it to run from the repository root with
`PYTHONPATH=.`, and have it write and execute ordinary scripts against the API documented in
Sections 4, 5, 8 and 10. The repository ships a `CLAUDE.md` at its root describing the
codebase layout for exactly this purpose.

```bash
cd /path/to/ASTRA
# ...start your agentic CLI here, in this directory...
```

A useful opening instruction is a factual one, because the agent can verify it:

> "This is the ASTRA repository. Run everything from this directory with `PYTHONPATH=.`.
> The domain registry is the only entry point that returns computed numbers with derived
> confidences — see `User_Manual/User_Manual.md` §10.3. Do not use `system.answer()` for
> quantitative work; §4.1 explains why."

#### 3.4.2 What a real session looks like

The script below is a complete, executable ASTRA session of the kind an agent would write.
It is the honest version of the transcript that used to appear here: same question, real
numbers, and an explicit statement of what the system did *not* do.

```text
$ cd /path/to/ASTRA
$ PYTHONPATH=. python3 - <<'EOF'
from astra_core.domains import DomainRegistry
registry = DomainRegistry()
registry.auto_load_domains({'statistical_mechanics': {}, 'mhd': {}})
for q in ["Jeans length for n = 1e4 cm^-3 and T = 10 K",
          "sonic scale for velocity_dispersion = 1.0 km/s, driving_scale = 2 pc, T = 10 K"]:
    r = registry.process_query(q)
    print(f"{r['domain']:22s} {r['confidence']:.2f}  {r['answer'].splitlines()[0]}")
EOF
statistical_mechanics  0.90  Jeans length of an isothermal self-gravitating gas: lambda_J = 0.2118 pc (formula: lambda_J = c_s sqrt(pi / (G rho)))
mhd                    0.90  Scale at which supersonic turbulence becomes subsonic: l_s = 0.0709 pc (formula: l_s = L M_s^(-1/p) from sigma(l) = sigma_L (l/L)^p with p = 0.5 (Larson linewidth-size law))
```

No filaments were detected, no survey was read, and no hypothesis was tested — two closed-form
expressions were evaluated on the numbers in the query, and the confidence of 0.90 means
exactly "a verified routine ran on parsed inputs" (Section 10.3).

#### 3.4.3 Guidance for agent-driven work

1. **Insist on execution.** Ask the agent to run the code and paste real output, not to
   describe what it would print. Every number in this manual was obtained that way.
2. **Prefer the domain registry** for quantitative questions (Section 10.3) and
   `astra_core.astro_physics` directly for anything involving data arrays (Sections 5 and 8).
3. **Treat any `confidence` outside the domain system as a default**, not a measurement.
4. **Ask for the provenance.** `result['metadata']['provenance']` distinguishes `computed`
   (0.90) from `capability_available` (0.40), `descriptive` (0.20) and `none` (0.0).
5. **Do not ask it for a paper.** Section 7.2.7 explains what the publication engine really
   produces.

#### 3.4.4 File operations

ASTRA can read FITS, VOTable and region files through
`astra_core.astro_physics.data_interface` and `astra_core.capabilities.safe_fits_reader`
(both need the `astro` extra). Results are ordinary Python objects; serialise them yourself
with `json` or `numpy.savez`. There is no ASTRA-level "save results" command.

#### 3.4.5 Troubleshooting

**`ModuleNotFoundError: No module named 'astra_core'`** — you are not in the repository root
and have not installed the package. Use `cd /path/to/ASTRA && PYTHONPATH=. python3 ...`.

**Domain modules "not loading"** — the diagnostic printed in earlier editions,
`from astra_core.domains import list_domains`, does not exist; `list_domains` is a *method* of
`DomainRegistry`. The working check is Section 10.2.

**A domain answers with prose when you wanted a number** — the query did not name the
computation, or did not supply its parameters. The reply lists the available signatures; see
Section 10.3.

**Memory** — `astra.memory_limit` is not a real setting. The system holds no large arrays of
its own; memory is dominated by whatever data *you* load.

---

## 4. Getting Started

### 4.1 Your First Analysis

#### Example 1: Create the system

`create_stan_system()` returns an `EnhancedUnifiedSTANSystem`. Pass
`auto_start_orchestrator=False`: with the orchestrator running, **every** call to
`answer()`/`process_query()` is intercepted and returns the fixed string
`"Query processed through orchestration system"` with a default confidence of 0.7, whatever
you asked. That is a real behaviour of the current tree, not a documentation simplification.

```python
from astra_core import create_stan_system

system = create_stan_system(auto_start_orchestrator=False)
print(type(system).__name__)
print("domains registered:", len(system.list_domains()))
print("physics models:", system.list_physics_models())
```

```text
EnhancedUnifiedSTANSystem
domains registered: 75
physics models: ['newtonian_gravity', 'schwarzschild_metric', 'orbital_velocity',
                 'blackbody', 'planck_law', 'stefan_boltzmann', 'ideal_gas',
                 'virial_theorem']
```

#### Example 2: Ask a quantitative question

Use `system.domain_registry.process_query()`. This is the path that reaches the computational
domains and returns a number with a *derived* confidence. (Continues from Example 1.)

```python
result = system.domain_registry.process_query(
    "Jeans mass for n = 1e4 cm^-3 and T = 10 K"
)
print(result['domain'])
print(result['answer'])
print("confidence:", result['confidence'])
print("provenance:", result['metadata']['provenance'])
print("inputs actually parsed:", result['metadata']['inputs'])
```

```text
statistical_mechanics
Jeans mass of an isothermal self-gravitating gas: M_J = 2.8680 Msun
    (formula: M_J = (pi^(5/2)/6) c_s^3 G^(-3/2) rho^(-1/2))
confidence: 0.9
provenance: computed
inputs actually parsed: {'density': 10000.0, 'temperature': 10.0}
```

> **Do not use `system.answer()` for quantitative work.** `answer()` is an alias for
> `process_query()`, whose `mode='domain'` branch uses a *second*, private router
> (`EnhancedUnifiedSTANSystem._find_relevant_domain`) that ranks domains purely by raw
> keyword count and breaks ties by registration order. For the query above it selects `ism`
> and returns curated reference text at confidence 0.20 — while the registry's own
> repaired router selects `statistical_mechanics` and returns `M_J = 2.8680 Msun` at 0.90.
> Until the private router is removed, go through `system.domain_registry` (or a
> `DomainRegistry` of your own, Section 10.2). This defect is recorded in Appendix E.8.

#### Example 3: Closed-form physics

`compute_physics()` evaluates one of the eight registered models. **The physics engine works
in CGS.** Give it grams, centimetres and seconds, or you will get an answer that is wrong by
powers of a thousand without any warning.

```python
print(system.compute_physics('stefan_boltzmann', {'temperature': 5778.0})['value'])
print(system.compute_physics('orbital_velocity',
                             {'mass': 1.989e33, 'radius': 1.496e13})['value'])
```

```text
63196526546.0292          # erg cm^-2 s^-1  (sigma T^4 for the solar T_eff)
2978822.982930735         # cm s^-1 = 29.79 km/s  (Earth's orbital velocity)
```

The parameter *names* must match the model's signature (Section 11.1); an unrecognised
keyword makes `compute_physics` return `value=None` rather than raise, so always check the
returned value is not `None`.

### 4.2 Understanding ASTRA's Output

A domain result is a plain dictionary. The keys below are the ones that are always present
and always meaningful:

```python
r = system.domain_registry.process_query("free-fall time for n = 1e4 cm^-3")
for key in ('success', 'domain', 'confidence'):
    print(key, '=', r[key])
print('reasoning_trace =', r['reasoning_trace'])
print('capabilities_used =', r['capabilities_used'])
print('metadata keys =', sorted(r['metadata']))
```

```text
success = True
domain = statistical_mechanics
confidence = 0.9
reasoning_trace = ['Selected capability: free_fall_time(density [cm^-3]) -> t_ff [yr]',
                   'Computed t_ff = np.float64(337077.6048485482)']
capabilities_used = ['free_fall_time']
metadata keys = ['computation', 'implementation_status', 'inputs', 'provenance',
                 'reference', 'units', 'value', 'verified_by']
```

| Field | What it really tells you |
|---|---|
| `answer` | Formatted sentence containing the value, its unit and the formula used |
| `confidence` | **Derived from provenance, never asserted**: 0.90 computed, 0.40 capability available but parameters missing, 0.20 curated text, 0.0 no implementation |
| `metadata['provenance']` | `computed` / `capability_available` / `descriptive` / `none` |
| `metadata['inputs']` | The parameters actually parsed out of your query. If a number you supplied is missing here, it was **not** used |
| `metadata['value']`, `['units']` | The raw number and its unit, for programmatic use |
| `metadata['reference']` | The formula, so you can check it |
| `metadata['verified_by']` | The regression test that pins this routine's output |
| `reasoning_trace` | Which capability was selected and what it computed |
| `capabilities_used` | Empty unless something was actually executed |

There is no "Recommendations" field, and no automatic physical-constraint validation of an
arbitrary answer; earlier editions listed both.

---

## 5. Core Capabilities Overview

> **Correction.** Earlier editions presented this section as four one-line calls on the
> system object: `system.detect_bias(...)`, `system.discover_scaling_relation(...)`,
> `system.perform_causal_inference(...)` and `system.fuse_multiwavelength(...)`.
> **None of those four methods exists**, and none ever did. `EnhancedUnifiedSTANSystem`
> exposes exactly: `answer`, `process_query`, `compute_physics`, `find_analogies`,
> `get_domain_info`, `list_domains`, `list_physics_models`, `adapt_to_domain`,
> `learn_physics_curriculum`, `get_system_status`, `get_orchestration_metrics`,
> `start_orchestrator`, `stop_orchestrator`.
>
> Three of the four capabilities are nevertheless **real** — they live in library modules and
> have simply never been surfaced on the system object. This section now shows the working
> calls. The fourth (catalogue cross-matching) is not implemented, and says so.
>
> The claim "ASTRA integrates 20+ analytical capabilities" is retired: see Appendix A for the
> audited list.

All examples below continue from `system` created in Section 4.1 and use `numpy` on
synthetic data with known ground truth, so you can see whether the routine recovers it.

### 5.1 Causal and Statistical Analysis

#### 5.1.1 Bias Detection

**Real** — `astra_core.reasoning.astrophysical_causal_discovery.BiasAwareCausalDiscovery`,
with `detect_malmquist_bias`, `detect_eddington_bias`, `correct_malmquist` and
`discover_with_bias_correction`. There is no generic `bias_type="..."` dispatcher; you call
the specific detector.

The test below builds a population with a *fixed* luminosity function (so there is no true
luminosity–distance correlation), then applies a flux limit. A correct detector should find
no bias in the parent sample and a large bias in the flux-limited one.

```python
import numpy as np
from astra_core.reasoning.astrophysical_causal_discovery import BiasAwareCausalDiscovery

rng = np.random.default_rng(0)
distance = rng.uniform(10.0, 200.0, 2000)                 # Mpc
luminosity = 10 ** rng.normal(10.0, 0.4, 2000)            # Lsun
flux = luminosity / (4 * np.pi * distance ** 2)
flux_limit = float(np.percentile(flux, 60))
detected = flux > flux_limit

bias = BiasAwareCausalDiscovery()
print("volume-limited:", bias.detect_malmquist_bias(flux, distance, flux_limit))
print("flux-limited:  ", bias.detect_malmquist_bias(flux[detected],
                                                    distance[detected],
                                                    flux_limit))
```

```text
volume-limited: (np.False_, 0.0)
flux-limited:   (np.True_, np.float64(0.3802304503553361))
```

The return is `(bias_present, bias_magnitude)`, where the magnitude is the fitted
mean-luminosity trend across distance quartiles normalised by the mean luminosity. Read the
source before quoting it: the detection threshold is the hard-coded heuristic
`trend > 0.1 * mean(L)`, not a significance test.

#### 5.1.2 Scaling Relations Discovery

> **Not implemented as advertised.** There is no general "give me two columns and I will
> discover the scaling law" routine, and nothing in the tree produces the
> `L ∝ T^(2.5±0.3)` output shown in earlier editions.

**What exists instead** is a specific, correctly implemented power-law fit for the
Kennicutt–Schmidt relation, `StarFormationLaw.fit_ks`, which returns the index, the
normalisation, their 1σ errors and the scatter. Below it is given data drawn from
Σ_SFR = 2.5×10⁻⁴ Σ_gas^1.4 with 0.12 dex of scatter.

```python
from astra_core.astro_physics.star_formation import StarFormationLaw

rng = np.random.default_rng(7)
sigma_gas = 10 ** rng.uniform(0.5, 3.0, 60)                       # Msun/pc^2
sigma_sfr = 2.5e-4 * sigma_gas ** 1.4 * 10 ** rng.normal(0, 0.12, 60)

ks = StarFormationLaw.fit_ks(sigma_gas, sigma_sfr)
print(f"N = {ks['N']:.2f} +/- {ks['N_error']:.2f}   (input 1.40)")
print(f"A = {ks['A']:.3e}                (input 2.5e-04)")
print(f"scatter = {ks['scatter_dex']:.3f} dex        (input 0.12)")
```

```text
N = 1.42 +/- 0.02   (input 1.40)
A = 2.248e-04                (input 2.5e-04)
scatter = 0.101 dex        (input 0.12)
```

For an arbitrary power law, fit it yourself — `numpy.polyfit(log x, log y, 1, cov=True)` is
what `fit_ks` does — or use the samplers in
`astro_physics.uncertainty_quantification` if you need posteriors rather than a covariance.

#### 5.1.3 Causal Inference

**Real** — `astra_core.capabilities.causal_discovery.CausalDiscoveryEngine`, implementing PC,
GES and a hybrid, over `PartialCorrelationTest` and `MutualInformationTest`. Below, the data
are generated from the chain mass → sfr → luminosity.

```python
from astra_core.capabilities.causal_discovery import CausalDiscoveryEngine

rng = np.random.default_rng(11)
n = 400
mass = rng.normal(0, 1, n)
sfr = 0.8 * mass + rng.normal(0, 0.3, n)
lum = 0.9 * sfr + rng.normal(0, 0.3, n)
data = np.column_stack([mass, sfr, lum])

graph = CausalDiscoveryEngine(algorithm='pc').discover(
    data, ['mass', 'sfr', 'luminosity'])
for e in graph.edges:
    print(e.source, e.edge_type.value, e.target)
```

```text
mass undirected sfr
sfr undirected luminosity
```

The skeleton is recovered exactly and the spurious mass–luminosity edge is correctly removed.
The edges come back **undirected**, which is the honest answer: a chain and its reverse are
Markov-equivalent, and with no v-structure the orientation is not identifiable from
observational data alone. Do not read a direction into this output. To orient edges you must
add background knowledge — `PhysicsConstrainedGraph` in
`reasoning/astrophysical_causal_discovery.py` accepts forbidden-edge constraints for that
purpose.

### 5.2 Data Integration and Analysis

#### 5.2.1 Multi-Wavelength Fusion

> **Not implemented.** There is no `fuse_multiwavelength`, and **no positional
> cross-matching anywhere in the tree** — the `matching_radius=2.0` arcsec argument shown in
> earlier editions corresponds to no code. If you need to merge catalogues by sky position,
> use `astropy.coordinates.SkyCoord.match_to_catalog_sky` before handing the merged table to
> ASTRA.

**What exists instead** is genuine multi-wavelength *modelling*: given photometry that is
already matched, `astro_physics.sed_fitting` fits a composite SED (dust, stars, AGN,
synchrotron) by differential evolution with Hessian-based uncertainties. Here a modified
blackbody with T = 25 K, β = 1.8, M_dust = 10⁸ M⊙ at 50 Mpc is sampled in seven far-IR/submm
bands with 5 % errors and refitted.

```python
from astra_core.astro_physics.sed_fitting import CompositeSED, SEDFitter

sed = CompositeSED(distance_Mpc=50.0)
sed.add_dust()
truth = {'T_dust': 25.0, 'beta': 1.8, 'M_dust': 1e8,
         'kappa_0': 0.77, 'lambda_0': 850.0}
lam_A = np.array([70., 100., 160., 250., 350., 500., 850.]) * 1e4   # micron -> Angstrom
model = sed.evaluate(lam_A, truth)['total']
err = 0.05 * model
obs = model + np.random.default_rng(3).normal(0, err)

fit = SEDFitter(sed).fit(
    lam_A, obs, err,
    fixed={'kappa_0': 0.77, 'lambda_0': 850.0},
    bounds={'T_dust': (5., 80.), 'beta': (0.5, 3.0), 'M_dust': (1e6, 1e10)},
    log_params=['M_dust'], flux_unit='cgs')
for p in ('T_dust', 'beta', 'M_dust'):
    print(f"{p:8s} = {fit.best_params[p]:.4g} +/- {fit.param_errors[p]:.3g}"
          f"   (input {truth[p]:g})")
print("reduced chi^2 =", round(fit.reduced_chi_squared, 2))
```

```text
T_dust   = 25.33 +/- 0.573   (input 25)
beta     = 1.803 +/- 0.0638   (input 1.8)
M_dust   = 9.279e+07 +/- 3.42e+06   (input 1e+08)
reduced chi^2 = 2.79
```

Two traps, both learned the hard way:

* **Wavelengths are in Ångström and fluxes default to CGS** (erg s⁻¹ cm⁻² Hz⁻¹). Passing Jy
  without `flux_unit='Jy'` is a factor of 10²³ and the fitter will happily converge on
  nonsense — a documented historical failure of this routine.
* **`kappa_0` must be fixed.** It is exactly degenerate with `M_dust`; leave it free and
  `M_dust` rails against its bound with an error bar larger than the value itself, while
  T and β still look perfect.

---

## 6. V5.0 Discovery Enhancement System

### 6.1 Overview

> **Correction.** Earlier editions documented `system.genuine_discovery(...)` and
> `system.discover_physical_model(...)`. **Neither method exists**, on this or any other
> object in the tree; nor do the arguments they took (`knowledge_base="astrophysics_ontology"`
> — there is no such ontology — `novelty_threshold`, `model_space=[...]`).
>
> What exists is `astra_core.legacy.systems.v50.v50_discovery_engine`, a **legacy** module
> providing `V50DiscoveryEngine` with `discover(phenomenon, domain="")`,
> `answer(question, domain="", choices=None)` and `get_stats()`, plus the factories
> `create_v50_standard/_fast/_deep/_discovery/_gpqa`. Its `discover()` runs, but you should
> see what it returns before relying on it.

### 6.2 What `discover()` actually returns

```python
from astra_core.legacy.systems.v50.v50_discovery_engine import create_v50_standard

v50 = create_v50_standard()
out = v50.discover("What sets the width of interstellar filaments?")
print(sorted(out))
print("hypotheses returned:", len(out['hypotheses']))
print("causal models returned:", len(out['causal_models']))
print("simulation predictions:", out['simulations'][0]['predictions'])
```

```text
['abstractions', 'causal_models', 'cross_domain_analogies', 'domain', 'hypotheses',
 'phenomenon', 'simulations']
hypotheses returned: 0
causal models returned: 0
simulation predictions: {'final_position': [4.99, 0.0, -112.13], 'final_velocity': [1.0, 0.0, -48.95],
                         'final_energy': 98.60, 'max_height': 10, 'total_distance': 122.39}
```

Read that output carefully. Asked about interstellar filament widths, the engine classifies
the domain as "Physics", returns **zero** hypotheses and **zero** causal models, and reports
a "simulation" whose predictions are the trajectory of a ballistic projectile — a fixed demo
that has nothing to do with the question. The accompanying `confidence: 0.9` is a literal in
the source.

**Recommendation: do not use the V5.0 engine for scientific work.** For the tasks this
section used to claim — finding relationships in data, and choosing between candidate
functional forms — use the routines in Section 5.1.3 (causal structure) and Section 5.1.2 /
`astro_physics.uncertainty_quantification` (model fitting and comparison), which compute
real answers from your data.

---

## 7. V7.0 Autonomous Research Scientist

### 7.1 Overview

The V7.0 Autonomous Research Scientist runs a seven-stage pipeline:

```
Question → Hypothesis → Experiment → Analysis → Theory → Publication
```

> **Correction — import path and method names.** Earlier editions used
> `from astra_core.v7_autonomous_research import create_v7_scientist`. **There is no
> `astra_core.v7_autonomous_research` module**; the correct path is
> `astra_core.autonomous_research`. Five of the seven documented methods were also named
> incorrectly:
>
> | Documented (wrong) | Real method |
> |---|---|
> | `design_experiment(hypothesis, ...)` | `design_experiments(hypothesis, constraints=None) -> List[Dict]` |
> | `execute_experiment(experiment, ...)` | `execute_experiments(experiments, parallel=True) -> List[Dict]` |
> | `analyze_results(results, ...)` | `analyze_and_predict(results, hypothesis) -> Dict` |
> | `generate_publication(research_cycle, ...)` | `write_publication(research_summary, target_journal=...) -> Publication` |
> | `conduct_autonomous_research(...)` | `conduct_full_research_cycle(domain, research_context=None)` |
>
> `generate_research_questions`, `formulate_hypotheses`, `revise_theory` and `get_status` were
> named correctly. `conduct_guided_research` and `collaborative_research` **have no
> equivalent at all** (Section 7.4).
>
> **Correction — what it produces.** The example outputs printed in earlier editions
> ("547 filaments measured across 5 clouds", "Mean width: 0.103 ± 0.008 pc",
> "Width ∝ M^(-0.84±0.15)", "p < 0.001", "η² = 0.73", "Bayes factor > 100") were **never
> generated by this software.** The real outputs are shown below. The pipeline is a
> well-formed *workflow skeleton*: every stage runs and returns a structured object, but the
> scientific content is drawn from fixed templates, no data are read, and no experiment is
> executed. Use it to prototype a research loop, not to do research.

### 7.2 Core Components

```python
from astra_core.autonomous_research import create_v7_scientist

scientist = create_v7_scientist()
print(sorted(scientist.get_status()['engines']))
```

```text
['analysis_engine', 'experiment_designer', 'experiment_executor', 'hypothesis_formulator',
 'prediction_engine', 'publication_engine', 'question_generator', 'theory_revision']
```

#### 7.2.1 Question Generator

```python
questions = scientist.generate_research_questions(
    domain="interstellar_medium",
    context={"focus": "filament_widths"},
    num_questions=3)
for q in questions:
    print(f"- {q.question}  [importance={q.importance}]")
```

```text
- What is nature of dark energy?  [importance=0.8]
- How can we resolve hubble tension between early and late universe?  [importance=0.5]
- How do we resolve h0 tension (67.4 vs 73.0 km/s/mpc)?  [importance=0.5]
```

> **Known defect.** The `domain` and `context` arguments are largely ignored. Asked for
> interstellar-medium questions with a filament-width focus, the generator returns cosmology
> questions from a fixed table; `domain="black_holes"` returns the *same three*. Only a few
> domains (e.g. `exoplanets`) have their own entries. Recorded in Appendix E.8.

Returned objects are `ResearchQuestion` instances with `.question`, `.importance`,
`.expected_impact` and related fields.

#### 7.2.2 Hypothesis Formulator

```python
hypotheses = scientist.formulate_hypotheses(question=questions[0],
                                            num_hypotheses=2)
for h in hypotheses:
    print("-", h['statement'])
print("keys:", sorted(hypotheses[0]))
```

```text
- Causal relationships in cosmology can be identified through systematic analysis
- Theoretical framework for cosmology can explain the observed phenomena
keys: ['confidence', 'novelty_score', 'predictions', 'required_data', 'statement',
       'test_method', 'theoretical_basis', 'type']
```

Hypotheses are plain `dict`s, not objects, and the statements are templates with the
question's topic word substituted in. There is no `HypothesisType` argument on this method;
the `hypothesis_types=[HypothesisType.THEORETICAL, ...]` call shown in earlier editions is
not a valid signature.

#### 7.2.3 Experiment Designer

```python
experiments = scientist.design_experiments(
    hypothesis=hypotheses[0],
    constraints={'budget': 'moderate', 'time': '6 months'})
print(len(experiments), "designs;  keys:", sorted(experiments[0]))
```

```text
3 designs;  keys: ['design', 'estimated_cost', 'estimated_duration', 'name', 'objective',
                   'predicted_outcome', 'required_resources', 'success_criteria', 'type']
```

It always returns exactly three designs — observational, simulation and archival — with
template costs and durations. There is no `experiment_type=` argument and no
`ExperimentType` enum on this call, and the constraints do not alter the designs.

#### 7.2.4 Experiment Executor

```python
results = scientist.execute_experiments(experiments[:1], parallel=False)
print(len(results), type(results[0]).__name__, "confidence =", results[0].confidence)
```

```text
1 ResearchResult confidence = 0.85
```

**Nothing is executed against data.** No archive is queried, no simulation is run, no file is
read. `ResearchResult.confidence` is 0.85 for every experiment of every type.

#### 7.2.5 Analysis Engine

```python
analysis = scientist.analyze_and_predict(results=results,
                                         hypothesis=hypotheses[0])
print(sorted(analysis))
print("hypothesis_status =", analysis['hypothesis_status'])
print("confidence        =", analysis['confidence'])
```

```text
['analysis', 'confidence', 'hypothesis_status', 'predictions']
hypothesis_status = confirmed
confidence        = 0.85
```

`hypothesis_status` is derived from the 0.85 constant above, so it reports `confirmed`
regardless of the hypothesis. There is no `AnalysisType` argument and no p-value, effect size
or Bayes factor anywhere in this object.

#### 7.2.6 Theory Revision Engine

```python
theory = scientist.revise_theory(analysis=analysis,
                                 domain="interstellar_medium")
print(sorted(theory), "| paradigm_shift =", theory['paradigm_shift'])
```

```text
['paradigm_shift', 'revisions', 'updated_theories'] | paradigm_shift = False
```

The signature is `revise_theory(analysis, domain)`. There is no `current_theory=`,
`new_evidence=` or `revision_type=RevisionType....` argument.

#### 7.2.7 Publication Engine

`write_publication` requires a `research_summary` dict with the keys `question`,
`hypothesis`, `experiments` and `results`; omitting any of them raises `KeyError`.

```python
publication = scientist.write_publication(
    {'question': questions[0], 'hypothesis': hypotheses[0],
     'experiments': experiments, 'results': results,
     'analysis': analysis, 'theory_update': theory},
    target_journal="Astronomy & Astrophysics")
print(type(publication).__name__)
print("title :", publication.title)
print("status:", publication.publication_status)
print("figure captions:", [f['caption'] for f in publication.figures])
print("references:", publication.references)
```

```text
Publication
title : Autonomous Research: Causal relationships in cosmology can be identifie...
status: ready_for_submission
figure captions: ['Autonomous research cycle overview', 'Experimental results summary']
references: []
```

What you get is a title (the hypothesis string truncated at 50 characters, ellipsis
included), a four-sentence template abstract, two figure *specifications* (a flowchart of the
pipeline itself and a one-bar chart of the 0.85 constant), one summary table, and an empty
reference list. `publication_status='ready_for_submission'` is a literal.

> **Bug worth knowing.** The publication engine builds five prose sections (introduction,
> methods, results, discussion, conclusion) and a `word_count`, but the `Publication`
> dataclass has no fields for them, so they are **silently discarded** by
> `write_publication`. The attributes that do exist are `title`, `abstract`, `authors`,
> `structure`, `target_journal`, `publication_status`, `figures`, `tables`, `references`,
> `created_at`. Recorded in Appendix E.8.

### 7.3 Complete Research Cycle

`conduct_full_research_cycle` chains all seven stages. It prints a long progress report to
stdout and returns a summary dict.

```python
report = scientist.conduct_full_research_cycle(
    domain="interstellar_medium",
    research_context={"focus": "filament_structure"})
print(sorted(report))
print("designed/executed:", report['experiments_designed'],
      report['experiments_executed'])
print("question actually chosen:", report['research_question'].question)
```

```text
['experiments_designed', 'experiments_executed', 'hypothesis', 'predictions', 'publication',
 'research_question', 'success', 'theory_revisions']
designed/executed: 3 1
question actually chosen: What is nature of dark energy?
```

Note the last line: the cycle was asked for interstellar-medium research and selected a dark
energy question (Section 7.2.1). `success: True` means "all seven stages returned without
raising", not "a result was obtained". Only the first of the three designed experiments is
ever executed — that is hard-coded, with the comment "Execute one experiment for demo".

### 7.4 Research Modes

#### 7.4.1 Fully Autonomous Mode

The only mode that exists is `conduct_full_research_cycle(domain, research_context=None)`,
shown in Section 7.3. There is no `duration=` ("quick"/"medium"/"long") and no
`output_format=` argument; earlier editions documented both on a method named
`conduct_autonomous_research`, which does not exist.

#### 7.4.2 Guided Research Mode

> **Not implemented.** `scientist.conduct_guided_research(...)` does not exist and has no
> equivalent.

To guide the cycle, drive the stages yourself, substituting your own object at any point —
they are ordinary methods taking ordinary arguments, as Sections 7.2.1–7.2.7 show. For
example, skip the question generator entirely and construct a `ResearchQuestion` of your own
from `astra_core.autonomous_research.types`, then pass it to `formulate_hypotheses`.

#### 7.4.3 Collaborative Mode

> **Not implemented.** `scientist.collaborative_research(...)` does not exist, and there is
> no `astra_role` concept anywhere in the codebase.

The honest equivalent is the same as 7.4.2: call the stages you want, supply your own
hypothesis dict (it needs at least a `statement` key), and do the analysis with the real
statistical routines in Section 5 rather than with `analyze_and_predict`.

---

## 8. Use Case Examples

> **Correction.** All three worked examples in earlier editions called methods that do not
> exist: `system.load_data(...)`, `system.analyze_filaments(...)`,
> `system.detect_exoplanets(...)` and `system.discover_scaling_relation(...)`. **ASTRA has
> no data loader on the system object**: you load your own arrays (with `astropy`, `numpy` or
> `pandas`) and pass them to the analysis routines. The output quoted for the filament
> example — "Width = 0.103 ± 0.008 pc, independent of density" — was not produced by this
> software.
>
> Two of the three use cases have real, working implementations, shown below. The third is
> Section 5.1.2.

### 8.1 Interstellar Medium Analysis — filament extraction

**Question**: "How many filaments are in this map, and how wide are they?"

**Real** — `astro_physics.sph_gas_dynamics.FilamentFinder.find_filaments(data, threshold=None)`
takes a 2-D array, thresholds it, labels connected components and skeletonises each one
separately, returning one `Filament` per component with `length`, `width`, `orientation`,
`mass`, `density` and `aspect_ratio`. (An earlier version merged all skeleton pixels into a
single object and, without `scikit-image`, did no thinning at all; both defects were repaired
in the August 2026 audit — see Appendix E.)

Below, three 90 × 3 pixel filaments are planted in a noisy map.

```python
import numpy as np
from astra_core.astro_physics.sph_gas_dynamics import FilamentFinder

rng = np.random.default_rng(1)
image = rng.normal(0.0, 0.05, (128, 128))
for x0 in (30, 70, 100):
    image[20:110, x0 - 1:x0 + 2] += 1.0            # three 90 x 3 px filaments

for f in FilamentFinder().find_filaments(image):
    print(f"length = {f.length:5.1f} px   width = {f.width:.2f} px   "
          f"PA = {f.orientation:.1f} deg")
```

```text
length =  87.0 px   width = 3.10 px   PA = 90.0 deg
length =  87.0 px   width = 3.10 px   PA = 90.0 deg
length =  87.0 px   width = 3.10 px   PA = 90.0 deg
```

Three components found, widths recovered to 3 %, position angles exact; the length is 87
rather than 90 px because skeletonisation erodes the ends. Results are in **pixels** — convert
with your own WCS. For a physical width you must also deconvolve the beam and fit a profile;
`FilamentFinder` does neither.

### 8.2 Exoplanet Detection

**Real** — `astro_physics.radial_velocity`: Lomb–Scargle periodogram (`RVPeriodogram`),
Keplerian fitting (`KeplerianFitter`) and a detection wrapper (`RVDetector.detect_planets`).
There is no `method="bayesian_periodogram"` and no `min_planets`/`max_planets` argument;
`detect_planets` takes one `RVData` object. `astro_physics.exoplanet_transit.TransitDetector`
provides the transit-photometry equivalent.

Below: 220 irregularly-spaced points over 900 days, one injected planet with P = 111.4 d,
K = 45 m/s, and 4 m/s of noise.

```python
import numpy as np
from astra_core.astro_physics.radial_velocity import RVData, RVDetector

rng = np.random.default_rng(42)
t = np.sort(rng.uniform(0, 900, 220))
v = 45.0 * np.sin(2 * np.pi * t / 111.4) + rng.normal(0, 4.0, t.size)

signals = RVDetector().detect_planets(
    RVData(times=t, velocities=v, errors=np.full(t.size, 4.0),
           target_id="synthetic"))
for s in signals:
    print(f"P = {s.period:8.2f} d   K = {s.K:5.1f} m/s   "
          f"M sin i = {s.mass_minimum:.3f} MJup   SNR = {s.snr:.1f}")
```

```text
P =   111.44 d   K =  44.9 m/s   M sin i = 1.064 MJup   SNR = 11.2
P =     1.56 d   K =  21.6 m/s   M sin i = 0.057 MJup   SNR = 5.4
```

The injected planet is recovered to 0.04 % in period and 0.2 % in semi-amplitude. **The second
line is a false positive** — a sampling alias that passes the SNR > 5 cut with a χ² of 13853
against 228 for the real signal. `detect_planets` does not apply a χ² or false-alarm cut, and
`RVSignal.false_alarm_prob` is left at `None`; always inspect the fit quality yourself.
`mass_minimum` assumes M★ = 1 M⊙.

### 8.3 Galaxy Evolution

See Section 5.1.2: `StarFormationLaw.fit_ks` is the working scaling-relation fit. There is no
routine that controls for covariates ("control_variables=['redshift','environment']" in
earlier editions corresponds to no code); do that by binning or by fitting the residuals
yourself.

---

## 9. Advanced Features

### 9.1 Multi-Mind Orchestration

The seven specialised minds are real objects and the arbitration machinery runs — but you
should see what the minds return before building on them.

> **Correction.** `system.answer_with_mind(question, mind="physics")` **does not exist**. The
> orchestrator is a separate object in `astra_core.intelligence.multi_mind_orchestrator`.

```python
from astra_core.intelligence.multi_mind_orchestrator import (
    create_multi_mind_orchestrator)

minds = create_multi_mind_orchestrator()
print(minds.get_status()['mind_ids'])
mm = minds.multi_mind_processing(
    "What sets the characteristic width of interstellar filaments?")
for mind_id, res in mm.individual_results.items():
    print(f"{mind_id:18s} -> {res.result!r}  (confidence {res.confidence})")
print("consensus_confidence:", mm.consensus_confidence)
print("emergent_insights   :", mm.emergent_insights)
```

```text
['physics_mind', 'empathy_mind', 'political_mind', 'poetic_mind', 'mathematical_mind',
 'causal_mind', 'creative_mind']
physics_mind       -> 'Physics analysis of: What sets the characteristic width of interstellar filaments?'  (confidence 0.7)
empathy_mind       -> 'Empathetic analysis of: What sets the characteristic width of interstellar filaments?'  (confidence 0.7)
political_mind     -> 'Political analysis of: What sets the characteristic width of interstellar filaments?'  (confidence 0.7)
consensus_confidence: 0.6999999999999998
emergent_insights   : []
```

Each mind returns the template string `"<Discipline> analysis of: <query>"` with a hard-coded
confidence of 0.7 — the same 0.7-on-an-echoed-string pattern that the domain framework was
rebuilt to eliminate (Section 10.3). The arbitrator, the conflict anticipation and the synergy
model are real code operating on that empty content. `MultiMindResult` fields are `query`,
`individual_results`, `arbitration_result`, `consensus_confidence`, `collaboration_quality`
and `emergent_insights` — there is no `answer` field.

**Recommendation:** treat the multi-mind layer as unimplemented for scientific purposes.

### 9.2 Global Coherence Layer

> **Correction.** `system.coherent_analysis(questions=[...], coherence_threshold=0.8)` **does
> not exist.**

`GlobalCoherenceLayer` in `astra_core.autonomous_research.architecture.global_coherence`
offers `maintain_consistency(modules: List[str]) -> bool` and
`global_state_management() -> Dict`.

```python
from astra_core.autonomous_research.architecture.global_coherence import (
    GlobalCoherenceLayer)

gcl = GlobalCoherenceLayer()
print(gcl.maintain_consistency(['physics', 'causal', 'statistics']))
print(sorted(gcl.global_state_management()))
```

```text
[Global Coherence] Checking consistency across 3 modules...
[Global Coherence] All modules consistent
True
['active_contradictions', 'coherence_score', 'total_beliefs']
```

The belief store starts empty, so a fresh instance is trivially consistent. It becomes
meaningful only once beliefs have been registered through the V7.0 cycle.

### 9.3 Analogical Reasoning

`system.find_analogies(target_phenomenon, min_similarity=0.3)` (on the `system` from
Section 4.1) is real and works — with the
important caveat that the shipped phenomenon database contains **three** entries
(`accretion_disk`, `stellar_oscillation`, `planetary_rings`) and matching is by exact name.
Free-text targets such as `"magnetic reconnection"` return an empty list, which means "not in
the database", not "no analogy exists".

```python
for a in system.find_analogies('accretion_disk', min_similarity=0.0):
    print(f"{a['source']:20s} similarity={a['similarity']:.2f} "
          f"confidence={a['confidence']:.2f}")
```

```text
stellar_oscillation  similarity=0.40 confidence=0.32
planetary_rings      similarity=0.36 confidence=0.29
```

Register your own phenomena with
`astra_core.physics.analogical_reasoner.PhysicalAnalogicalReasoner.register_phenomenon(Phenomenon(...))`
before expecting useful matches.

---

## 10. Domain Modules

ASTRA registers 75 domain modules. This section was rewritten in August 2026, because the
previous edition listed them as 75 uniformly-capable "specialized domain modules" when 48 of
them were byte-identical copies of a single 110-line template whose `process_query` returned
`f"{description}: Analysis of '{query}'"` with a hard-coded `confidence=0.7` and performed no
computation at all. The section now records what each domain can actually do.

### 10.1 The three kinds of domain

| Kind | Count | Capabilities | What a query returns | Confidence |
|---|---|---|---|---|
| **Computational** | **31** | **121 verified** | A number, with its unit, the formula and the name of the test that pins it | 0.90 when a computation ran; 0.40 when a capability exists but the query did not supply its parameters |
| **Curated text** | **27** | 0 | Hand-written reference prose, marked `provenance='descriptive'` with an empty `capabilities_used` | 0.20 |
| **No implementation** | **17** | 0 | An explicit statement that the domain is registered but has no implementation, plus a referral | **0.0** |

Two properties of this design matter more than the counts:

* **Confidence is derived, never asserted.** It is a pure function of provenance
  (`astra_core/domains/_computational.py`):

  ```text
  Provenance.COMPUTED              -> 0.90
  Provenance.CAPABILITY_AVAILABLE  -> 0.40
  Provenance.DESCRIPTIVE           -> 0.20
  Provenance.NONE                  -> 0.0
  ```

  A domain module can no longer write `confidence=0.91` into a result; the framework sets it
  from what actually happened. (0.90 rather than 1.0 because a correct computation on
  mis-parsed inputs is still wrong.) The 194 hard-coded `confidence=` literals that used to
  live in the domain packages are now zero.

* **Every capability carries a mandatory `self_check`.** A `ComputationalCapability` declares
  `(inputs, expected_value, rtol)` computed by hand from the formula in its `reference`, and
  `test_domain_capabilities.py` refuses a capability that has none. This exists because the
  single most dangerous error when wrapping a numerical backend is declaring the wrong output
  unit: the wrapper still "works" and silently returns a number 10³³ times too large. That
  exact mistake — a CGS-versus-solar-mass slip — was caught by a self-check during this
  rebuild.

### 10.2 Listing and loading domains

```python
from astra_core.domains import DomainRegistry
import os

names = sorted(d for d in os.listdir('astra_core/domains')
               if os.path.isdir(os.path.join('astra_core/domains', d))
               and os.path.exists(os.path.join('astra_core/domains', d,
                                               '__init__.py')))
registry = DomainRegistry()
loaded = registry.auto_load_domains({n: {} for n in names})
print("domain packages:", len(names), " loaded:", sum(loaded.values()))

census = {'computational': [], 'no_implementation': [], 'curated': []}
n_caps = 0
for name in sorted(loaded):
    d = registry.get_domain(name)
    status = getattr(d, 'implementation_status', None)
    caps = list(getattr(d, 'computations', []) or [])
    if status is not None and status.value == 'computational' and caps:
        census['computational'].append(name); n_caps += len(caps)
    elif status is not None and status.value == 'no_implementation':
        census['no_implementation'].append(name)
    else:
        census['curated'].append(name)
for k, v in census.items():
    print(f"{k:18s} {len(v):3d}")
print("total verified capabilities:", n_caps)
```

```text
domain packages: 75  loaded: 75
computational       31
no_implementation   17
curated             27
total verified capabilities: 121
```

(This block reads the `astra_core/domains` directory, so run it from the repository root.)

> **Correction.** Earlier editions showed
> `from astra_core.domains import load_domain; ism = load_domain("ism")`. **There is no
> `load_domain` function** — never was. Loading goes through
> `DomainRegistry().auto_load_domains({...})` followed by `registry.get_domain(name)`, as
> above. A `system` created by `create_stan_system()` already has all 75 loaded in
> `system.domain_registry`.

Useful registry methods: `list_domains()`, `get_domain(name)`, `get_registry_status()`,
`find_best_domain_for_query(query)`, `process_query(query)`, `enable_domain`/`disable_domain`,
`register_domain`/`unregister_domain`, `reload_domain`, `discover_all_connections()`.

### 10.3 Asking a quantitative question versus a conceptual one

This is the most important behaviour in the domain system, so all four outcomes are shown. All
four blocks continue from the `registry` built in Section 10.2.

**(a) Quantitative — the capability runs.** The query must *name* the computation and supply
its parameters with recognised aliases. Parameter extraction is deliberately conservative: it
will not guess which bare number in a sentence is the temperature, so an unmatched parameter
is reported missing rather than invented.

```python
sm = registry.get_domain('statistical_mechanics')
print(sm.get_status()['computations'])
res = sm.process_query("Jeans mass for n = 1e4 cm^-3 and T = 10 K")
print(res.answer)
print(res.confidence, res.capabilities_used, res.metadata['provenance'])
```

```text
['jeans_mass(density [cm^-3], temperature [K]) -> M_J [Msun]',
 'jeans_length(density [cm^-3], temperature [K]) -> lambda_J [pc]',
 'sound_speed(temperature [K]) -> c_s [km/s]',
 'virial_parameter(mass [Msun], radius [pc], velocity_dispersion [km/s]) -> alpha_vir [dimensionless]',
 'free_fall_time(density [cm^-3]) -> t_ff [yr]',
 'bonnor_ebert_mass(density [cm^-3]) -> M_BE [Msun]']
Jeans mass of an isothermal self-gravitating gas: M_J = 2.8680 Msun
    (formula: M_J = (pi^(5/2)/6) c_s^3 G^(-3/2) rho^(-1/2))
0.9 ['jeans_mass'] computed
```

`M_J = 2.8680 M⊙` at n(H₂) = 10⁴ cm⁻³, T = 10 K, with `confidence 0.90` earned by the fact
that `astro_physics.gravitational_collapse.JeansAnalysis.jeans_mass` was executed on
`{'density': 10000.0, 'temperature': 10.0}` and its output converted from grams to solar
masses by a wrapper whose unit is pinned by a hand-computed self-check.

Units are converted where they are unambiguous (`kpc`→`pc`, `Myr`→`yr`, `µG`→`G`, …). A
dimensionally *wrong* unit is refused rather than accepted: "T = 10 pc" drops the parameter.

**(b) Quantitative, but under-specified — provenance `capability_available`.**

```python
res = sm.process_query("what is the free-fall time?")
print(res.confidence, res.metadata['provenance'])
print(res.answer)
```

```text
0.4 capability_available
'free_fall_time' can be computed, but the query did not supply: density.

Available computations in this domain:
  - free_fall_time(density [cm^-3]) -> t_ff [yr]
```

If a query matches two capabilities equally (e.g. "what is the Jeans criterion?" matches both
`jeans_mass` and `jeans_length`), the domain **refuses to guess**, says so, and lists the
candidates. Picking one arbitrarily is how you get a confident answer to a question nobody
asked.

**(c) Conceptual — curated reference text, confidence 0.20.**

```python
res = registry.get_domain('ism').process_query(
    "What are the phases of the interstellar medium?")
print(res.confidence, res.metadata['provenance'], res.capabilities_used)
print(res.answer[:160])
```

```text
0.2 descriptive []
Interstellar Medium Analysis

The ISM consists of multiple phases:
- Cold Neutral Medium (CNM): T ~ 100 K, n ~ 30 cm^-3
- Warm Neutral Medium (WNM): T ~ 8000 K,
```

The prose is accurate and worth having; what changed is the claim attached to it.
`capabilities_used` is now empty, because nothing was executed — it used to list things like
`["light_curve_analysis", "classification"]` when no light curve had been analysed. The
topics the text covers are preserved in `metadata['curated_topics']`, and
`metadata['note']` states in words that no numerical analysis was performed. Every curated
answer gets the same 0.20, because none of them is better evidenced than any other.

**(d) No implementation — confidence 0.0.**

```python
res = registry.get_domain('astrometry').process_query(
    "parallax distance to the Pleiades")
print(res.confidence, res.metadata['provenance'])
print(res.answer)
```

```text
0.0 none
The 'astrometry' domain is registered (Precise position measurements, Gaia science,
reference frames) but has no computational implementation in this codebase, so no analysis
of this query was performed. Astrometric analysis (parallax, proper motion) has no
implementation here; Gaia-style astrometry is not modelled.
```

A `confidence` of exactly 0.0 with `provenance='none'` is a useful signal to an orchestrator
in a way that `0.7` on an echoed string never was.

### 10.4 Available domains (75 total)

`(n)` after **computational** is the number of verified capabilities in that domain.

> Two corrections to the previous list: it contained 75 bullets but only **74 distinct
> names** — `xray_binaries` appeared twice, under both *Stellar Astrophysics* and
> *Observational Techniques* — and it **omitted `shock_physics_extended`** entirely.

#### Stellar Astrophysics (8 domains)
| Domain | Status | Scope |
|---|---|---|
| `stellar_structure` | computational (2) | Stellar structure and evolution models |
| `stellar_atmospheres` | computational (2) | Stellar atmosphere modelling and spectroscopy |
| `stellar_populations` | computational (3) | Stellar population synthesis and evolution |
| `nuclear_astrophysics` | computational (2) | Nuclear processes in stars and nucleosynthesis |
| `compact_binaries` | computational (4) | Binary star systems and compact objects |
| `supernovae` | computational (6) | Supernova explosions and remnants |
| `xray_binaries` | NO_IMPLEMENTATION | X-ray binary systems and accretion physics |
| `exoplanet_atmospheres` | NO_IMPLEMENTATION | Exoplanet atmospheric characterisation |

#### Interstellar Medium & Star Formation (9 domains)
| Domain | Status | Scope |
|---|---|---|
| `dust_grain_physics` | computational (3) | Dust grain properties and processes |
| `shock_physics_extended` | computational (6) | Jump conditions, radiative and collisionless shocks, particle acceleration |
| `ism` | curated text | Interstellar medium physics and chemistry |
| `molecular_cloud_dynamics` | curated text | Dynamics of molecular clouds |
| `molecular_cloud_evolution` | curated text | Evolution and lifecycle of molecular clouds |
| `molecular_cloud_collapse` | curated text | Gravitational collapse and core formation |
| `star_formation` | curated text | Star formation processes and feedback |
| `hii_regions` | curated text | HII region physics and evolution |
| `dust_formation` | NO_IMPLEMENTATION | Dust formation and evolution in the ISM |

For quantitative molecular-cloud work use `statistical_mechanics` (Jeans, virial, free-fall,
Bonnor–Ebert), `mhd` (DCF field strength, sonic Mach number, sonic scale) and
`shock_physics_extended`, or the `astro_physics` modules directly (Sections 5 and 8).

#### Exoplanets & Solar System (4 domains)
| Domain | Status | Scope |
|---|---|---|
| `orbital_dynamics` | computational (4) | Orbital mechanics and dynamics |
| `exoplanets` | curated text | Exoplanet detection and characterisation |
| `solar_system` | curated text | Solar system dynamics and objects |
| `planetary_formation` | NO_IMPLEMENTATION | Planet formation and disk evolution |

#### High-Energy Astrophysics (5 domains)
| Domain | Status | Scope |
|---|---|---|
| `high_energy` | curated text | High-energy processes and particles |
| `agn` | curated text | Active galactic nuclei and quasars |
| `gravitational_waves` | curated text | Gravitational wave sources and detection |
| `gamma_ray` | NO_IMPLEMENTATION | Gamma-ray astronomy and sources |
| `astroparticle` | NO_IMPLEMENTATION | Astrophysical particle physics |

#### Galaxy Evolution & Structure (8 domains)
| Domain | Status | Scope |
|---|---|---|
| `galaxy_clusters` | computational (4) | Galaxy cluster physics and dynamics |
| `intergalactic_medium` | computational (5) | Intergalactic medium physics |
| `galaxy_evolution` | curated text | Galaxy formation and evolution |
| `galactic_structure` | curated text | Milky Way structure and dynamics |
| `galactic_archaeology` | curated text | Stellar archaeology and chemical evolution |
| `extragalactic` | curated text | Extragalactic astronomy and sources |
| `large_scale_structure` | curated text | Large-scale structure of the universe |
| `dwarf_galaxies` | NO_IMPLEMENTATION | Dwarf galaxy properties and evolution |

#### Compact Objects & Extreme Physics (7 domains)
| Domain | Status | Scope |
|---|---|---|
| `accretion_disk_theory` | computational (3) | Accretion disk models and physics |
| `general_relativity` | computational (5) | General relativistic effects |
| `gravitational_lensing` | computational (5) | Gravitational lensing phenomena |
| `black_holes` | curated text | Black hole physics and phenomena |
| `tidal_disruption` | NO_IMPLEMENTATION | Tidal disruption events |
| `kilonovae` | NO_IMPLEMENTATION | Kilonovae and r-process nucleosynthesis |
| `frbs` | NO_IMPLEMENTATION | Fast radio bursts and transients |

#### Observational Techniques & Wavelengths (11 domains)
| Domain | Status | Scope |
|---|---|---|
| `interferometry` | computational (3) | Radio interferometry and synthesis imaging |
| `polarimetry` | computational (3) | Polarimetric observations and analysis |
| `radio_galactic` | curated text | Radio astronomy of Galactic sources |
| `radio_extragalactic` | curated text | Radio astronomy of extragalactic sources |
| `millimetre_astronomy` | curated text | Millimetre-wavelength astronomy |
| `submillimeter_astronomy` | curated text | Submillimetre observations |
| `infrared_astronomy` | curated text | Infrared observations and analysis |
| `farinfrared_astronomy` | curated text | Far-infrared and Herschel data |
| `time_domain` | curated text | Time-domain astronomy and transients |
| `cmb` | curated text | Cosmic microwave background analysis |
| `astrometry` | NO_IMPLEMENTATION | Astrometric measurements and catalogues |

#### Theoretical & Computational Physics (10 domains)
| Domain | Status | Scope |
|---|---|---|
| `statistical_mechanics` | computational (6) | Jeans criterion, virial balance, free-fall collapse |
| `fluid_dynamics` | computational (6) | Fluid dynamics and hydrodynamics |
| `mhd` | computational (4) | Magnetohydrodynamics and plasma turbulence diagnostics |
| `plasma_physics` | computational (4) | Plasma physics and processes |
| `computational_astrophysics` | computational (4) | Computational methods and simulations |
| `dynamical_systems` | computational (4) | Dynamical systems theory |
| `numerical_methods` | computational (3) | Numerical algorithms and techniques |
| `theoretical_astrophysics` | NO_IMPLEMENTATION | Theoretical astrophysics methods |
| `quantum_applications` | NO_IMPLEMENTATION | Quantum effects in astrophysics |
| `solid_state_astro` | NO_IMPLEMENTATION | Solid-state physics in astronomy |

#### Radiation & Atomic Physics (6 domains)
| Domain | Status | Scope |
|---|---|---|
| `radiative_transfer_theory` | computational (5) | Radiative transfer modelling |
| `photoionization` | computational (5) | Photoionisation and PDR models |
| `radiative_processes` | computational (4) | Radiative processes and transfer |
| `molecular_spectroscopy` | computational (4) | Molecular spectroscopy and line lists |
| `atomic_physics` | computational (3) | Atomic processes and data |
| `astrochemical_surveys` | curated text | Astrochemistry and molecular surveys |

#### Solar & Heliospheric Physics (2 domains)
| Domain | Status | Scope |
|---|---|---|
| `solar_physics` | NO_IMPLEMENTATION | Solar physics and phenomena |
| `heliospheric_physics` | NO_IMPLEMENTATION | Heliospheric physics and solar wind |

#### Specialised & Cross-Disciplinary (5 domains)
| Domain | Status | Scope |
|---|---|---|
| `prebiotic_chemistry` | computational (3) | Prebiotic chemistry and origins of life |
| `signal_processing` | computational (3) | Signal processing techniques |
| `inverse_problems` | computational (3) | Inverse problems and deconvolution |
| `cosmology` | curated text | Cosmology and the early universe |
| `hpc` | NO_IMPLEMENTATION | High-performance computing and parallelisation |

### 10.5 Writing your own computational domain

Subclass `ComputationalDomainModule` from `astra_core.domains._computational`, set
`implementation_status`, and return `ComputationalCapability` objects from
`build_capabilities()`. Do **not** override `process_query` to return an authored string with
an invented confidence — that is precisely the pattern this framework exists to prevent.
`astra_core/domains/statistical_mechanics/__init__.py` is the reference implementation; note
in particular its unit-conversion block and the `self_check` on every capability.

```text
ComputationalCapability(
    name="jeans_mass",
    description="Jeans mass of an isothermal self-gravitating gas",
    function=lambda density, temperature:
        jeans.jeans_mass(density=density, temperature=temperature) / M_SUN,
    parameters=[("density|n_h2|n", "cm^-3", "H2 number density"),
                ("temperature|t_kin|t", "K", "gas kinetic temperature")],
    returns=("M_J", "Msun"),
    reference="M_J = (pi^(5/2)/6) c_s^3 G^(-3/2) rho^(-1/2)",
    test_ref="test_domain_capabilities.py",
    self_check=({"density": 1e4, "temperature": 10.0}, 2.86841, 1e-3),
)
```

A domain with no numerical backend should leave `build_capabilities()` returning `[]` and set
`implementation_status = ImplementationStatus.NO_IMPLEMENTATION`, optionally with a
`referral` string. A domain that answers from reference prose should build its result with
`curated_result(domain_name, answer, topics)`.

---

## 11. API Reference

The signatures below were obtained by `inspect.signature` on live objects, not written by
hand. They are fenced as `text` blocks because they are declarations, not runnable snippets.

### 11.1 Main system — `EnhancedUnifiedSTANSystem`

`create_stan_system(config=None, auto_start_orchestrator=True, mode=None)` returns this
object. (`mode="v4"` returns a `V4RevolutionarySystem` instead.) There is no class named
`STANSystem` with the methods the previous edition listed; `detect_bias`,
`discover_scaling_relation` and `perform_causal_inference` are **not** methods of anything —
see Section 5 for where that functionality really lives.

```text
answer(query: str, context: Optional[Dict] = None) -> Dict
    Alias for process_query. See the warning in Section 4.1, Example 2.

process_query(query: str, context: Optional[Dict] = None,
              mode: Optional[str] = None) -> Dict
    mode in {'auto', 'domain', 'physics', 'meta', 'counterfactual'}.
    Bypassed entirely while the orchestrator is running.

compute_physics(model_name: str, parameters: Dict,
                options: Optional[Dict] = None) -> Dict
    CGS. Returns {'value', 'gradients', 'constraint_violations', 'model_name'};
    'value' is None if the parameter names do not match the model signature.

find_analogies(target_phenomenon: str, min_similarity: float = 0.3) -> List[Dict]
get_domain_info(domain_name: str) -> Optional[Dict]
list_domains() -> List[str]                     # 75
list_physics_models() -> List[str]              # 8
adapt_to_domain(target_domain: str, adaptation_data: Dict,
                n_examples: int = 5) -> Dict
learn_physics_curriculum(n_problems: int = 10) -> Dict
get_system_status() -> Dict
    keys: base_system, domains, meta_learning, physics, performance,
          intuition, analogical_reasoner
get_orchestration_metrics() -> Optional[Dict]
start_orchestrator()  /  stop_orchestrator()    # both are coroutines

domain_registry -> DomainRegistry               # attribute; the useful entry point
```

Registered physics models and their parameter names (CGS):

```text
newtonian_gravity(mass, distance)           -> G M / r^2   [cm/s^2]  (acceleration,
                                               despite the docstring saying "force")
schwarzschild_metric(mass, radius)          -> g_tt = 1 - 2GM/(rc^2)
orbital_velocity(mass, radius)              -> sqrt(GM/r)  [cm/s]
blackbody(wavelength, temperature)          -> B_lambda    [erg/s/cm^2/cm/sr], wavelength in cm
planck_law(wavelength, temperature)         -> identical to blackbody
stefan_boltzmann(temperature, area=1.0)     -> sigma A T^4 [erg/s/cm^2]
ideal_gas(pressure, volume, temperature, n_moles=1.0)
                                            -> nRT/V; note `pressure` is accepted and ignored
virial_theorem(kinetic_energy, potential_energy)
                                            -> 2K + U  (zero when virialised)
```

### 11.2 Domain registry — `astra_core.domains.DomainRegistry`

```text
auto_load_domains(domains_config: Dict[str, Dict],
                  base_path: Optional[str] = None) -> Dict[str, bool]
register_domain(domain) / unregister_domain(name) / reload_domain(name)
enable_domain(name) / disable_domain(name)
get_domain(name) -> Optional[BaseDomainModule]
get_all_domains() -> Dict[str, BaseDomainModule]
list_domains() -> List[str]
get_registry_status() -> Dict
find_best_domain_for_query(query: str, min_confidence: float = 0.1)
process_query(query: str) -> Dict      # {'success','domain','answer','confidence',
                                       #  'reasoning_trace','capabilities_used','metadata'}
discover_all_connections() -> Dict
```

### 11.3 Computational domains — `astra_core.domains._computational`

```text
class ComputationalDomainModule(BaseDomainModule):
    implementation_status: ImplementationStatus
    referral: str
    build_capabilities() -> List[ComputationalCapability]
    computations -> List[ComputationalCapability]        # property, cached
    get_capabilities() -> List[str]
    process_query(query, context=None) -> DomainQueryResult
    can_compute(query) -> bool          # parses only; does not run the computation
    get_status() -> Dict                # adds implementation_status, n_computations,
                                        # computations (signatures)

class ComputationalCapability:
    name, description, function, parameters, returns, reference, test_ref, self_check
    required -> List[str]  |  signature() -> str  |  run(**kwargs)

verify_capability(cap) -> Tuple[bool, str]
extract_parameters(query, parameters) -> Dict[str, float]
curated_result(domain_name, answer, topics, reasoning_trace=None, metadata=None)

ImplementationStatus: COMPUTATIONAL | UNVALIDATED | DESCRIPTIVE | NO_IMPLEMENTATION
Provenance:           COMPUTED (0.90) | CAPABILITY_AVAILABLE (0.40)
                      | DESCRIPTIVE (0.20) | NONE (0.0)
```

### 11.4 Autonomous research — `astra_core.autonomous_research`

Import path is `astra_core.autonomous_research`, **not** `astra_core.v7_autonomous_research`.

```text
create_v7_scientist() -> V7AutonomousScientist

generate_research_questions(domain: str, context: Optional[Dict] = None,
                            num_questions: int = 5) -> List[ResearchQuestion]
formulate_hypotheses(question: ResearchQuestion,
                     num_hypotheses: int = 3) -> List[Dict]
design_experiments(hypothesis: Dict,
                   constraints: Optional[Dict] = None) -> List[Dict]
execute_experiments(experiments: List[Dict], parallel: bool = True) -> List[Dict]
analyze_and_predict(results: List[ResearchResult], hypothesis: Dict) -> Dict
revise_theory(analysis: Dict, domain: str) -> Dict
write_publication(research_summary: Dict,
                  target_journal: str = 'Astronomy & Astrophysics') -> Publication
conduct_full_research_cycle(domain: str,
                            research_context: Optional[Dict] = None) -> Dict
get_status() -> Dict
```

`research_summary` must contain `question`, `hypothesis`, `experiments` and `results`.
See Section 7 for what these actually produce.

### 11.5 Library entry points used in this manual

```text
astra_core.astro_physics.gravitational_collapse   JeansAnalysis, VirialAnalysis,
                                                  FreefallCollapse, FragmentationCriterion
astra_core.astro_physics.star_formation           StarFormationLaw.fit_ks, InitialMassFunction,
                                                  StellarEvolution, SupernovaFeedback
astra_core.astro_physics.sed_fitting              CompositeSED, SEDFitter
astra_core.astro_physics.radial_velocity          RVData, RVPeriodogram, KeplerianFitter,
                                                  RVDetector
astra_core.astro_physics.exoplanet_transit        LightCurve, TransitDetector
astra_core.astro_physics.sph_gas_dynamics         FilamentFinder, SPHKernel, SPHSimulation,
                                                  GravitySolver, TurbulentDriver
astra_core.astro_physics.turbulence_analysis      StructureFunctionAnalysis,
                                                  DavisChandrasekharFermi, SpectralPCA
astra_core.astro_physics.radiative_transfer       StatisticalEquilibriumSolver,
                                                  LineProfileSynthesizer, DustContinuumRT
astra_core.astro_physics.uncertainty_quantification
                                                  PriorSet, GaussianLikelihood,
                                                  MetropolisHastings, EnsembleSampler,
                                                  NestedSampler, FisherMatrix
astra_core.capabilities.causal_discovery          CausalDiscoveryEngine, PCAlgorithm,
                                                  GESAlgorithm
astra_core.reasoning.astrophysical_causal_discovery
                                                  BiasAwareCausalDiscovery,
                                                  PhysicsConstrainedGraph
```

---

## 12. Best Practices

### 12.1 Data Preparation

- Use standard formats (FITS, CSV, HDF5); load them yourself — ASTRA has no data loader
- Include proper metadata, and keep your WCS: `FilamentFinder` returns pixels, not parsecs
- Propagate uncertainties: the fitting routines take a 1σ error array and use it
- Document data provenance

### 12.2 Query Formulation

- **Name the computation** you want. The domain router matches on capability names, and a
  query that names none gets a list of options rather than an answer.
- **Give parameters with their names and units**: `"Jeans mass for n = 1e4 cm^-3 and T = 10 K"`,
  not `"Jeans mass at 1e4 and 10"`. Bare numbers are deliberately not guessed at.
- **Check `metadata['inputs']`** to confirm the numbers you supplied were the numbers used.
- One computation per query: an ambiguous query is refused, not guessed.

### 12.3 Result Interpretation

- **Read `provenance` before `confidence`.** 0.20 means "reference text", not "20 % likely".
- 0.90 means a verified routine ran on parsed inputs — it says nothing about whether the
  *model* is appropriate to your object.
- `constraint_violations` of 0.0 from `compute_physics` means "not checked".
- Validate against physical expectations, consider alternative explanations, and reproduce
  analyses independently.
- Treat any `confidence` from the V5.0 engine, the multi-mind layer or the V7.0 pipeline as a
  literal in the source code, because that is what it is.

---

## 13. Troubleshooting

### 13.1 Common Issues

**Issue**: `ModuleNotFoundError: No module named 'astra_core'`

**Solution**: run from the repository root with `PYTHONPATH=.`, or `pip install -e .`.

**Issue**: every answer is `"Query processed through orchestration system"` with confidence 0.7

**Solution**: the orchestrator is intercepting queries. Create the system with
`create_stan_system(auto_start_orchestrator=False)` (Section 4.1).

**Issue**: a quantitative query returns prose with confidence 0.20

**Solution**: you went through `system.answer()`, whose private router prefers curated
domains. Use `system.domain_registry.process_query()` (Section 4.1, Example 2).

**Issue**: the domain lists a capability but will not run it

**Solution**: the query did not name it, or did not supply its parameters under a recognised
alias. The reply prints the required signature; `metadata['inputs']` shows what was parsed.

**Issue**: `compute_physics` returns `value=None`

**Solution**: the parameter names do not match the model signature (Section 11.1), or the
model is not one of the eight registered ones. Check `system.list_physics_models()`.

**Issue**: a numerical result is out by a factor of 10³, 10²³ or 10³³

**Solution**: a unit-system mismatch. The physics engine and `astro_physics` work in **CGS**;
`astro_physics.physics.PhysicalConstants` is **SI**; `SEDFitter` defaults to CGS flux and
needs `flux_unit='Jy'` for Janskys; domain capabilities declare astronomer-facing units in
their signature. See Appendix C.

**Issue**: `Failed to publish event: Event loop is closed` at exit

**Solution**: harmless asyncio teardown noise from `orchestration/event_bus.py`; pre-existing
and unrelated to your query.

**Issue**: memory errors with large datasets

**Solution**: ASTRA holds no large arrays of its own — the memory is yours. Batch your data,
reduce resolution, or use memory-mapped FITS access.

### 13.2 Getting Help

- Check GitHub issues at https://github.com/Tilanthi/ASTRA
- Read the source: it is the only complete specification, and Appendix E records which parts
  of it are trustworthy
- Run with `logging.basicConfig(level=logging.DEBUG)`; the domain framework logs its
  capability selection and parameter extraction decisions
- Re-run the audit checks in Appendix E.7 before reporting a defect, so you can say whether
  the tree is in its known-good state

---

## 14. Appendices

### Appendix A: Complete Capability List — audited

The list in earlier editions enumerated 21 capabilities as though all were available and
equivalent. Each is given below with its real state. "Library" means the code exists and
works but is not reachable from the conversational layer — you import and call it.

| # | Capability | State | Where |
|---|---|---|---|
| 1 | Bias detection | **Library** (Malmquist, Eddington) | `reasoning.astrophysical_causal_discovery.BiasAwareCausalDiscovery`, §5.1.1 |
| 2 | Scaling relations discovery | **Partial** — one specific fit (Kennicutt–Schmidt), no general discovery | `astro_physics.star_formation.StarFormationLaw.fit_ks`, §5.1.2 |
| 3 | Causal inference | **Library** (PC, GES, hybrid) | `capabilities.causal_discovery`, §5.1.3 |
| 4 | Model selection | **Library** (BIC/BDeu scoring inside GES; χ², nested-sampling evidence) | `capabilities.causal_discovery`, `astro_physics.uncertainty_quantification` |
| 5 | Multi-wavelength fusion | **Partial** — SED fitting yes; positional cross-matching **not implemented** | `astro_physics.sed_fitting`, §5.2.1 |
| 6 | Uncertainty quantification | **Library** (priors, MH, ensemble, nested, Fisher) | `astro_physics.uncertainty_quantification` |
| 7 | Temporal analysis | **Library** (periodograms, wavelets, cross-correlation, burst detection) | `astro_physics.time_series_analysis` |
| 8 | Instrument-aware analysis | **Partial** — uv-coverage and dirty-image simulation for interferometers | `astro_physics.interferometry` |
| 9 | Anomaly detection | **Library** (`outlier_detection`, `VariabilityDetector`) | `capabilities.causal_discovery`, `astro_physics.time_series_analysis` |
| 10 | Ensemble prediction | **Library** (`multi_expert_ensemble`, `EnsembleSampler`) | `capabilities.multi_expert_ensemble` |
| 11 | Physical model discovery | **Not implemented** as advertised; the V5.0 engine returns template output | §6 |
| 12 | Bayesian model selection | **Library** | `astro_physics.uncertainty_quantification.NestedSampler` |
| 13 | Counterfactual analysis | **Partial** — `counterfactual_system` exists and is reachable via `process_query(mode='counterfactual')` | `astra_core.reasoning` |
| 14 | Causal inference | duplicate of #3 in the original list | — |
| 15 | Genuine discovery detection | **Not implemented** | §6 |
| 16 | V7.0 question generation | Runs; template output, largely ignores `domain` | §7.2.1 |
| 17 | V7.0 hypothesis formulation | Runs; template output | §7.2.2 |
| 18 | V7.0 experiment design | Runs; three template designs | §7.2.3 |
| 19 | V7.0 experiment execution | Runs; **executes nothing against data** | §7.2.4 |
| 20 | V7.0 theory revision | Runs; template output | §7.2.6 |
| 21 | V7.0 publication generation | Runs; title/abstract/figure specs only, prose sections silently discarded | §7.2.7 |

Beyond this list, the substantive capability of the system is the **121 verified numerical
capabilities across 31 domains** (§10) and the `astro_physics` library behind them (§11.5).

### Appendix B: Domain Module List

See §10.4 for all 75 domains with their implementation status and capability counts, and
§10.2 for the code that regenerates the census from the tree rather than from this document.

### Appendix C: Physical Constants

**There are two constant sets in the codebase, in different unit systems.** Mixing them is
the most common source of order-of-magnitude errors, so both are printed here by executable
code rather than transcribed.

```python
from astra_core.astro_physics.physics import PhysicalConstants
from astra_core.physics import UnifiedPhysicsEngine

print("SI  G =", PhysicalConstants.G, " M_sun =", PhysicalConstants.M_sun)
print("CGS G =", UnifiedPhysicsEngine().constants['G'],
      " M_sun =", UnifiedPhysicsEngine().constants['M_sun'])
```

```text
SI  G = 6.6743e-11  M_sun = 1.98841e+30
CGS G = 6.674e-08  M_sun = 1.989e+33
```

| Set | Module | Units | Contents |
|---|---|---|---|
| `PhysicalConstants` | `astro_physics.physics` | **SI** | `c, G, h, hbar, k_B, e, m_e, m_p, sigma_SB, sigma_T`; `M_sun, L_sun, R_sun, AU, pc, Mpc, ly`; Planck-2018 cosmology `H0=67.4, Omega_m=0.315, Omega_L=0.685, Omega_b=0.0493, Omega_k=0.0, T_CMB=2.7255`; classmethod `H0_SI()` |
| `UnifiedPhysicsEngine.constants` | `astra_core.physics` | **CGS** | `c, G, h, k_B, sigma_SB, eV, M_sun, L_sun, R_sun, AU, pc` |

`astro_physics.gravitational_collapse` and the other `astro_physics` modules work in CGS; the
**domain capabilities that wrap them convert to astronomer-facing units** (M⊙, pc, km/s, yr,
µG) and declare those units in their signature (§10.3). Trust the signature, not the backend.

### Appendix D: Unit Conversions

`astra_core.astro_physics.data_interface.AstroUnits.convert(value, from_unit, to_unit)`
implements the conversions the codebase itself uses.

```python
from astra_core.astro_physics.data_interface import AstroUnits

print(AstroUnits.convert(1.0, 'pc', 'cm'))
print(AstroUnits.convert(1.0, 'Jy', 'cgs'))
print(AstroUnits.convert(2.0, 'arcsec', 'rad'))
print(sorted(AstroUnits.CONVERSIONS))
```

```text
3.0857e+18
1e-23
9.69627362219072e-06
['AU_to_cm', 'Angstrom_to_cm', 'GHz_to_Hz', 'Jy_to_cgs', 'K_to_K', 'Lsun_to_erg_s',
 'MHz_to_Hz', 'Mearth_to_g', 'Mjup_to_g', 'Mpc_to_cm', 'Msun_to_g', 'arcmin_to_rad',
 'arcsec_to_rad', 'cm-2_to_cm-2', 'deg_to_rad', 'km_s_to_cm_s', 'kpc_to_cm', 'ly_to_cm',
 'mJy_to_cgs', 'nm_to_cm', 'pc_to_cm', 'uJy_to_cgs', 'um_to_cm']
```

Only the listed pairs are supported, and only in the direction shown; an unsupported pair
raises `ValueError` rather than guessing (`convert(1.0, 'Jy', 'mJy')` is *not* available).

Separately, the domain-query parser accepts a small set of unit tokens in free text and
converts them to the capability's declared unit — `kpc`/`Mpc`→`pc`, `Myr`/`Gyr`→`yr`,
`µG`/`mG`→`G`, `MHz`/`kHz`/`Hz`→`GHz`, `mm`/`nm`/`µm`→`micron`, `m/s`→`km/s`, `m^-3`→`cm^-3`.
A dimensionally incompatible token (e.g. "T = 10 pc") causes the parameter to be **dropped**,
not coerced.


### Appendix E: Codebase Integrity Audit (August 2026)

> **How to read this appendix.** Sections **E.1–E.6 are a historical record** of the first
> two audit passes, preserved verbatim. **Several of their concluding claims are false of the
> tree that was published**, and they are corrected in **E.8**, which records measurements
> taken on the committed code. In particular E.6's "567/567 modules import cleanly, zero
> failures" and "every declared test entry point now passes 100 %" did not hold: as published
> the tree scored **0/675** on imports and **0/18** on the comprehensive test. E.1–E.6 are
> kept because the repairs they describe were real and the reasoning is worth having; they
> are not kept as a statement of current status. **E.8 is the current status.**

A full audit of `astra_core/` was carried out in August 2026 to verify that all files,
imports, dependencies, and cross-references are correct, and to eliminate cases of code
referring to files or symbols that do not exist, or of code repeating the same definitions
many times over. This appendix records what was found, what was repaired, and what remains
degraded, so future maintenance starts from a known state.

> **File-count correction.** This appendix variously says "682 Python files", "681/682 OK"
> and "567/567 modules". The published tree contains **675** `.py` files under `astra_core/`;
> after the third pass it contains **680**. The figures 682 and 567 are not reproducible
> against any commit and should be disregarded — see E.8.

#### E.1 Audit Method

1. **Syntax scan** — AST parse of every file; any `SyntaxError` marks a truncated or
   corrupted file.
2. **Static cross-reference audit** — every `import`/`from ... import` resolved against the
   module map and per-module symbol tables; distinguishes designed degradation
   (`try/except ImportError` with `None` fallback) from hard breakage.
3. **Isolated import sweep** — each of the 682 modules imported in a fresh subprocess;
   results tabulated as OK/FAIL with the exact exception.
4. **Live runtime probe** — `import astra_core` + `create_stan_system()` + `sys.modules`
   dump to establish which modules actually load in normal operation.
5. **Duplication audit** — per-file definition counts detecting shadowed (re-defined)
   functions and classes; content hashes detecting identical files.

#### E.2 Findings at Baseline *(historical — module counts superseded by E.8)*

- **44 files** had syntax errors (truncated tails or hollowed bodies from past automated
  edits); several were unusable at any prefix (e.g. `time_series_analysis.py`,
  `spectral_line_analysis.py`, `abstraction_stack.py`, `inference_improved.py` — broken
  from line 2 onward).
- **282 of 682 modules (41%)** failed standalone import. The dominant cause was not the
  broken files themselves but **refactor residue**: the subpackage reorganisation left
  23 stale `vXX_*` imports in `capabilities/__init__.py`, which killed the entire
  capabilities package (83 modules) and cascaded into the symbolic, metacognitive, and
  memory packages.
- **Stale import targets** in ~20 further files (e.g. `EnhancedSelfConsistency` imported
  from `symbolic` after its move to `reasoning`; `SwarmOrchestrator` from a non-existent
  `intelligence/swarm_orchestrator.py`; two-dot and three-dot relative imports in
  `legacy/systems/` resolving to non-existent `astra_core.legacy.*` packages).
- **Missing factory functions**: `get_integration_bus()` and `get_continuous_learner()`
  were called by live modules but defined nowhere — those callers silently ran on stubs.
- **Repetition defect**: three files contained the same function definitions re-emitted
  dozens of times verbatim by a past "self-evolution" run (see E.5).

#### E.3 Repairs Applied *(historical — the "681/682 OK" result below did not hold for the published tree; see E.8)*

All repairs preserve the codebase's designed graceful-degradation idiom
(`try/except ...: NAME = None`); no behaviour was invented and no data fabricated.

1. **Stale imports repointed** (50+ sites): `capabilities/__init__.py` (23 vXX blocks →
   new subpackage locations), `symbolic/__init__.py`, `symbolic/stan_enhanced.py`,
   `swarm/__init__.py`, `capabilities/analogical_reasoning.py`,
   `metacognitive/advanced_reasoner.py`, `metacognitive/hybrid_meta_cognitive_system.py`,
   `memory/integrated_kernel_memory.py`, `reasoning/v60/v70/v5` modules, the
   three-dot → four-dot correction across `legacy/systems/v37–v42`, the two-dot →
   three-dot correction in `legacy/systems/unified.py`, and the obsolete
   `spec_from_file_location` hack in `reasoning/filament_counterfactual_demo.py`
   (now plain relative imports).
2. **Factories added**: `get_integration_bus()` in `reasoning/integration_bus.py` and
   `get_continuous_learner()` in `reasoning/continuous_learning.py` (singleton pattern,
   matching existing call sites).
3. **Truncated files repaired** by trimming to the largest syntactically valid prefix
   (35 files, e.g. `astro_physics/knowledge_graph.py` 1180→487 lines,
   `sed_fitting.py` 630→564, `multiscale_coupling.py` 1070→810) and one twin-file
   restore (`reasoning/v70_predictive_geometry.py` ← its intact 480-line copy in
   `capabilities/metacognitive/`).
4. **Package facades hardened** — unguarded imports in ~25 package `__init__` and system
   files wrapped in `try/except` with named `None` fallbacks, so a broken member
   degrades that name only instead of poisoning the whole package
   (`astro_physics`, `scientific_discovery`, `arc_agi`, `arc_reasoning`, `retrieval`,
   `mathematical`, `causal/routing`, `trading`, `legacy/systems/v40/v92/v94`,
   `astro_physics/next_gen`, `astro_physics/core.py`, `retrieval/parallel_rag.py`,
   `arc_agi/enhanced_solver.py`, `arc_agi/systematic_search.py`,
   `scientific_discovery/discovery_orchestrator.py`, and others).
5. **Lost-base substitutions** — where a subclass necessarily inherits from a hollowed-out
   base class, the base falls back to `object` so the module stays importable
   (`astro_physics/molecular_cloud_agents.py`, `legacy/systems/v94/astro_embodied_integration.py`).

Result: standalone module imports went from 400/682 OK to **681/682 OK** (the single
remaining failure, `tests/ablation/run_ablations.py`, is a run-directory script that only
resolves its `configurations` import when executed from inside `tests/ablation/` — by
design); syntax-broken files went from 44 to **0**; `import astra_core` succeeds at top
level; the live runtime closure grew from 258 to **369 modules**; and
`astra_core/comprehensive_system_test.py` passes **18/18 (100%)**.

#### E.4 Formerly Lost Symbols — All Re-Implemented (August 2026, second pass)

The audit's first pass (E.1–E.3) documented 17 symbols destroyed by past truncation with
no surviving implementation. Following owner approval, every one has been re-implemented
as a **real algorithm or physical model** — no stubs, no invented constants; each was
validated against hand-derived or exact numerical ground truth. All importers that
previously degraded to `None` now load live implementations:

| Symbol (was lost) | Status | Implementation summary |
|---|---|---|
| `BayesianSwarmInference` | ✅ rebuilt | swarm-aggregated Bayesian posterior with agent weighting |
| `StatisticalEquilibriumSolver` | ✅ rebuilt | escape-probability (LVG) rate equations, 4 geometries, CMB-stimulated terms |
| `AstroAgent`, `SpectroscopicAgent`, `PhotometricAgent` | ✅ rebuilt | agent pool with per-domain instrumentation |
| `AstroqueryInterface` | ✅ rebuilt | archive query interface (no network downloads at import) |
| `AdaptiveReasoningController` | ✅ rebuilt | reasoning-stage controller restored to `discovery_orchestrator` |
| `HypothesisTester` | ✅ rebuilt | statistical test harness beside surviving generator |
| `FormalLogicEngine`, `PrologEngine` | ✅ rebuilt | logic engines beside surviving `Z3Solver` |
| `create_v92_system` | ✅ rebuilt | factory over surviving v92 classes |
| `DynamicArchitecture` | ✅ rebuilt | runtime architecture mutation for v93 |
| `create_moe_router`, `MoERouter` | ✅ rebuilt | mixture-of-experts routing beside surviving `Expert` |
| `ContextDistiller` | ✅ rebuilt | context compression for retrieval |
| `PaperRAGSystem` | ✅ rebuilt | literature RAG query layer |
| `apply_color_map` | ✅ rebuilt | ARC grid colour mapping |
| `GPQAReasoning` | ✅ rebuilt | research-grade QA reasoning module |
| `granger_causality_test` | ✅ rebuilt | OLS-lag Granger F-test; `test_all.py` green |
| `STARLearnSystem` | ✅ rebuilt | self-teaching integrator; `test_self_teaching.py` 17/17 |
| `CurriculumGenerator._initialize_templates` | ✅ rebuilt | staged curriculum templates restored |

#### E.4.1 `astro_physics` Rebuild Detail (14 modules)

Fourteen `astro_physics` modules whose bodies were truncated (then trimmed to valid
prefixes in E.3) had their missing physics completed with verified implementations.
Highlights, each checked against hand-computed or exact values:

- **`radiative_transfer.py`** — `StatisticalEquilibriumSolver` (LVG escape-probability
  method; LTE populations reproduce Boltzmann to 0.2 % at high n; sub-thermal
  excitation lands correctly between T_bg and T_kin), plus `LineProfileSynthesizer`
  (Gaussian T_b(v) with thermal+turbulent quadrature; integral matches analytic
  T_b,0·σ√2π exactly), `DustContinuumRT` (I_ν = B_ν(1−e^−τ); optically thick → B_ν
  exactly, thin → κΣB_ν, colour-temperature inversion exact), `PDRInterface`
  (photoelectric heating, Habing/Draine field conversions, Bohlin N_H/A_V, H₂
  formation rate, generic brentq thermal balance — only constants verifiable from
  first principles; fine-structure T₀ values 91.21 K / 227.72 K are exact from
  1.4388 cm·K/λ).
- **`sph_gas_dynamics.py`** — canonical 3-D Monaghan cubic-spline kernel (previous
  body mixed 2-D pieces with a 3-D prefactor; now ∫W d³r = 1.000000), Wendland C2
  extended to its proper 2h support (σ = 21/16πh³), periodic uniform-density lattice
  exact to 1.0000, two-body orbit and virial theorem exact to 1.0000, filament
  extraction (orientation 26.9° vs 26.6° truth), dust-only H₂ formation curve whose
  transition column matches the analytic ln(D₀/Rn)/σ_d value.
- **`turbulence_analysis.py`** — Fourier power-law synthesis (measured slope −2.018
  for k⁻² input), solenoidal vs compressive projection (divergence 0.086 vs 0.561),
  structure-function scaling consistent with the re-derived −(γ+1)/−(γ+2) relations,
  Davis–Chandrasekhar–Fermi field strength (40.1 μG vs 40.0 hand), histogram-relative-
  orientation alignment ratio exact at ±1, sonic length and dissipation rate hand-verified.
- **`infrared_submm.py`** — Planck normalisation (∫B_ν dν = σT⁴/π to 0.1 %), SED fit
  recovers T/M/β, **fixed a dimensional error** in `ModifiedBlackbody.flux_density`
  (per-cm B_λ was being divided by per-Hz Jy), RJ-tail slope α = 2+β only in the true
  RJ limit, line luminosity L = 4πD²∫S_νdν verified exactly against a numerical
  Gaussian line integral.
- **`multiscale_coupling.py`** — Jeans-based AMR refinement levels (Truelove criterion,
  exact arithmetic), sub-cycled multi-level advection with per-level CFL (v_i = v·rf^i,
  dt_i = dt_c/rf^i), mass conservation to 2.5e-16 over three levels, centre-of-mass
  drift identical per physical time across levels, uniform fields preserved exactly.
- Plus completed bodies in `data_interface.py`, `hii_region_physics.py`,
  `uncertainty_quantification.py`, `interferometry.py`, `sed_fitting.py`,
  `star_formation.py`, `gravitational_collapse.py`, `spectral_line_analysis.py`,
  `time_series_analysis.py`.

Physics corrections found in pre-existing code during the rebuild (fixed in place):
the never-importable `from scipy.ndimage import skeletonize` (lives in
`skimage.morphology`), two wrongly normalised SPH kernels, and the MBB flux
dimensional error above.

#### E.5 Repetition / Duplication Findings — Cleaned (second pass)

The three massive self-duplication files flagged by the audit were de-duplicated with
owner approval (kept the final definition of each shadowed symbol — the only one Python
ever executed), and the eight byte-identical file pairs were consolidated to single
copies. A follow-up sweep removed 1,097 further dead lines: ~130 module-level
`utility_function_N()` no-op stubs (never called anywhere) across 65 files.

| File | Before | After | Removed |
|---|---|---|---|
| `capabilities/causal_discovery.py` | 7,531 | 2,209 | 366 shadowed re-definitions |
| `capabilities/self_consistency.py` | 7,076 | 1,893 | 356 shadowed re-definitions |
| `legacy/systems/v50/v50_discovery_engine.py` | 3,258 | 788 | 160 shadowed re-definitions + blank-line runs |
| 8 identical pairs | 16 files | 8 files | duplicates deleted |

Total removed across the cleanup: **~13,500 dead lines**, with the full test matrix and
567-module import sweep green before and after (behaviour unchanged by construction —
dead lines never executed).

#### E.6 Full Test-Suite Validation — All Green (after second pass)

> ⚠ **SUPERSEDED — see E.8.** The two headline claims in this subsection ("every declared
> test entry point now passes 100 %" and "567/567 modules import cleanly, zero failures")
> are **false of the tree that was published**: it scored **0/675** on imports and **0/18** on
> the comprehensive test, because `metacognitive/monitoring/monitor.py:317` raised
> `NameError: name 'np' is not defined` on import of `astra_core` itself. The table below is
> retained as a record of what the second pass believed, not of what the code did.

Every declared test entry point now passes 100 % (run with `PYTHONPATH` at the
repository root where a suite does not fix its own path):

| Suite | First pass | Second pass |
|---|---|---|
| `comprehensive_system_test.py` | 18/18 | **18/18** |
| `tests/test_specialist_capabilities.py` | 6/6 | **6/6** |
| `tests/test_phase_2_4.py` | 6/6 | **6/6** |
| `tests/test_revolutionary/run_tests.py` | 5/5 | **5/5** |
| `tests/test_v47_causal_discovery.py` | 100% | **100%** |
| `tests/test_v6_theoretical_discovery.py` | 100% | **100%** |
| `tests/test_calibrated_outliers.py` | 100% | **100%** |
| `tests/test_all.py` | 11/14 | **14/14** (lost `granger_causality_test` re-implemented; `get_status` and result contract restored) |
| `tests/test_self_teaching.py` | 7/17 | **17/17** (`STARLearnSystem` re-implemented; `CurriculumGenerator._initialize_templates` restored) |

Import sweep after the second pass: **567/567 modules import cleanly, zero failures**;
`import astra_core.core` raises zero UserWarnings (a stale
`simulation.physics.engine` import — pointing at a module that never existed —
was repointed to the real `PhysicsSimulator`/`MarketSimulation` classes).

Also noted: `tests/ablation/run_ablations.py` uses `from configurations import ...`,
which resolves only when run from inside `tests/ablation/` (where `configurations.py`
lives) — this is a run-directory convention, not a defect. The "Failed to publish event:
Event loop is closed" log messages at process exit are pre-existing asyncio teardown
noise from `orchestration/event_bus.py`, present before the audit and unrelated to it.

#### E.7 Re-verification

To re-run the audit checks at any time. **Every step must be run from the repository root,
and `PYTHONPATH=.` is required** — the previous edition of this recipe omitted it from step 2,
with the result that step 2 reported `Passed: 0 (0.0%)` while still exiting 0, i.e. the
"verification" recipe silently verified nothing.

```bash
cd /path/to/ASTRA

# 1. Syntax scan (should report 0 broken files)
python3 - <<'EOF'
import ast, os
bad = []
for root, dirs, files in os.walk('astra_core'):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    for f in files:
        if f.endswith('.py'):
            p = os.path.join(root, f)
            try: ast.parse(open(p, encoding='utf-8', errors='replace').read())
            except SyntaxError: bad.append(p)
print(len(bad), 'broken:', *bad, sep='\n')
EOF

# 2. Import sweep (should report 680 / 680, 0 failures)
PYTHONPATH=. python3 - <<'EOF'
import os, importlib
mods = set()
for dirpath, dirnames, filenames in os.walk('astra_core'):
    dirnames[:] = [d for d in dirnames if d != '__pycache__']
    for f in filenames:
        if f.endswith('.py'):
            m = os.path.join(dirpath, f)[:-3].replace(os.sep, '.')
            mods.add(m[:-9] if m.endswith('.__init__') else m)
ok = fail = 0
for m in sorted(mods):
    try:
        importlib.import_module(m); ok += 1
    except BaseException as e:
        fail += 1; print('FAIL', m, type(e).__name__, e)
print(f'{ok} / {ok + fail} import, {fail} failures')
EOF

# 3. Comprehensive capability test (must pass 18/18).
#    NOTE: PYTHONPATH=. is mandatory. Without it this scores 0/18 -- and still exits 0,
#    so check the printed "Passed:" line, not the exit status.
PYTHONPATH=. python3 astra_core/comprehensive_system_test.py | grep -E 'Capabilities tested|Passed|Failed'

# 4. Unit tests (333 passed)
PYTHONPATH=. python3 -m pytest -q

# 5. Live probe
PYTHONPATH=. python3 -c "from astra_core import create_stan_system; \
    s = create_stan_system(auto_start_orchestrator=False); \
    print(s.domain_registry.process_query('Jeans mass for n = 1e4 cm^-3 and T = 10 K')['answer'])"
```

Expected state after the third pass: 0 syntax-broken files; 680/680 modules import;
comprehensive test 18/18; `pytest -q` 333 passed; step 5 prints
`M_J = 2.8680 Msun`. Remaining log warnings refer to optional heavy dependencies (reportlab,
JAX, differentiable physics, deep-learning backends) and to `arc_agi`/`next_gen`/`legacy`
submodules that degrade to `None` by design.

#### E.8 Re-audit of the published tree (August 2026, third pass)

The first two passes were reported against a working copy. This section reports what the
**committed, published** tree actually did, and what it does after the third pass. Every
number here was measured with the commands in E.7.

**As published**

| Check | E.6 claimed | Measured |
|---|---|---|
| `.py` files under `astra_core/` | "682" / "567" | **675** |
| Modules importing cleanly | "567/567, zero failures" | **0 / 675** |
| Cause | — | `NameError: name 'np' is not defined` at `metacognitive/monitoring/monitor.py:317`, which poisons `import astra_core` and therefore every module beneath it |
| `comprehensive_system_test.py` | "18/18 (100 %)" | **0 / 18** — and the script **exits 0** regardless, so CI could not have caught it |
| "every declared test suite passes 100 %" | — | not reproducible; the suites could not import the package |

A single undefined name made the entire published package unimportable. The README shipped
alongside it stated "567/567 modules import cleanly" and a "100 % pass rate". That is the
central finding of this pass: **the verification claims were not being verified**, because
the harness reported success without checking, and the recipe that was supposed to check it
(E.7) omitted `PYTHONPATH=.`.

**After the third pass**

| Check | Result |
|---|---|
| `.py` files under `astra_core/` | **680** (5 added: the computational-domain framework, its regression tests and CI config) |
| Isolated import sweep | **680 / 680**, 0 failures |
| `comprehensive_system_test.py` | **18 / 18** |
| `pytest -q` | **333 passed** |
| Names exported as `None` by degraded imports | 63 → 5 |
| Hard-coded `confidence=` literals in `astra_core/domains/` | 194 → **0** |
| Domains with real computational capabilities | 0 → **31**, with **121** capabilities, each carrying a hand-computed `self_check` |
| Domains honestly reporting `NO_IMPLEMENTATION` | 0 → **17** |
| Curated-text domains relabelled `provenance='descriptive'`, confidence 0.20 | **27** |

**Defects found in the course of correcting this manual, and still open.** These are recorded
here rather than silently worked around, and are cross-referenced from the sections that hit
them:

1. **The system object has a second, unrepaired domain router.**
   `EnhancedUnifiedSTANSystem._find_relevant_domain` ranks domains by raw keyword count and
   breaks ties by registration order, bypassing the registry's repaired
   `find_best_domain_for_query`. `"Jeans mass for n = 1e4 cm^-3 and T = 10 K"` therefore
   returns curated `ism` text at confidence 0.20 through `system.answer()`, and
   `M_J = 2.8680 Msun` at 0.90 through `system.domain_registry.process_query()` (§4.1).
2. **A running orchestrator swallows every query.** With `auto_start_orchestrator=True`
   (the default), `process_query`/`answer` always return
   `"Query processed through orchestration system"` with a default confidence of 0.7,
   whatever was asked (§4.1).
3. **`comprehensive_system_test.py` exits 0 on total failure** (see above). Until that is
   fixed, its exit status is not a usable CI signal.
4. **V7.0 `generate_research_questions` largely ignores its `domain` argument**, returning a
   fixed cosmology list for most domains (§7.2.1).
5. **`write_publication` silently discards the prose sections** the publication engine builds,
   because the `Publication` dataclass has no field for them (§7.2.7).
6. **The seven specialised minds return `"<Discipline> analysis of: <query>"` with a
   hard-coded confidence of 0.7** — the same pattern the domain framework was rebuilt to
   eliminate (§9.1).
7. **`physics._newtonian_gravity` returns an acceleration (`GM/r²`) while its docstring says
   "force"**, and `_ideal_gas_law` accepts a `pressure` argument that it never uses (§11.1).
8. **`compute_physics` returns `value=None` instead of raising** when the parameter names do
   not match the model signature, which is easy to miss (§4.1).
9. **`RVDetector.detect_planets` applies no χ² or false-alarm cut** and leaves
   `RVSignal.false_alarm_prob` as `None`, so alias periods pass the SNR > 5 test (§8.2).
10. **The analogical-reasoning phenomenon database ships with three entries** and matches on
    exact names, so free-text targets return an empty list that reads like "no analogy
    exists" (§9.3).

**Correction to this manual itself.** The August 2026 pass also re-executed every Python code
block in this document. Eighteen documented methods and two documented import paths did not
exist; several worked examples printed numbers this software has never computed. All of it is
corrected in the body of the manual, with the incorrect claim quoted in each place so the
change is auditable. If you have quoted figures from an earlier edition of this manual — in a
paper, a grant application or a README — check them against §10.1 and this section before
they are used again.

---

## Index

No index is maintained for this document; use your reader's search. The entry points most
often looked for are:

| Looking for | Go to |
|---|---|
| Getting a number out of ASTRA | §4.1 Example 2, §10.3(a) |
| Why an answer has confidence 0.20 | §10.3(c), §10.1 |
| The real method names on the V7.0 scientist | §7.1, §11.4 |
| What is *not* implemented | §5 (bias / scaling / fusion boxes), §6, §7.4, §9.1–9.2 |
| Unit systems and constants | Appendix C, §11.1 |
| Whether the tree is in a known-good state | Appendix E.7 |
| What the earlier editions got wrong | Appendix E.8 |

---

**Document Version**: 7.3
**Last Updated**: 21 August 2026 — correction pass. Every Python code block in this manual was
re-executed against the repaired tree (680/680 modules import, comprehensive test 18/18,
`pytest -q` 333 passed). 18 non-existent methods and 2 non-existent import paths were removed
or replaced with working equivalents; §3.4, §5, §6, §7, §8 and §9 were rewritten around code that
runs; §10 was rewritten for the 31 / 17 / 27 domain split and provenance-derived confidence;
Appendix E.7 was fixed (`PYTHONPATH=.`) and Appendix E.8 added, recording the measured state
of the published tree.
**Authors**: Glenn J. White, Open University and Rutherford Appleton Laboratory, England
**License**: Apache License 2.0 - see `LICENSE` and `NOTICE` in the repository root.

For the latest version, visit: https://github.com/Tilanthi/ASTRA
