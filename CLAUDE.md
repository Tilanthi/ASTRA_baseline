# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ASTRA** (Autonomous Scientific Discovery in Astrophysics) is a unified AGI-inspired framework for autonomous hypothesis generation and validation in astronomy and astrophysics. The system integrates ~317,000 lines of clean, functional code across modular cognitive capabilities.

**Version**: 4.0.0 (`astra_core.__version__`)

> Version numbers in this tree are inconsistent: this file previously said 4.7,
> `astra_core.__version__` says `"4.0.0"`, the base system's runtime metadata
> reports `"3.1.0-ASTRO"`, and subpackages declare `37.0`, `41.0`, `50.0.0` and
> `60.0`. `astra_core.__version__` is the one to trust; the rest are historical
> labels.
>
> The former "AGI Capability Estimate: 70-75%" line has been removed: it was not
> derived from any measurement in this repository. For calibration, 48 of the 75
> "specialized astrophysics domain modules" are the same 110-line template whose
> `process_query` returns an f-string with a hard-coded `confidence=0.7`, and 73
> of the 75 contain no numerical computation at all.

### Codebase Integrity (August 2026 audit)

Two audits have been run. The first (recorded in **`User_Manual/User_Manual.md`,
Appendix E**) reported 567/567 clean imports and 100% on every suite. An
independent **re-audit** of the published tree found those numbers did not
describe the committed code: `import astra_core` failed outright at
`astra_core/metacognitive/monitoring/monitor.py:317`
(`NameError: name 'np' is not defined`), so 0 of 675 modules imported and
`comprehensive_system_test.py` scored 0/18 while still exiting 0. Appendix E's
own re-verification recipe (E.7) fails at step 2.

**Current, re-measured state** (reproduce with the commands under *Testing*):

| Check | Result |
|---|---|
| Files parsing (AST) | 675/675 |
| Fresh-interpreter import sweep | 664/675 — the 11 failures are `torch`/`astropy` absent (optional extras) |
| `comprehensive_system_test.py` | 18/18, exit 0 |
| `tests/test_all.py` / `tests/test_self_teaching.py` | 14 passed / 17 passed |
| Top-level names resolving to `None` | 5 (was 63) |

Read `BASELINE_README.md` → *Re-audit and repairs* and the *Known limitations*
section before trusting any physics output. Anything marked `# AUDIT-FLAG:` in
the source is known-suspect and deliberately unfixed.

### IMPORTANT: Naming Convention

The system was previously known as "STAN-XI-ASTRO" or "STAN". **It must now be referred to exclusively as "ASTRA"** in all:
- Academic papers and documentation
- External communications
- User-facing text
- Paper titles and abstracts

The internal codebase has been renamed from `stan_core` to `astra_core` for consistency with the ASTRA project name. Function names like `create_stan_system()` are retained for API backward compatibility.

**Full name**: ASTRA: Autonomous Scientific Discovery in Astrophysics
**Subtitle**: An AGI-inspired framework for autonomous hypothesis generation and validation

---

## CRITICAL: Persistent Memory Initialization

**IMPORTANT**: At the start of EVERY session, initialize the persistent memory system. This ensures:
- Previous session context is restored
- Known hallucinations are loaded and prevented
- User preferences are applied
- Anti-hallucination protection is active

```python
# RUN THIS AT SESSION START
from astra_core.memory.persistent import create_integrator, quick_hallucination_check

integrator = create_integrator()
integrator.initialize_session()
```

### Before Making Any Factual Claim

ALWAYS verify numerical claims against the hallucination register:

```python
result = integrator.verify_claim_before_output("54 MHz observations")
if not result.safe:
    # Use the correct value instead
    correct = result.hallucination_match.correct_value
```

### Known Hallucinations

The hallucination register is stored in `~/.astra_persistent/hallucination_register.json`.
To view or manage entries:

```python
from astra_core.memory.persistent import BootstrapMemory
bm = BootstrapMemory()
bm.list_hallucinations()  # View all entries
bm.remove_hallucination("54 MHz")  # Remove if no longer needed
```

### Document Review Protocol

When reviewing ANY document:
1. Extract key info first (frequencies, sample sizes, instruments)
2. Verify each claim with `quick_hallucination_check()`
3. Include mandatory anti-hallucination verification table in all reviews

### Checkpoint During Long Sessions

```python
# Periodically save session state
integrator.create_session_checkpoint({"current_task": "your task description"})
```

---

## Quick Start

### Basic System Usage

```python
from astra_core import create_stan_system

# Create system with auto-optimized capabilities
system = create_stan_system()

# Answer queries with automatic capability selection
result = system.answer("What causes filament width variations?")
print(result['answer'])
```

### V4.0 Revolutionary Capabilities

```python
from astra_core.revolutionary import create_v4_system, IntegrationMode

# Create V4.0 system with MCE, ASC, CRN, MMOL capabilities
system = create_v4_system()

# Process with different integration modes
result = system.process_query("Anze query", mode=IntegrationMode.FULL)
```

### Individual Capability Usage

```python
# Meta-Context Engine
from astra_core.metacognitive.meta_context_engine import create_meta_context_engine
mce = create_meta_context_engine()
result = mce.layer_context(query, dimensions=["temporal", "perceptual"])

# Domain modules
from astra_core.domains import DomainRegistry
registry = DomainRegistry()
registry.auto_load_domains()   # NB: `load_all_domains()` does not exist
result = registry.process_query("pulsar timing analysis")

# Physics engine
from astra_core.physics import UnifiedPhysicsEngine
physics = UnifiedPhysicsEngine()
result = physics.compute("blackbody", {"temperature": 5778, "wavelength": 500e-7})

# MAML optimizer
from astra_core.reasoning.maml_optimizer import create_maml_optimizer
optimizer = create_maml_optimizer(model_fn, loss_fn, n_inner_steps=5)
```

---

## Testing

**Run tests from the repository root with `PYTHONPATH=.`** — several suites do not fix their own import path. The suites below pass as of the August 2026 re-audit. Note that three of them could not previously report a truthful result: `comprehensive_system_test.py` never called `sys.exit`, `tests/test_comprehensive_integration.py` was structurally incapable of failing, and `tests/test_installation.py` was structurally incapable of passing. All three are fixed.

### Run All Tests

```bash
# Comprehensive system verification (18/18 capabilities required)
PYTHONPATH=. python astra_core/comprehensive_system_test.py

# V4.0 capability tests (5/5 suites)
PYTHONPATH=. python astra_core/tests/test_revolutionary/run_tests.py

# Specialist capability tests (66 V45 capabilities, 6/6)
PYTHONPATH=. python astra_core/tests/test_specialist_capabilities.py

# Phase 2-4 enhancement tests (6/6)
PYTHONPATH=. python astra_core/tests/test_phase_2_4.py

# Full unit suite (14/14) and self-teaching suite (17/17)
PYTHONPATH=. python astra_core/tests/test_all.py
PYTHONPATH=. python astra_core/tests/test_self_teaching.py

# Causal / discovery / outlier suites (all 100%)
PYTHONPATH=. python astra_core/tests/test_v47_causal_discovery.py
PYTHONPATH=. python astra_core/tests/test_v6_theoretical_discovery.py
PYTHONPATH=. python astra_core/tests/test_calibrated_outliers.py
```

### Run Specific Tests

```bash
# V4.0 individual capabilities (this runner self-fixes its import path)
python astra_core/tests/test_revolutionary/run_tests.py --mce        # Meta-Context Engine
python astra_core/tests/test_revolutionary/run_tests.py --asc        # Autocatalytic Self-Compiler
python astra_core/tests/test_revolutionary/run_tests.py --crn        # Cognitive-Relativity Navigator
python astra_core/tests/test_revolutionary/run_tests.py --mmol       # Multi-Mind Orchestration
python astra_core/tests/test_revolutionary/run_tests.py --integration # Integration tests
```

### Test Individual Components

```python
# Test physics modules
python -c "from astra_core.physics.relativistic_physics import RelativisticPhysics; print(RelativisticPhysics.schwarzschild_radius(1.989e33))"

# Test domain modules
python -c "from astra_core.domains.high_energy import create_high_energy_domain; d = create_high_energy_domain(); print(d.get_capabilities())"

# Test MAML optimizer
python -c "from astra_core.reasoning.maml_optimizer import MAMLOptimizer; print('MAML imported')"
```

---

## Architecture Overview

### System Layers (Bottom to Top)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Entry Points (Top Layer)                     │
│  create_stan_system() | create_v4_system() | process_query()   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                 V4.0 Revolutionary Capabilities                  │
│  MCE (Context) | ASC (Self-Improvement) | CRN (Abstraction)    │
│  MMOL (7 Specialized Minds)                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Domain Architecture                          │
│  BaseDomainModule → DomainRegistry → Specialized Domains        │
│  (9 domains: ISM, Star Formation, Exoplanets, GW, Cosmology,   │
│   Solar System, Time Domain, High-Energy, Galactic Archaeology) │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                Cross-Domain Meta-Learning                       │
│  MAMLOptimizer | CrossDomainMetaLearner | AdaptationResult      │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                   Physics & Causal Engines                      │
│  UnifiedPhysicsEngine | StructuralCausalModel | PCAlgorithm      │
│  PhysicsCurriculum | PhysicalAnalogicalReasoner                │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                  Memory & Knowledge Systems                     │
│  MORKOntology | MemoryGraph | VectorStore | WorkingMemory       │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Capabilities Registry                         │
│  66+ specialist capabilities (V36-V94) with auto-selection      │
└─────────────────────────────────────────────────────────────────┘
```

### Module Communication Patterns

**Domain Hot-Swapping**: All domain modules inherit from `BaseDomainModule` with standardized `process_query()` interface. Domains are loaded/unloaded at runtime via `DomainRegistry`. No system restart required.

**Graceful Degradation**: Every import wrapped in try/except with fallback. Check `BASE_UNIFIED_AVAILABLE`, `DomainRegistry`, etc. for availability before use. System continues in degraded mode when components missing.

**Meta-Learning Coordination**: `CrossDomainMetaLearner` observes all domain queries, builds transfer learning models, enables few-shot adaptation. Connected to `MAMLOptimizer` for inner-loop optimization.

**Multi-Mind Orchestration**: 7 specialized minds (Physics, Empathy, Politics, Poetry, Mathematics, Causal, Creative) process queries in parallel. `MindArbitrator` resolves conflicts using anticipatory confidence prediction.

---

## Key Design Patterns

### 1. Capability Auto-Selection

The system automatically selects capabilities based on task analysis. Do not manually invoke capabilities unless specifically testing individual components.

```python
# WRONG: Manual capability selection
result = system.reasoning.causal_discovery(query)

# CORRECT: Let system auto-select
result = system.answer(query)  # Auto-selects best capabilities
```

### 2. Module Registration Pattern

All domain modules use `@register_domain` decorator or explicit `DomainModuleRegistry.register()`. This enables runtime discovery and hot-swapping.

```python
from astra_core.domains import BaseDomainModule, register_domain

@register_domain
class MyDomain(BaseDomainModule):
    def get_default_config(self):
        return DomainConfig(
            domain_name="my_domain",
            version="1.0.0",
            keywords=["keyword1", "keyword2"],
            capabilities=["capability1", "capability2"]
        )
```

### 3. Factory Function Pattern

All major components use factory functions for creation, not direct constructors. This enables configuration injection and graceful fallback.

```python
# Use factory functions
system = create_stan_system()
mce = create_meta_context_engine()
optimizer = create_maml_optimizer(model_fn, loss_fn)

# NOT: system = UnifiedSTANSystem()  # Avoid direct constructors
```

### 4. Physics Curriculum Learning

Physics capabilities develop through staged curriculum (`ComplexityLevel.BASIC` → `EXPERT`). Do not skip stages. Use `PhysicsCurriculum.get_next_stage()` for progression.

---

## File Organization Conventions

### Capability Files

- **Capabilities**: `astra_core/capabilities/*.py` and subpackages (`cognitive/`, `metacognitive/`, `memory/`, `integration/`, `causal/`) — organized by function, not by version number
- **Legacy versioned systems**: `astra_core/legacy/systems/vXX/` (v36–v94 era code kept for back-compatibility; e.g. the V36 causal engines used by `astro_physics/molecular_cloud_v36.py`)
- **Astrophysics modules**: `astra_core/astro_physics/*.py` (radiative_transfer.py, sph_gas_dynamics.py, turbulence_analysis.py, infrared_submm.py, multiscale_coupling.py, … — all rebuilt/validated in the Aug 2026 audit)
- **Physics modules**: `astra_core/physics/*.py` (relativistic_physics.py, quantum_mechanics.py, nuclear_astro.py)
- **Domain modules**: `astra_core/domains/<domain_name>/__init__.py`
- **Meta-learning**: `astra_core/reasoning/maml_optimizer.py`, `cross_domain_meta_learner.py`

### Memory Hierarchy

- **MORK Ontology**: `astra_core/memory/mork_ontology.py` (concept hierarchies)
- **Memory Graph**: `astra_core/memory/context_graph.py` (context relationships)
- **Working Memory**: `astra_core/memory/working/` (7±2 capacity constraint)

### Test Files

- **V4 integration tests**: `astra_core/tests/test_revolutionary/test_v4_integration.py`
- **Capability tests**: `astra_core/tests/test_specialist_capabilities.py`
- **Validation**: `astra_core/tests/validation_benchmarks.py` (a library of benchmarks, exercised via `test_phase_2_4.py`; produces no standalone output)

---

## Important Constants

### Physics Constants (CGS units)

Defined in `UnifiedPhysicsEngine.constants`:
- `G`: 6.674e-8 (gravitational)
- `c`: 2.998e10 (speed of light)
- `h`: 6.626e-27 (Planck)
- `k_B`: 1.381e-16 (Boltzmann)
- `M_sun`: 1.989e33 (solar mass)
- `R_sun`: 6.957e10 (solar radius)

### Abstraction Scale (CRN)

0 = atomic facts, 50 = concepts, 100 = pure philosophy

### Cognitive Frames (MCE)

PREDICTIVE, ANALYTICAL, EMOTIONAL, CREATIVE, CRITICAL, SYNTHETIC, NARRATIVE, CONTEMPLATIVE

---

## Common Pitfalls

1. **Missing Imports**: Always check for import availability. Most imports wrapped in try/except with None fallback. Test `if MODULE is not None:` before use.

2. **Direct Construction**: Never directly instantiate capability classes. Use factory functions: `create_<module>()`.

3. **Hardcoded Physics Values**: Always use `UnifiedPhysicsEngine.constants`, never hardcode physical constants.

4. **Skipping Initialization**: Domain modules must call `.initialize(global_config)` after creation before `.process_query()`.

5. **Backup File Accumulation**: Stray `*.bak` / `*.backup` files exist in the tree (e.g. `astra_core/core/__init__.py.bak`, `astra_core/utils/pdf_generator.py.backup`); the old `cleanup_astra_core.py` / `cleanup_bloat.py` scripts are gone — delete stale backups manually when encountered.

6. **Test Paths and PYTHONPATH**: Run every suite from the repository root with `PYTHONPATH=.` (see Testing). `tests/ablation/run_ablations.py` must be run from inside `tests/ablation/` by design.

7. **NumPy 2.x**: `np.trapz` was removed — use `np.trapezoid`. `skimage.morphology.skeletonize`, not `scipy.ndimage.skeletonize` (the scipy import path never existed and silently degraded imports).

8. **Silent Degradation Hides Stale Imports**: A `try/except ImportError` fallback means a wrong module path fails *silently* (symbol becomes `None`, `*_AVAILABLE` becomes `False`). After moving or renaming any module, grep for old paths — the Aug 2026 audit found several features disabled for months this way (see User_Manual Appendix E).

---

## PDF Generation Requirements

When generating PDF documents using `astra_core/utils/pdf_generator.py`:

### Critical Rules

1. **NEVER convert single asterisks to italic**: The markdown `*text*` pattern MUST NOT be converted to `<i>text</i>` because asterisks are used in mathematical expressions (e.g., `dyn*cm^2/g^2`). Converting this would produce broken output like `dyn<i>cm^2/g^2</i>`.

2. **Only convert bold formatting**: Only `**text**` should be converted to `<b>text</b>`. This is safe because double asterisks are rarely used in scientific notation.

3. **Escape HTML properly**: All HTML special characters (`<`, `>`, `&`) must be escaped to `&lt;`, `&gt;`, `&amp;` EXCEPT for the intentionally converted bold tags.

4. **Convert unicode to ASCII**: All non-ASCII characters must be converted to ASCII equivalents. Greek letters become names (alpha, beta, gamma), mathematical symbols become ASCII approximations (± -> +/-, × -> x, etc.).

5. **Test PDF output**: Always verify generated PDFs do not contain:
   - Raw HTML tags like `<i>`, `</i>`, `<b>` appearing as visible text
   - Markdown formatting like `**bold**` appearing literally
   - Unicode replacement characters (boxes, question marks)
   - Broken formatting from asterisk-to-italic conversion

### Implementation Pattern

```python
def _process_inline_formatting(self, text: str) -> str:
    # Step 1: Protect bold tags with placeholders
    text = re.sub(r'\*\*([^*]+?)\*\*', r'%%BOLD_START%%\1%%BOLD_END%%', text)

    # Step 2: Escape ALL HTML special characters
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')

    # Step 3: Restore protected bold tags
    text = text.replace('%%BOLD_START%%', '<b>')
    text = text.replace('%%BOLD_END%%', '</b>')

    # DO NOT convert single * to <i> - causes math expression corruption!
    return text
```

---

## Development Workflow

1. **Test before modifying**: Always run relevant tests first to establish baseline
2. **Respect graceful degradation**: Any new module must have try/except imports and fallback behavior
3. **Use factory functions**: Create via `create_<module>()` pattern
4. **Register new domains**: Use `@register_domain` decorator for discoverability
5. **Update exports**: Add new public classes to `__all__` in module `__init__.py`

---

## MNRAS Paper Writing Guidelines

When generating LaTeX papers for submission to *Monthly Notices of the Royal Astronomical Society* (MNRAS), always conform to these standards:

### Document Class and Template

```latex
\documentclass{mnras}
% OR for older papers:
% \documentclass[useAMS]{mnras}
```

- **Always use** the official `mnras` class (not article, aastex, etc.)
- **Never hardcode** page breaks, margins, or formatting—let the class handle it
- **Two-column** format is standard (single-column only for "paper" type if requested)

### Required Packages

```latex
\usepackage{newtxtext,newtxmath} % Times font (MNRAS standard)
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{natbib}      % MNRAS citation style
\usepackage{hyperref}     % For \url{} commands
```

### Title and Authors

```latex
\title[Short Title]{Full Title of the Paper}
\author[Author et al.]{
First Author,\(^1\)
Second Author,\(^1\) 
Third Author\(^{1,2}\) 
and Fourth Author\(^3\)
\\
\(^1\)Institution One, \(^2\)Institution Two, \(^3\)Institution Three
}
\date{Accepted XXX. Received YYY; in original form ZZZ.}
```

### Section Headings

- **Section**: `\section{Title}` (numbered, bold)
- **Subsection**: `\subsection{Title}` (numbered, bold)
- **Subsubsection**: `\subsubsection{Title}` (numbered, italic)
- **Paragraph**: `\paragraph{Title}` (not numbered, italic—use sparingly)

### Mathematics

- **Inline math**: Use `$...$` (not `$$...$$` or `\(...\)`)
- **Display math**: Use `\[...\]` or `$$...$$` for equations
- **Equation numbering**: Use `\begin{equation}...\end{equation}` for numbered equations
- **Alignment**: Use `\begin{align}...\end{align}` for multi-line equations
- **Units**: Use `siunitx` package: `\unit{m.s^{-1}}` or `\si{m.s^{-1}}`

### Figures and Tables

```latex
% Figures
\begin{figure}[t!] % Place at top
\includegraphics[width=\columnwidth]{filename.pdf}
\caption{Caption text.}
\label{fig:label}
\end{figure}

% Tables
\begin{table}[t!]
\caption{Caption text.}
\label{tab:label}
\begin{tabular}{lcc}
Column 1 & Column 2 & Column 3 \\
...
\end{tabular}
\end{table}
```

- **Figures**: Submit as high-resolution PDF, EPS, or PNG (min 300 dpi)
- **Placement**: Use `[t!]` for top, `[b!]` for bottom, `[h!]` for here
- **Captions**: Place **above** tables, **below** figures
- **Width**: Use `\columnwidth` for single-column, `0.5\textwidth` for two-column figures

### References and Citations

```latex
% In text
\citep{Author2020}      % (Author, 2020)
\citet{Author2020}      % Author (2020)
\citeauthor{Author2020} % Author
\citeyear{Author2020}   % 2020

% Multiple citations
\citep{Author1a2020,Author1b2020,Author2}

% In bibliography
\bibliographystyle{mnras}
\bibliography{references}
```

- **Style**: Use `natbib` with `mnras` bibliography style
- **BibTeX**: Store references in `references.bib`
- **Format**: `Author, A. A., Year, MNRAS, vol, pages`
- **DOIs**: Always include DOI in BibTeX entries

### Common BibTeX Entry Format

```bibtex
@article{Author2020,
  author = {Author, A. A. and Second, B. B.},
  title = {Title of the Paper},
  journal = {MNRAS},
  year = {2020},
  volume = {123},
  pages = {456--467},
  doi = {10.1093/mnras/staa123}
}
```

### Common MNRAS Commands

- `\mnras{volume}{page}` → MNRAS, 123, 456
- `\aap{volume}{page}` → A\&A, 123, 456
- `\apj{volume}{page}` → ApJ, 123, 456
- `\apjl{volume}{page}` → ApJL, 123, 456
- `\araa{volume}{page}` → ARA\&A, 123, 456

### Color Usage

- **Main text**: Black only
- **Figures**: Color allowed (online), ensure grayscale readability for print
- **Highlights**: Avoid red/green (colorblindness considerations)

### Code and Algorithms

- **Pseudocode**: Use `algorithm` environment with `algorithmic` package
- **Listings**: Use `listings` package for code snippets
- **File names**: `\texttt{filename.py}` (monospaced)

### Critical Formatting Rules

1. **Never** use `\newpage` or `\clearpage`—let LaTeX paginate
2. **Never** hardcode page numbers or margins
3. **Always** check for widows/orphans in final draft
4. **Always** run BibTeX after `\cite{}` changes
5. **Never** use `\mathcal{}` for standard math—use `\mathit{}` for multi-letter variables
6. **Always** escape special characters: `& % $ # _ { } ~ ^ \`
7. **Never** use `\\` for line breaks in text (use blank line instead)

### Compilation Sequence

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

### Submission Checklist

- [ ] Document class is `mnras`
- [ ] All figures are high-resolution (min 300 dpi)
- [ ] All tables have `\caption{}` above
- [ ] All figures have `\caption{}` below
- [ ] References use `natbib` with `\citep{}` or `\citet{}`
- [ ] BibTeX file is `references.bib`
- [ ] DOI included for all modern references
- [ ] No hardcoded page breaks or formatting
- [ ] Math uses proper delimiters (`$...$` inline, `\[...\]` display)
- [ ] Author affiliations use superscripts: `First Author,\(^1\)`
- [ ] Abstract is 250 words or less
- [ ] Keywords included (5-8 recommended)

### When to Override These Rules

Only deviate from MNRAS style when:
1. Explicitly requested by referee
2. Using `arXiv` overlay class (then revert to `mnras` for submission)
3. Author guidelines have been updated (check current MNRAS author guide)

---

## Post-Upgrade Verification Testing

**CRITICAL**: After any substantial upgrade to ASTRA functionality or astra_core components, comprehensive verification testing MUST be performed to ensure all dependencies, files, and components remain properly linked.

### When to Run Comprehensive Tests

Run the comprehensive system verification after:
- Adding new domain modules
- Modifying core architecture (unified.py, unified_enhanced.py)
- Updating physics engine or models
- Changes to memory systems
- Adding or modifying reasoning capabilities
- Refactoring module dependencies
- Any changes to import chains or module registration

### Comprehensive Test Procedure

```bash
# Run the comprehensive system test (from repo root)
PYTHONPATH=. python astra_core/comprehensive_system_test.py

# Expected output: All 18 capabilities should PASS (100%)
```

The comprehensive test verifies:
- **75 Domain Modules**: Import, instantiation, and query handling (100% pass rate required)
- **Memory Systems**: MORK Ontology, Context Graph, Working Memory, Episodic Memory
- **Physics Engine**: UnifiedPhysicsEngine with all models and constraints
- **Causal Discovery**: V50, V70, and astrophysical causal discovery engines
- **Advanced Reasoning**: Swarm reasoning, hierarchical Bayesian meta-learning
- **V4 Capabilities**: Meta-Context Engine (if available)
- **Orchestrator Integration**: create_stan_system(), answer(), process_query()

After any substantial change, also confirm:
- **Full import sweep**: 664 of 675 modules import in a fresh interpreter; the 11 that do not are the `torch`/`astropy` optional extras (re-measured Aug 2026 — the previously quoted 567/567 was not reproducible, and the tree contains 675 Python files, not 567)
- **Zero UserWarnings on `import astra_core.core`** — a warning here means a stale import path is silently disabling a component

### Fix-Test Loop

If errors are found:
1. **Fix the identified error** (missing imports, broken dependencies, incorrect signatures, etc.)
2. **Re-run the comprehensive test**
3. **Repeat** until ALL capabilities pass (100% pass rate)
4. **Document the fix** if it's a recurring pattern

### Test Files Reference

- **Comprehensive Test**: `astra_core/comprehensive_system_test.py`
- **Domain Validation**: `astra_core/tests/validation_benchmarks.py`
- **V4 Integration Tests**: `astra_core/tests/test_revolutionary/test_v4_integration.py`
- **Specialist Capabilities**: `astra_core/tests/test_specialist_capabilities.py`

### Verification Report

The standing verification record lives in **`User_Manual/User_Manual.md` Appendix E** (current version documents the August 2026 audit: 17 re-implemented symbols, the 14-module `astro_physics` rebuild, ~13,500 dead lines removed, and every suite green). After a successful verification, update Appendix E (E.6 test matrix and the document version footer) with the new date and results rather than creating separate report files.

---

## Code Statistics

- **Total Lines**: 316,898 Python lines
- **Python Files**: 675
- **Directory Size**: ~13 MB (excluding `__pycache__`)
- **Specialist Capabilities**: 66 (V45 baseline)
- **Domain Modules**: 75 (23 core + 48 astrophysics)
- **Physics Stages**: 15 learning stages (relativistic, quantum, nuclear)
- **Audit Delta (Aug 2026)**: ~200 files changed — +12.6k lines of re-implemented lost symbols, −34.8k dead/broken lines (see User_Manual Appendix E)
