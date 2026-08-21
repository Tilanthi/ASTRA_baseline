# Installing ASTRA

Every command below was executed against a clean clone of this branch, in an
empty folder, before being written down. Where output is quoted, it is the
actual output of that run.

## Requirements

- **Python 3.9–3.12.** Check with `python3 --version`. On a Mac, if it is older
  than 3.9: `brew install python@3.12`.
- **git**. Pre-installed on macOS, or `xcode-select --install`.
- About **2 GB** of disk, or ~5 GB if you add PyTorch.

## Install

> **The `--branch` argument matters.** The fixes live on
> `audit-fixes-aug2026`. A plain `git clone` gives you `main`, which does not
> import at all.

```bash
# 1. a genuinely fresh folder
mkdir -p ~/ASTRA_fresh && cd ~/ASTRA_fresh

# 2. clone the corrected branch
git clone --branch audit-fixes-aug2026 \
    https://github.com/Tilanthi/ASTRA_baseline.git .

# 3. an isolated environment
python3 -m venv .venv
source .venv/bin/activate

# 4. install
pip install --upgrade pip
pip install -e ".[astro,image,dev]"
```

`-e` installs in editable mode, so `git pull` updates your installation with no
reinstall. The three extras give you FITS I/O (`astro`), the better filament
skeletoniser (`image`), and `pytest` so you can run the suite (`dev`).

Other extras, none of them required:

```bash
pip install -e ".[science]"  # matplotlib, emcee, corner, statsmodels
pip install -e ".[pdf]"      # reportlab, for PDF generation
pip install -e ".[dl]"       # PyTorch (large) -- only for astro_physics/deep_learning
pip install -e ".[all]"      # everything
```

Without an extra, the modules needing it degrade to `None` and log why; they do
not crash the package.

## Verify

### 1. The test suites

```bash
# from the repo root, venv active
PYTHONPATH=. python astra_core/comprehensive_system_test.py   # expect 18/18
python -m pytest astra_core/tests -q                          # expect 364 passed
```

Both exit non-zero on failure, so they are safe in a script. They did not before
this branch: `comprehensive_system_test.py` exited 0 while reporting 0/18.

### 2. Every module imports

```bash
python - <<'EOF'
import importlib, os, warnings
mods = []
for dp, dn, fn in os.walk('astra_core'):
    dn[:] = [d for d in dn if d != '__pycache__']
    for f in fn:
        if f.endswith('.py'):
            m = os.path.join(dp, f)[:-3].replace(os.sep, '.')
            mods.append(m[:-9] if m.endswith('.__init__') else m)
fail = []
for m in sorted(set(mods)):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            importlib.import_module(m)
    except BaseException as e:
        fail.append((m, type(e).__name__))
print(f"{len(set(mods)) - len(fail)}/{len(set(mods))} modules import")
for m, t in fail:
    print("  ", m, t)
EOF
```

Expected with `[astro,image,dev]`: **684/690**, the six exceptions being the
four `astro_physics.deep_learning` modules (PyTorch) and two ablation plotting
scripts (matplotlib). With `[all]`: **690/690**.

### 3. ★ That the install is genuinely self-contained

This is the check that matters most, and the one whose absence caused a real
failure: a developer machine holding several ASTRA checkouts can satisfy an
import from *a different folder*, so the install looks fine locally and fails
for everyone else.

**Run this from a directory nowhere near any other ASTRA checkout** — `/tmp` is
ideal:

```bash
cd /tmp && python - <<'EOF'
import os, astra_core
print("astra_core resolved from:", os.path.dirname(astra_core.__file__))

# the published V5.0 discovery API, which analysis scripts import
from astra_core.capabilities.v101_temporal_causal import create_temporal_fci_discovery
from astra_core.capabilities.v102_counterfactual_engine import create_counterfactual_engine
from astra_core.capabilities.v103_multimodal_evidence import create_multimodal_evidence_fusion
from astra_core.capabilities.v104_adversarial_discovery import create_adversarial_discovery_system
from astra_core.capabilities.v105_meta_discovery import create_meta_discovery_transfer_engine
from astra_core.capabilities.v106_explainable_causal import create_explainable_causal_reasoner
from astra_core.capabilities.v107_discovery_triage import create_discovery_triage_system
from astra_core.capabilities.v108_streaming_discovery import create_streaming_discovery_engine
print("all 8 published V5 capability imports: OK")
EOF
```

The printed path **must** be inside the folder you just installed. If it points
anywhere else, that is what you are really running.

### 4. That the documentation matches the code

```bash
python astra_core/tests/check_doc_examples.py    # expect 0 failing blocks
```

This executes every `python` example in `README.md`, `BASELINE_README.md`,
`CLAUDE.md` and `User_Manual/User_Manual.md`. Sixteen of twenty-four failed
before this branch.

## First use

```python
import astra_core

system = astra_core.create_stan_system()
print(system.answer("What causes supernovae?")["answer"])
```

A **quantitative** question routes to a domain that computes:

```python
import os
from astra_core.domains import DomainRegistry

names = sorted(d for d in os.listdir("astra_core/domains")
               if os.path.isdir(f"astra_core/domains/{d}") and d != "__pycache__")
registry = DomainRegistry()
registry.auto_load_domains({n: {} for n in names})

r = registry.process_query("What is the Jeans mass for n = 1e4 cm^-3 and T = 10 K?")
print(r["domain"], r["confidence"])
print(r["answer"])
# statistical_mechanics 0.9
# Jeans mass of an isothermal self-gravitating gas: M_J = 2.8680 Msun
#   (formula: M_J = (pi^(5/2)/6) c_s^3 G^(-3/2) rho^(-1/2))
```

`confidence` is derived from what actually happened, never asserted:

| `metadata['provenance']` | confidence | meaning |
|---|---:|---|
| `computed` | 0.90 | a verified routine ran on parameters taken from your query |
| `capability_available` | 0.40 | the routine exists, your query did not supply its parameters |
| `descriptive` | 0.20 | curated reference text; **no analysis was performed** |
| `none` | 0.0 | the domain has no implementation |

## Running existing analysis scripts

Scripts written against the published V5.0 API work unmodified — the flat
`astra_core.capabilities.v101_temporal_causal` … `v108_streaming_discovery`
paths, `edge[0]` indexing, `result.get('temporal_edges')`, `max_lag=`,
`Intervention(variable=, value=)` and the `'alerts'` result key are all
supported again.

You will still need to point any **hardcoded data paths** in your own scripts at
their own folders.

> **If you have results from `CounterfactualEngine` / V102 computed before
> 21 Aug 2026, recompute them.** The DML estimator divided by the mean of a
> residual (~0 by construction) and returned values with the wrong sign and a
> divergent magnitude — a true ATE of +2.0 came back as −175.8. It is fixed and
> pinned to known ground truth, and is now seeded so a result is reproducible.

## Before you rely on a number

Read **`BASELINE_README.md` → Known limitations**. In short: of the 75 domains,
31 compute (121 capabilities, each pinned to a hand-computed reference value),
17 report `NO_IMPLEMENTATION` honestly, and 27 return curated reference text.
Anything marked `# AUDIT-FLAG:` in the source is known-suspect and deliberately
left unfixed rather than papered over.

## Updating later

```bash
cd ~/ASTRA_fresh
source .venv/bin/activate
git pull
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'astra_core'`** — the virtualenv is not
active. `source .venv/bin/activate` (you should see `(.venv)` in your prompt).

**`NameError: name 'np' is not defined` on import** — you are on `main`, not
`audit-fixes-aug2026`. Check with `git rev-parse --abbrev-ref HEAD`.

**A suite says `No module named 'astra_core'`** — some suites do not fix their
own import path; run them from the repo root with `PYTHONPATH=.`.

**`No module named pytest`** — `pip install -e ".[dev]"`.

**Apple Silicon and PyTorch** — `[dl]` is needed only for the four
`astro_physics.deep_learning` modules. Everything else, including all 364 tests,
works without it.
