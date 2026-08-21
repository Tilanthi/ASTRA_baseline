# Installing ASTRA

Every command below was executed against a clean clone of this branch before
being written down. Where a step's output is quoted, that is the actual output.

## Requirements

- **Python 3.9–3.12.** On a Mac, `python3 --version` tells you what you have; if
  it is older than 3.9, install a newer one (`brew install python@3.12`).
- **git**. Pre-installed on macOS, or `xcode-select --install`.
- About **2 GB** of disk for the core install, or ~5 GB if you add PyTorch.

## Install

```bash
# 1. a fresh folder
mkdir -p ~/ASTRA && cd ~/ASTRA

# 2. clone this branch
git clone --branch audit-fixes-aug2026 \
    https://github.com/Tilanthi/ASTRA_baseline.git .

# 3. an isolated environment, so ASTRA cannot disturb your other Python work
python3 -m venv .venv
source .venv/bin/activate

# 4. install
pip install --upgrade pip
pip install -e .
```

`pip install -e .` installs in *editable* mode: the package points at the
checkout, so `git pull` updates your installation with no reinstall.

### Optional extras

The core install is deliberately small. Add only what you need:

```bash
pip install -e ".[astro]"    # astropy: FITS I/O, archive access
pip install -e ".[image]"    # scikit-image: better filament skeletonisation
pip install -e ".[science]"  # matplotlib, emcee, corner, statsmodels
pip install -e ".[dev]"      # pytest, pyflakes -- needed to run the test suite
pip install -e ".[pdf]"      # reportlab: PDF generation
pip install -e ".[dl]"       # PyTorch (large) -- only for astro_physics/deep_learning
pip install -e ".[all]"      # everything above
```

Without an extra, the modules that need it degrade to `None` and log why; they
do not crash the package. Measured on a core-only install:
**674 of 680 modules import**, the 6 exceptions being the four
`astro_physics.deep_learning` modules (PyTorch) and two ablation plotting
scripts (matplotlib).

With `[astro,image,dev]` added: **all 333 tests pass**.

## Verify your installation

```bash
# from the repo root, with the venv active
PYTHONPATH=. python astra_core/comprehensive_system_test.py   # expect 18/18
python -m pytest astra_core/tests -q                          # expect 333 passed
```

Both exit non-zero on failure, so they are safe to use in a script. (They did
not before this branch: `comprehensive_system_test.py` exited 0 while reporting
0/18.)

Full import check:

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

## First use

```python
import astra_core

system = astra_core.create_stan_system()
print(system.answer("What causes supernovae?")["answer"])
```

Asking a **quantitative** question routes to a domain that computes:

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

`confidence` is derived from what actually happened, not asserted:

| `metadata['provenance']` | confidence | meaning |
|---|---:|---|
| `computed` | 0.90 | a verified routine ran on parameters taken from your query |
| `capability_available` | 0.40 | the routine exists but your query did not supply its parameters |
| `descriptive` | 0.20 | curated reference text; **no analysis was performed** |
| `none` | 0.0 | the domain has no implementation |

## Before you rely on a number

Read **`BASELINE_README.md` → Known limitations**. In short: 31 of the 75
domains compute (121 capabilities, each pinned to a hand-computed reference
value); 17 report `NO_IMPLEMENTATION` honestly; 27 return curated text.
Anything marked `# AUDIT-FLAG:` in the source is known-suspect and deliberately
unfixed.

## Updating later

```bash
cd ~/ASTRA
source .venv/bin/activate
git pull
```

If `main` has been updated to include this work, switch with
`git checkout main && git pull`.

## Troubleshooting

**`ModuleNotFoundError: No module named 'astra_core'`** — the virtualenv is not
active. Run `source .venv/bin/activate` (you should see `(.venv)` in your
prompt).

**A test suite says `No module named 'astra_core'`** — some suites do not fix
their own import path; run them from the repo root with `PYTHONPATH=.`.

**`No module named pytest`** — `pip install -e ".[dev]"`.

**Apple Silicon and PyTorch** — `[dl]` is only needed for the four
`astro_physics.deep_learning` modules. Everything else, including all 333 tests,
works without it.
