# Continuous integration

`github-workflow-ci.yml` is a ready-to-use GitHub Actions workflow. It is parked
here rather than at `.github/workflows/ci.yml` because the token used to push
this branch does not carry the `workflow` OAuth scope, and GitHub rejects any
push that creates or modifies a workflow file without it.

To enable it:

```bash
mkdir -p .github/workflows
git mv ci/github-workflow-ci.yml .github/workflows/ci.yml
git commit -m "Enable CI"
git push
```

(or paste the file in through the GitHub web UI, which does not need the scope).

## What it checks, and why

Each job is one of the checks that would have caught the state this branch
repairs — a tree whose README advertised "567/567 module imports, zero failures"
while `import astra_core` failed on the first statement it reached.

| Job | Gate |
|---|---|
| `import-sweep` | every file AST-parses; every module imports in a **fresh interpreter** with all extras installed (678/678); the `undefined name` count does not grow beyond the current 14 |
| `tests` | `pytest astra_core/tests`; the comprehensive system test (which now exits non-zero on failure — it previously exited 0 at 0/18); every declared print-based suite |
| `no-fabricated-tests` | rejects `return {'passed': True}` helpers outside `tests/` — 44 such always-green "tests" were embedded in shipped science modules |

The single most important of these is the fresh-interpreter import sweep. The
defect that broke the published tree was invisible to any check that imported
`astra_core` only once, or that ran inside an already-warm interpreter.
