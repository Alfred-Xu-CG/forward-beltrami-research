# Forward Beltrami research repository

This repository is the canonical, synchronized home for the forward Beltrami
solver exploration. It contains source code, experiment drivers, compact JSON
or CSV receipts, route ledgers, and self-contained theory summaries.

## Current Phase VI

The active 18-hour dense-homeomorphism research plan is
[`docs/research_phase6/PLAN.md`](docs/research_phase6/PLAN.md). Start with the
[self-contained Phase VI report](docs/research_phase6/REPORT.md), then use its
route-specific links and [`results.csv`](docs/research_phase6/results.csv) to
find exact setups, code, and per-case raw JSON. The current phase rules in
[`AGENTS.md`](AGENTS.md) and
[`docs/research_phase6/AGENTS_RESEARCH_MODE.md`](docs/research_phase6/AGENTS_RESEARCH_MODE.md)
supersede the older review checklist below; no additional audit framework or
skill workflow is required for Phase VI.

## Layout

- `src/qcopt/`: reusable numerical implementations.
- `src/qcopt/experiments/`: reproducible experiment entry points.
- `artifacts/`: compact receipts committed with the corresponding experiment.
  Large `.npz`, `.npy`, `.png`, and `.html` intermediates remain on the D: drive
  and are intentionally ignored.
- `docs/forward_beltrami/`: live route ledger, validation notes, and the final
  thematic summary.
- `tests/`: regression and differentiability tests.

## Review protocol

Every new result should include:

1. the exact command and mesh resolution;
2. a JSON receipt containing timings, errors, topology checks, and limitations;
3. an update to `docs/forward_beltrami/route_status.md`;
4. an independent orientation/injectivity audit; and
5. a focused test or a documented reason why no test applies.

The repository is deliberately conservative: a finite-resolution zero-flip
experiment is evidence, not a universal homeomorphism theorem. The acceptance
criteria for the final solver are in
`docs/forward_beltrami/summary/10_experiment_protocol.md`.

## Synchronization

The canonical GitHub remote is the private repository
`Alfred-Xu-CG/forward-beltrami-research`. Pushes are made from
`D:\QC_optimization`; no SSH credentials or temporary tunnel files are stored
in the repository.
