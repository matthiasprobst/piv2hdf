# Dependency maintenance

Updated 2026-11-01 for `2.0.0a3` on `fix/scientific-conversion-integrity`.

The package requires Python 3.9-3.11 and h5rdmtoolbox >=2.8.3,<3.0.0.
Validation uses the bundled published standard-name table through a read-only
rdflib index. The legacy ssnolib object parser and the unused pivmetalib dependency
were removed; no migration to their next major API families is required.
ontolutils >=0.27.5,<0.28.0 supplies unit identifiers; rdflib >=7.1,<8.0 parses
the table. These direct requirements match `setup.cfg` and `requirements.txt`.
Keep development installations isolated from unrelated workspace environments.

| File | Before | After |
| --- | --- | --- |
| setup.cfg | `h5rdmtoolbox>=2.2.0,<3.0.0` | `h5rdmtoolbox>=2.8.3,<3.0.0` |
| requirements.txt | `h5rdmtoolbox>=2.2.0,<3.0.0` | `h5rdmtoolbox>=2.8.3,<3.0.0` |

Dependencies on related packages must exclude the next major release.
Retain tighter existing pins unless an upgrade is deliberate and verified.
Pre-1.0 API changes can also require a minor-version ceiling.
Self-references that compose extras are exempt.

For future work: inspect status, create a task branch before editing, preserve
existing changes, stage explicit paths, inspect the staged diff, run appropriate
checks, and commit on that branch. Keep this record, README, and duplicated
requirements in sync. Ask for explicit approval before starting subagents.

The workspace coordination repository contains the complete inventory and
repeatable checks in `docs/dependency-audit.md` and
`tools/audit_dependencies.py`. The audit covers manifests and emitted package
metadata; version ranges alone are not a runtime compatibility guarantee.

## Verification history

The scientific implementation is tested with the adjacent editable h5RDMtoolbox
2.8.3 checkout. Install that checkout first when testing this workspace baseline:
`python -m pip install -e ../h5RDMtoolbox`, then install this project's development
requirements. Local checkout results do not certify a published artifact or every
dependency combination permitted by the bounds. See `MIGRATION.md` for changed
scientific and configuration contracts.

The following audit predates the 2.0.0a3 implementation:

Workspace manifest audit: 14 folders, 39 manifests, 40 cross-package
requirements checked; all related dependencies exclude their next major.
All h5rdmtoolbox requirements accept 2.8.3 and reject 3.0.0.

Setuptools emitted package metadata successfully in a temporary metadata-only
scaffold using the actual manifests. Related Requires-Dist bounds passed.
This verifies dependency metadata rather than building or installing the package.

Broad application suites and a full online dependency resolution were not run.
These checks do not certify every release in the permitted version ranges.
