# Development and dependencies

Use an isolated **Python 3.9–3.11** environment. From the repository root:

```shell
python -m pip install -e ".[test]"
python -m pytest -q
```

For DaVis coverage on Windows/Linux, install `python -m pip install -e ".[test,davis]"`.
Without `lvpyio`, the DaVis-specific tests are not exercised. CI uses
`requirements-dev.txt` on Windows/Linux and `requirements-dev-macos.txt` on macOS.

Useful checks:

```shell
python -m pytest tests/test_scientific_integrity.py -q
python -m pytest --cov --cov-report=xml
python -m pip install build
python -m build
```

These run the scientific regressions, generate a coverage report, and build
distributions in `dist/`. Do not use `.build.bat` for a local check: it also uploads
distributions. CI covers Python 3.9–3.11 on Windows, Linux and macOS; DaVis is
unavailable on macOS. No minimum coverage percentage is configured.

## Dependency policy

`setup.cfg` is the package manifest; keep duplicated bounds in `requirements.txt`
in sync. The current integration targets `h5rdmtoolbox>=2.8.3,<3.0.0`,
`ontolutils>=0.27.5,<0.28.0`, and `rdflib>=7.1,<8.0`. The default standard-name
table is bundled; its read-only validator does not require ssnolib or pivmetalib.
Keep next-major ceilings on related packages and verify deliberate upgrades.

To test a local toolbox change, install its checkout first, for example
`python -m pip install -e ../h5RDMtoolbox`, then install this project. Check the
actual imported version and path; a local pass does not certify every version in
the permitted range. `tests/test_version.py` checks consistency between installed
metadata, `setup.cfg`, `CITATION.cff`, and `codemeta.json`.

## Contribution workflow

Create a new task branch before edits and preserve existing work. Add meaningful
regressions for changed scientific behavior, run relevant checks, review explicit
staged paths, and commit on that branch. Ask before delegating to subagents.
Keep local investigation notes and generated evidence in `.work/`; it is ignored
and must not be committed or force-added. Update [usage](docs/usage.md) and
[migration notes](MIGRATION.md) when public behavior changes.
