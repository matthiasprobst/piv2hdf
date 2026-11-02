# Runnable examples

These notebooks use sample files in `tests/resources/`, which require an editable
source checkout. From the repository root, in a Python 3.9–3.11 environment:

```shell
python -m pip install -e .
python -m pip install jupyterlab
jupyter lab examples
```

Choose this environment's Python kernel and run cells from top to bottom. For
DaVis on Windows/Linux, first install `python -m pip install -e ".[davis]"`.

| Notebook | Learn to |
| --- | --- |
| [PIVview snapshot](01_snapshot_pivview.ipynb) | Convert one result, validate metadata and inspect units/validity. |
| [PIVview plane](02_plane_pivview.ipynb) | Define recording times and inspect sample means/counts. |
| [DaVis snapshot](01_snapshot_davis.ipynb) | Preserve vendor units and inspect raw/calibrated components. |
| [Custom operation](01_snapshot_advanced.ipynb) | Add an in-plane speed field through a `udo` callback. |

Notebooks write HDF5 files under `outputs/` beside the notebook; reruns replace
those outputs. They do not modify sample inputs. Recording dates/rates and creator
names are illustrative, not recovered acquisition metadata. For your measurements,
use actual paths and metadata from the [usage guide](../docs/usage.md).
