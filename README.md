# piv2hdf

Convert processed particle image velocimetry (PIV) results to HDF5 for analysis,
sharing and archiving. Inputs can come from PIVview, the supported OpenPIV text
format, or LaVision DaVis. This package converts results; it does not process
image pairs or establish the accuracy of an experiment.

## Install

Use **Python 3.9–3.11**. From this checkout, install the current **2.0.0a3** alpha:

```shell
python -m pip install -e .
```

For DaVis `.vc7` files on Windows/Linux, use `python -m pip install -e ".[davis]"`.
The DaVis reader is unavailable on macOS. Runtime dependencies include
**h5rdmtoolbox >=2.8.3,<3.0.0**. For a published release instead of this checkout,
use `python -m pip install piv2hdf`; its behavior may differ from this alpha.

## Convert a PIVview time series

Put the `.nc` snapshots and their `.par` parameter file in one folder. The example
uses planar **2D2C** data; use `PIVViewStereoNcFile` with its `.cfg` file for
stereo **2D3C** data. Replace the paths, recording start, sampling rate and
creator below with your experiment's values.

```python
from datetime import datetime
from piv2hdf import PIVPlane, set_config
from piv2hdf.pivview import PIVViewNcFile
from piv2hdf.pivview.user_operations import add_standard_name_operation

plane = PIVPlane.from_folder(
    "data/plane01",
    time_info=(datetime(2026, 1, 1, 12), 100.0),  # start and result sampling rate (Hz)
    pivfile=PIVViewNcFile,
    parameter="data/plane01/piv_parameters.par",
    udo=add_standard_name_operation,
)

metadata = {"creator": {"name": "Your name"}, "piv_medium": "air"}
with set_config(convention="planar_piv"):
    output = plane.to_hdf(
        hdf_filename="results/plane01.hdf",
        piv_attributes=metadata,
    )
print(output)
```

Files are sorted by filename; use zero-padded numbers to preserve acquisition
order. The default search selects filenames ending in a digit before `.nc`.
The sampling rate describes successive results, **not the laser pulse delay**.

The `udo` hook assigns standard names, enabling automatic velocity sample means
and their valid-sample counts. Explicit convention validation checks metadata;
it is disabled by default. The bundled standard-name table works offline.

## Inspect the result

```python
import h5py

with h5py.File(output, "r") as data:
    print(data["u"].shape)          # (number of results, ny, nx)
    print(data["u"].attrs["units"])
    print(data["mean_u"][...])     # sample mean over valid, finite observations
```

Coordinates retain units; invalid vectors become NaN and raw fields are retained
where supported. Check units, validity and acquisition times before analysis.
Successful conversion replaces an existing destination; a failed public conversion
preserves it. Use a new filename to retain an earlier result.

## Choose the next step

- [Usage guide](docs/usage.md): single snapshots, formats, timing, masking,
  multiplane data, statistics and troubleshooting.
- [Runnable notebooks](examples/README.md): bundled-data walkthroughs.
- [Migration to 2.0.0a3](MIGRATION.md): changed results, metadata and APIs.
- [Development and dependencies](DEPENDENCIES.md): installation, tests and bounds.

A snapshot contains one result; a plane contains a time series at one position;
a multiplane file combines planes at different z positions. Separately acquired
planes do not constitute a simultaneous volume. Numerical tests check software
behavior, not calibration quality, convergence or measurement uncertainty.

[Citation metadata](CITATION.cff) · [MIT license](LICENSE)
