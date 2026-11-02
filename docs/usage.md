# Using piv2hdf for scientific data

Start with the [README quickstart](../README.md). Examples below use your own
processed results; substitute the paths and acquisition metadata. For bundled
inputs, run the [notebooks](../examples/README.md) from an editable source checkout.

## Select the input format

| Input | Reader | Parameter input |
| --- | --- | --- |
| PIVview planar, 2D2C | `piv2hdf.pivview.PIVViewNcFile` | `.par` file |
| PIVview stereo, 2D3C | `piv2hdf.pivview.PIVViewStereoNcFile` | `.cfg` file |
| Supported OpenPIV tab-separated text | `piv2hdf.openpiv.OpenPIVFile` | `.par` file |
| DaVis `.vc7`, Windows/Linux | `piv2hdf.davis.VC7File` | Read from the `.vc7` file |

Use `parameter=...`, not `parameter_filename`. PIVview/OpenPIV readers can find a
single parameter file beside the input; pass its path explicitly if several exist.
OpenPIV input must form a complete rectangular grid, ordered by y then x. General
OpenPIV text exports are not interchangeable with this supported column layout.
Coordinate/velocity columns are `# x`, `y`, `u`, `v` (optionally `w`). Source flag
codes 0/1/2 mean active/disabled/interpolated; other codes are rejected.

For already-scaled OpenPIV positions and velocities in millimetres:

```python
from piv2hdf.openpiv import OpenPIVFile
from piv2hdf.openpiv.user_operations import add_standard_name_operation as add_openpiv_standard_names

def add_openpiv_names(data):
    add_openpiv_standard_names(data)
    data["reltime"].attrs["standard_name"] = "relative_time"

reader = OpenPIVFile(
    "data/openpiv/vectors.txt", parameter="data/openpiv/openpiv.par",
    coordinate_units="mm", velocity_units="mm/s", udo=add_openpiv_names,
)
```

Defaults are pixel positions, a scaling factor in pixel/m, and **saved physical
velocities in m/s**. For raw pixel positions calibrated in pixel/mm, set
`scaling_factor_units="pixel/mm"`. Physical positions are not scaled again.
For physical-coordinate inputs, `ix`/`iy` are grid indices.
Velocities must already be physical; the reader does not infer them from pulse
separation. These unit options also pass through `PIVPlane.from_folder`.
For OpenPIV planes, supply `z=...` on `to_hdf`; the text reader has no z coordinate.
DaVis retains vendor component names and units: pixel displacement is not velocity.
Multiplane merging currently expects a `u` component; DaVis names such as `U0`
are not mapped automatically. Use snapshot/plane conversion for those inputs.

## Convert one snapshot

```python
from datetime import datetime
from piv2hdf import PIVSnapshot
from piv2hdf.pivview import PIVViewNcFile
from piv2hdf.pivview.user_operations import add_standard_name_operation

reader = PIVViewNcFile("data/plane01/E00A1.nc", udo=add_standard_name_operation)
snapshot = PIVSnapshot(reader, recording_dtime=datetime(2026, 1, 1, 12))
snapshot_output = snapshot.to_hdf(
    hdf_filename="results/snapshot.hdf", piv_attributes={"creator": {"name": "Your name"}},
)
```

`recording_dtime` is optional. Without it, no absolute `time` dataset is written;
relative time is zero. A single snapshot has no time-series mean. Use
`z="5 mm"` on `to_hdf` to override a position; a numeric z is interpreted in metres.
Keep the source position when its calibration is already correct.

## Define time and combine planes

`PIVPlane` needs at least two snapshots. `time_info` accepts `(start_datetime,
frequency_hz)` or a list of recording datetimes in file order. Frequencies must
be finite and positive; timestamps must increase. Supply one timestamp per file:
too few are rejected; extras are trimmed with a warning. Within a plane, grids,
variable sets, shapes and units must match; no regridding is performed.

```python
from datetime import datetime
from piv2hdf import PIVMultiPlane
from piv2hdf.pivview import PIVViewNcFile
from piv2hdf.pivview.user_operations import add_standard_name_operation

case = PIVMultiPlane.from_folders(
    ["data/plane01", "data/plane02"],
    time_infos=[(datetime(2026, 1, 1, 12), 100.0),
                (datetime(2026, 1, 1, 13), 100.0)],
    pivfile=PIVViewNcFile, udo=add_standard_name_operation,
)
case_output = case.to_hdf(
    piv_attributes={"creator": {"name": "Your name"}},
    hdf_filename="results/multiplane.hdf", z=["0 mm", "5 mm"],
)
```

Use the actual plane positions and acquisition times. Merging checks physical x/y
coordinates and distinct z positions. Unequal lengths require explicit
`trim_time_vectors=True`; losses appear in `dropped_snapshot_count`. Relative
times must agree within `atol=1e-6` seconds and `rtol=1e-3` (the usual combined
absolute/relative tolerance). Approximately aligned times are averaged with a
warning, without resampling vectors. `relative_time_per_plane` and `time` retain
each plane's times for the retained samples. These planes need not be simultaneous.

## Control validity and statistics

PIVview options use a separate module dictionary and persist until changed:

```python
from piv2hdf import pivview
pivview.config.update(masking="strict", interpolation=False)
```

| Mode | Accepted flags, alone or combined |
| --- | --- |
| `strict` | ACTIVE (1) |
| `sepeaks` | ACTIVE, REPLACED (64) |
| `slack` (default) | ACTIVE, FILTERED (16), INTERPOLATED (32), REPLACED, MANUALEDIT (128) |

INACTIVE (0), MASKED (2), NORESULT (4), DISABLED (8) and unknown bits are rejected.
PIVview/OpenPIV processed floating fields are masked; `valid` is 0 or 1.
`raw_*` retains unmasked fields. PIVview `raw_piv_flags` preserves source bits;
OpenPIV flags are translated to the common scheme before masking.
PIVview interpolation is off by default. When enabled, linear interpolation uses
valid vectors inside their convex hull; boundary holes may remain NaN. Original
flags are retained, and successfully filled vectors get ACTIVE|INTERPOLATED.

Automatic velocity means require velocity standard names and one attached time
scale with `standard_name="relative_time"`. The PIVview hook in the
quickstart supplies these. The OpenPIV hook labels velocities but does not label
`reltime`; the `add_openpiv_names` callback above supplies that extra label.
Means use finite, flag-valid observations and report `mean_*_valid_count`.
The default is a **sample arithmetic mean**, including for irregular sampling.
For duration weighting, set the postprocessor before converting:

```python
from functools import partial
from piv2hdf import set_config
from piv2hdf.postproc import compute_time_averages

with set_config(postproc=[partial(compute_time_averages, averaging="duration")]):
    duration_output = plane.to_hdf(
        hdf_filename="results/duration_mean.hdf", piv_attributes=metadata,
    )
```

Here `plane` and `metadata` come from the quickstart. Duration weighting integrates
linearly between adjacent valid endpoints and excludes gaps. Existing means with
a different/unknown method are rejected. Use `postproc=[]` to disable means.
Neither sample counts nor interpolation supplies an uncertainty estimate.

Stereo PIVview reads compute `dwdx` and `dwdy` from `w` on the regular input grid.
`compute_dwdz=True` also derives `dwdz=-(dudx+dvdy)` when the stereo input provides
those in-plane gradients. This assumes incompressible flow; establish that
assumption for the experiment before using the derived field.

## Read and check an output

Default velocity shapes are `(y, x)` for snapshots, `(reltime, y, x)` for planes,
and `(z, reltime, y, x)` for multiplane data. Coordinates carry `units` attributes.
With h5py, read arrays as `data["u"][...]`; with h5rdmtoolbox, `data["u"][()]`
returns an xarray DataArray with attached coordinates. Read absolute timestamps
from `time`; do not infer simultaneity from shared relative-time indices.

Outputs include `piv_parameters`, schema/software versions, source paths and
conversion options. Multiplane files keep per-plane metadata in `plane0`,
`plane1`, etc. Archive inputs and custom callback code separately.
`get_config()` returns a copy; use `set_config(...)`, preferably as a context,
for compression, coordinate dtypes, postprocessing and convention validation.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| No snapshots found | Folder, reader suffix and `prefix_pattern`; use `"*"` for names not ending in digits. |
| Several parameter files found | Supply `parameter=...` explicitly. |
| No `mean_u` | Supply the format's standard-name hook and at least two snapshots. |
| Metadata validation fails | With `convention="planar_piv"`, supply `creator={"name": ...}`; inspect the reported dataset/unit. |
| Planes cannot merge | Coordinate units/grids, duplicate z, lengths and timing; normalize inputs or keep separate files. |
| Unexpected NaNs | Inspect flags and `valid`; interpolation cannot extrapolate or recover all-invalid fields. |

The default table validates offline; custom remote tables may need network access.
Validation checks metadata consistency, not the physical validity of an experiment.
