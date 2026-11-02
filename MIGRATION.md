# Migrating to 2.0.0a3

This alpha corrects scientific values and changes some output fields and APIs.
Existing HDF5 files remain readable with h5py/h5rdmtoolbox. Reconvert original
inputs **to a new destination** and compare outputs before replacing an archive.
There is no automatic migration: an old file's metadata may not reveal the
intended calibration. Current workflows are in the [usage guide](docs/usage.md).

## Check earlier results

| Earlier behavior | 2.0.0a3 behavior / action |
| --- | --- |
| Plane z overrides could retain the old unit | Values and units change together. Verify older positions against acquisition records. |
| PIVview masking modes left invalid vectors in the data | Modes now mask values; means can therefore change. Inspect `valid` and `raw_*`. |
| DaVis coordinates were labeled as metres regardless of source | Vendor units are retained; component slope/offset is applied once, with raw values retained. Pixel displacement remains displacement. |
| Per-call experiment attributes could leak into later conversions | Metadata applies only to that conversion. Supply it on every call or use scoped defaults. |
| Unequal plane lengths were trimmed by default | Pass `trim_time_vectors=True` only when discarding trailing samples is intended. |

Changing grids, invalid sampling frequencies, non-increasing timestamps and
duplicate physical z positions are rejected. Multiplane comparisons normalize
coordinate units. Approximate time alignment retains original per-plane times
for the samples kept, records dropped counts, and emits a warning. It does not
resample velocities or imply simultaneous acquisition.

## Update code and readers

- Use `parameter=...` and `udo=...`; use `PIVPlane.from_folder(...)` for a series.
- OpenPIV flag aliases become `piv_flags`. Positions default to pixels, scaling
  to pixel/m, and saved velocities to m/s. Declare alternative input units; see
  [format selection](docs/usage.md#select-the-input-format).
- Find flags by dataset name. Fields outside the bundled standard-name table use
  descriptive metadata rather than undefined standard names. Gradients use the
  table's `x_derivative_of_x_velocity` naming pattern.
- Read units from `units`, including for DaVis. New files include
  `piv2hdf_schema_version="2.0"`, source paths, software versions and options.
  Keep inputs and custom callback code separately; this metadata is not a full
  archive of the experiment.

Velocity means are sample arithmetic means by default. They exclude invalid and
nonfinite observations and add `mean_*_valid_count`. Duration weighting is an
explicit alternative; it excludes intervals with missing endpoints. Existing
means with a different or unknown method are rejected, not silently reused.
See the [statistics example](docs/usage.md#control-validity-and-statistics).

## Configuration and validation

`get_config()` and `get_pivattrs()` return copies. Change defaults with their
setters; context managers restore prior settings. Conversions snapshot options
and restore the caller's active toolbox convention. Use separate processes for
parallel work requiring different conventions, which remain process-global.

The packaged convention loads on demand through `piv2hdf.cv` or
`set_config(convention="planar_piv")`. Validation checks nested datasets and
raises by default; `raise_layout_validation_error=False` requests warnings.
The unchanged default table is bundled for offline validation, without the legacy
SSNO object parser. Runtime dependencies require Python 3.9–3.11 and
h5rdmtoolbox >=2.8.3,<3.0.0.

Public converters replace the destination only after writing, callbacks and
validation succeed. Callback handles refer to a staging file until success;
do not retain them or their temporary paths. Failed conversions preserve existing
outputs. `PIV2HDF_USER_DIR` redirects contacts and temporary files; file logging is
opt-in via `PIV2HDF_LOG_DIR`.
