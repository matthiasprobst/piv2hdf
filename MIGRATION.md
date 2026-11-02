# Migrating to 2.0.0a3

Version 2.0.0a3 continues the 2.0 prerelease series. It intentionally changes
scientific output and configuration behavior. Existing HDF5 files remain readable
with h5py/h5rdmtoolbox; no files are modified automatically. Reconvert original
inputs to a new destination to obtain corrected outputs and compare them before
replacing archived results.

## Values and units

Older plane z overrides could combine a new value with the old unit. Verify these
files using acquisition records; their stored metadata alone cannot recover the
intended position. New conversions preserve the override quantity correctly.

Davis coordinates retain their vendor calibration units, including pixel-only
coordinates. Components apply their scale slope/offset once and retain raw source
values and calibration provenance. Pixel displacement is not relabeled as a
physical velocity. The metadata key is consistently `units` rather than `unit`.

OpenPIV defaults describe the supported raw-coordinate format: positions in
pixels, `scaling_factor` in pixel/m, and saved velocities in m/s. Set
`coordinate_units`, `scaling_factor_units`, and `velocity_units` on `OpenPIVFile`
when the input contract differs; these options also pass through `from_folder`.
For already-scaled positions, supply their physical coordinate unit; they are not
scaled again. Fractional pixel locations are preserved. Flag aliases normalize to
`piv_flags`. Unknown grid/column layouts must be corrected at the input boundary,
not guessed from array size.

## Validity and statistics

PIVview flags now affect values. Strict accepts ACTIVE only; sepeaks additionally
accepts REPLACED; slack also accepts FILTERED, INTERPOLATED and MANUALEDIT.
MASKED, DISABLED, NORESULT, unknown bits and inactive vectors are rejected in
every mode. Combined flags must contain only accepted bits. Excluded values
become NaN. `raw_*` fields and flags retain the input values; `valid` describes
the processed field. Interpolated vectors get ACTIVE|INTERPOLATED and original
flags remain in `raw_piv_flags`. No extrapolation is performed outside the valid
points' convex hull.

Means exclude invalid and nonfinite samples and expose `mean_*_valid_count`.
Sample arithmetic means remain the default even with irregular sampling.
`averaging='duration'` uses trapezoidal integration over valid adjacent endpoints;
it neither imputes gaps nor supplies a measurement uncertainty estimate.
An existing mean with a different or unknown averaging method is rejected instead
of silently reused. To select duration weighting during conversion, pass a callable:

```python
from functools import partial
from piv2hdf import set_config
from piv2hdf.postproc import compute_time_averages

set_config(postproc=[partial(compute_time_averages, averaging='duration')])
```

## Merging and metadata

Unequal plane lengths require explicit `trim_time_vectors=True`. Merged files
record dropped counts, alignment policy, original relative times per plane and
separate absolute acquisition timestamps. Approximate relative-time alignment
uses the mean within the requested tolerances and emits a warning. It does not
interpolate observations or assert simultaneity. Coordinate units are normalized
for compatibility checks; duplicate physical z positions are rejected.

New outputs carry `piv2hdf_schema_version='2.0'`. Standard names are assigned only
when defined by the bundled published table (or the documented arithmetic-mean
derivation); descriptive `long_name` and flag metadata describe other fields.
Find flag datasets by name, not by the former undefined `piv_flags` standard name.
The original published JSON-LD table remains unchanged, including its historical
version field; validation indexes the names and units without reparsing unrelated
metadata into legacy ontology model classes.
Outputs also record source-file paths, piv2hdf/h5rdmtoolbox versions and conversion
options. Adapter-specific unit and validity policies are recorded with the source
metadata; merged files retain each plane's metadata in its parameter group parent.
Custom postprocessor identities are recorded, but their code and external state
must still be archived separately to reproduce an experiment.

Public snapshot/plane/multiplane converters stage the complete output in the
destination directory, replacing the target only after callbacks and validation
succeed. Failed conversions preserve an existing output and remove their staging
file. Callback file handles therefore refer to a temporary filename until success.

`get_config()` and `get_pivattrs()` return copies. Per-call metadata is local to
its conversion. Context setters restore prior state, including nested settings.
Conversions snapshot their options and restore the caller's active toolbox
convention. Native h5rdmtoolbox convention changes remain process-global; parallel
conversions requiring different conventions should use separate processes.
The packaged convention is initialized on demand (`piv2hdf.cv` or
`set_config(convention='planar_piv')`). File logging is opt-in via
`PIV2HDF_LOG_DIR`; `PIV2HDF_USER_DIR` redirects contacts and temporary data.

## Dependencies and verification

Install in a Python 3.9-3.11 environment. h5rdmtoolbox must be >=2.8.3,<3.0.0.
SSNO model parsing and pivmetalib were removed from the runtime dependency path;
the read-only validator directly uses bounded rdflib/ontolutils dependencies.
Run `python -m pytest` for conversion and independent scientific reference cases.
