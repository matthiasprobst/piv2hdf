"""Read-only standard-name/unit validation, independent of ontology object models."""
import json
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

import rdflib
from h5rdmtoolbox import get_ureg
from ontolutils.utils.qudt_units import qudt_lookup

TABLE_URL = 'https://zenodo.org/records/15297431/files/Standard_Name_Table_for_Particle_Image_Velocimetry.jsonld'
TABLE_FILE = Path(__file__).parent / 'resources/convention/piv_standard_names.jsonld'
SSNO = rdflib.Namespace('https://matthiasprobst.github.io/ssno#')
_UNIT_SYMBOLS = {str(iri): symbol for symbol, iri in qudt_lookup.items()}
_UNIT_SYMBOLS['http://qudt.org/vocab/unit/UNITLESS'] = ''


def unit_quantity(units):
    if str(units).startswith(('http://', 'https://')):
        if str(units) not in _UNIT_SYMBOLS:
            raise ValueError(f'Unknown unit IRI: {units}')
        units = _UNIT_SYMBOLS[str(units)] or ''
    return get_ureg().Quantity(1, str(units).strip())


def table_key(value):
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


@lru_cache(maxsize=16)
def load_table(key):
    """Index names and units; retain the original published metadata unchanged."""
    if key == TABLE_URL:
        text = TABLE_FILE.read_text(encoding='utf-8')
    elif key.startswith('{'):
        text = key
    elif key.startswith(('http://', 'https://')):
        with urlopen(key, timeout=15) as response:
            text = response.read().decode('utf-8')
    else:
        text = Path(key).read_text(encoding='utf-8')
    graph = rdflib.Graph().parse(data=text, format='json-ld')
    names = {}
    for node, name in graph.subject_objects(SSNO.standardName):
        unit = graph.value(node, SSNO.unit)
        if unit is None:
            raise ValueError(f'Missing units for standard name {name}')
        if str(name) in names:
            raise ValueError(f'Duplicate standard name {name}')
        names[str(name)] = str(unit)
    if not names:
        raise ValueError('No SSNO standard names found in the table')
    return names


def validate_standard_name(name, parent):
    if parent is None:
        raise ValueError('A parent dataset is required to validate a standard name')
    file_attrs = parent.file.attrs
    table = getattr(file_attrs, 'raw', file_attrs).get('standard_name_table')
    if table is None:
        raise ValueError('No standard_name_table on the file')
    names = load_table(table_key(table))
    reference = name
    # A sample arithmetic mean preserves the physical dimension of its input.
    if name not in names and name.startswith('arithmetic_mean_of_'):
        reference = name[len('arithmetic_mean_of_'):]
    if reference not in names:
        raise ValueError(f'Standard name is not defined by the table: {name}')
    units = getattr(parent.attrs, 'raw', parent.attrs).get('units')
    if units is None:
        raise ValueError('Dataset units are required')
    if unit_quantity(units).dimensionality != unit_quantity(names[reference]).dimensionality:
        raise ValueError(f'Incompatible units {units} for {name}: expected {names[reference]}')
    return name
