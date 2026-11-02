"""Pydantic hooks for the PIV convention; no global table or network default."""
from typing import Dict, Union

from pydantic.functional_validators import WrapValidator
from typing_extensions import Annotated

from piv2hdf.standard_names import load_table, table_key, unit_quantity, validate_standard_name


def __validate_standard_name_table(value, handler, info):
    value = handler(value)
    load_table(table_key(value))
    return value


def equal_base_units(u1, u2):
    return unit_quantity(u1).dimensionality == unit_quantity(u2).dimensionality


def __validate_standard_name(value, handler, info):
    value = handler(value)
    parent = (info.context or {}).get('parent')
    return validate_standard_name(value, parent)


standardNameTableType = Annotated[Union[str, Dict], WrapValidator(__validate_standard_name_table)]
standardNameType = Annotated[str, WrapValidator(__validate_standard_name)]
