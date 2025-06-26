#!/usr/bin/env python

import sys
import importlib

from xradio.schema.export import export_schema_json_file

SCHEMA_MAP = {
    'VisibilityXds': ('xradio.measurement_set.schema', 'VisibilityXds'),
    'SpectrumXds': ('xradio.measurement_set.schema', 'SpectrumXds')
}

# Enough arguments?
if len(sys.argv) < 3 or sys.argv[1] not in SCHEMA_MAP:
    print('Usage:')
    print('  $ python export_schema.py [schema name] [file name]')
    print()
    print('Available schemas:', ', '.join(SCHEMA_MAP.keys()))
    exit(1)

# Import schema
mod_name, class_name = SCHEMA_MAP[sys.argv[1]]
mod = importlib.import_module(mod_name)
cls = getattr(mod, class_name)

# Perform export
export_schema_json_file(cls, sys.argv[2])
