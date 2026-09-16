#!/usr/bin/env python

import sys

from xradio.schema.export import import_schema_json_file

print(import_schema_json_file(sys.argv[1]))
