from collections.abc import Generator
from contextlib import contextmanager

from xradio._utils._casacore.backend import tables


@contextmanager
def open_table_ro(infile: str) -> Generator[tables.table, None, None]:
    table = tables.table(
        infile, readonly=True, lockoptions={"option": "usernoread"}, ack=False
    )
    try:
        yield table
    finally:
        table.close()


@contextmanager
def open_query(table: tables.table, query: str) -> Generator[tables.table, None, None]:
    if hasattr(tables, "taql"):
        ttq = tables.taql(query)
    else:
        ttq = table.taql(query)
    try:
        yield ttq
    finally:
        ttq.close()
