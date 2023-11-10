from typing import Generator
from contextlib import contextmanager

from casacore import tables


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
    ttq = tables.taql(query)
    try:
        yield ttq
    finally:
        ttq.close()
