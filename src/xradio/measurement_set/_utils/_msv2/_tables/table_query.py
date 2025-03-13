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


class TableManager:

    def __init__(
        self,
        infile: str,
        taql_where: str = "",
    ):
        self.infile = infile
        self.taql_where = taql_where
        self.taql_query = taql_where.replace("where ", "")

    def get_table(self):
        # Performance note:
        # table.query("(DATA_DESC_ID = 0)") is slightly faster than
        # tables.taql("select * from $table (DATA_DESC_ID = 0)")
        with tables.table(
            self.infile, readonly=True, lockoptions={"option": "usernoread"}, ack=False
        ) as mtable:
            query = f"select * from $mtable {self.taql_where}"
            return tables.taql(query)
