from casacore import tables
from contextlib import contextmanager
from typing import Dict, Generator

# common casacore table handling code


def extract_table_attributes(infile: str) -> Dict[str, Dict]:
    """
    return a dictionary of table attributes created from MS keywords and column descriptions
    """
    with open_table_ro(infile) as tb_tool:
        kwd = tb_tool.getkeywords()
        attrs = dict([(kk, kwd[kk]) for kk in kwd if kk not in os.listdir(infile)])
        cols = tb_tool.colnames()
        column_descriptions = {}
        for col in cols:
            column_descriptions[col] = tb_tool.getcoldesc(col)
        attrs["column_descriptions"] = column_descriptions
        attrs["info"] = tb_tool.info()
    return attrs


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
def open_table_rw(outfile: str) -> Generator[tables.table, None, None]:
    table = tables.table(
        outfile, readonly=False, lockoptions={"option": "permanentwait"}, ack=False
    )
    try:
        yield table
    finally:
        table.close()
