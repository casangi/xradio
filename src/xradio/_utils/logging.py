from types import ModuleType
import logging

try:
    import toolviper.utils.logger as _logger  # noqa: F841
except ImportError:
    _logger = logging


def xradio_logger() -> ModuleType:
    """Returns the toolviper logging module if available,
    otherwise the standard Python logging module"""
    return _logger
