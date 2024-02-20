import os
from graphviper.utils.logger import setup_logger

if not os.getenv("VIPER_LOGGER_NAME"):
    os.environ["VIPER_LOGGER_NAME"] = "xradio"
    setup_logger(
        logger_name="xradio",
        log_to_term=True,
        log_to_file=False,
        log_file="xradio-logfile",
        log_level="INFO",
    )
