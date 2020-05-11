import coloredlogs
import logging

# set up logging
logger = logging.getLogger(__name__)
coloredlogs.install(
    level='DEBUG',
    logger=logger,
    fmt='[%(levelname)s] %(message)s',
    level_styles=coloredlogs.parse_encoded_styles('spam=22;debug=28;verbose=34;info=226;notice=220;warning=202;success=118,bold;error=124;critical=background=red'))


def get_logger():
    return logger
