# src/cli/train/logger_config.py
import logging
import sys

# ANSI Colours (optional)
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"

def color_text(text, color):
    return f"{color}{text}{RESET}"

# set the logging config level
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout
)

# set the logger
logger = logging.getLogger(__name__)
