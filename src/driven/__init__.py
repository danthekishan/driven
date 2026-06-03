import logging
import sys


def setup_logging():
    """Configure this once at the startup of your application."""
    logging.basicConfig(
        level=logging.INFO,  # Minimum level to display (DEBUG, INFO, WARNING, ERROR)
        format="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)  # Outputs to console
            # logging.FileHandler("actor_system.log") # Uncomment to also write to a file
        ],
    )


setup_logging()
