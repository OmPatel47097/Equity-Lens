import logging
from pathlib import Path

class LoggerManager:
    @staticmethod
    def get_logger(name: str, log_file: str = "app.log", level: int = logging.INFO):
        # Create a logger
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Create file handler which logs even debug messages
        log_path = Path(__file__).resolve().parent.parent / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        log_file_path = log_path / log_file

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(level)

        # Create console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)

        # Create a formatter and set it for the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        if not logger.handlers:  # To avoid adding handlers multiple times
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger
