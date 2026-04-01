"""Logging utility for MatAgent

Provides unified logging configuration that:
- Saves logs to file AND displays in console
- Works regardless of how the script is run (run.sh or python main.py)
- Supports different log levels (DEBUG, INFO, WARNING, ERROR)
- Creates timestamped log files
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class LoggerSetup:
    """
    Centralized logger configuration for MatAgent

    Usage:
        logger = LoggerSetup.setup_logger(
            output_dir="data/results/run_20251128_120000",
            level=logging.INFO
        )
    """

    @staticmethod
    def setup_logger(
        output_dir: Optional[str] = None,
        level: int = logging.INFO,
        log_filename: str = "run.log",
        verbose: bool = True
    ) -> logging.Logger:
        """
        Setup unified logging configuration

        Args:
            output_dir: Directory to save log file (if None, console only)
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_filename: Name of log file
            verbose: If True, show detailed format; if False, simplified

        Returns:
            Configured logger
        """
        # Get root logger
        logger = logging.getLogger()
        logger.setLevel(level)

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

        # Define format
        if verbose:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            log_format = '%(asctime)s - %(levelname)s - %(message)s'

        formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

        # 1. Console handler (always present)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. File handler (if output_dir specified)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            log_file = output_path / log_filename

            # Create file handler
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            logger.info(f"Logging to file: {log_file}")

        return logger

    @staticmethod
    def setup_run_directory(base_dir: str = "data/results", run_name: Optional[str] = None) -> Path:
        """
        Create a timestamped run directory

        Args:
            base_dir: Base directory for results
            run_name: Optional custom name (default: run_YYYYMMDD_HHMMSS)

        Returns:
            Path to run directory
        """
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"

        run_dir = Path(base_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        return run_dir

    @staticmethod
    def log_system_info(logger: logging.Logger):
        """Log system and environment information"""
        import platform
        import sys
        import os

        logger.info("="*60)
        logger.info("SYSTEM INFORMATION")
        logger.info("="*60)
        logger.info(f"Python version: {sys.version.split()[0]}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Working directory: {os.getcwd()}")

        # Check conda environment
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'N/A')
        logger.info(f"Conda environment: {conda_env}")

        # Log important packages
        try:
            import numpy as np
            logger.info(f"NumPy version: {np.__version__}")
        except ImportError:
            pass

        try:
            import chgnet
            logger.info(f"CHGNet available: Yes")
        except ImportError:
            logger.warning("CHGNet not available")

        logger.info("="*60)


class SummaryLogger:
    """
    Helper class to collect and log summary statistics

    Usage:
        summary = SummaryLogger()
        summary.add("Structures generated", 45)
        summary.add("Novel materials", 38)
        summary.log(logger)
    """

    def __init__(self):
        self.stats = []

    def add(self, key: str, value, unit: str = ""):
        """Add a statistic"""
        self.stats.append((key, value, unit))

    def log(self, logger: logging.Logger, title: str = "SUMMARY"):
        """Log all statistics"""
        logger.info("\n" + "="*60)
        logger.info(title)
        logger.info("="*60)

        for key, value, unit in self.stats:
            if unit:
                logger.info(f"{key}: {value} {unit}")
            else:
                logger.info(f"{key}: {value}")

        logger.info("="*60)


# Convenience function
def setup_experiment_logging(output_dir: str, level: int = logging.INFO) -> logging.Logger:
    """
    Quick setup for experiment logging

    Args:
        output_dir: Directory to save logs and results
        level: Logging level

    Returns:
        Configured logger
    """
    logger = LoggerSetup.setup_logger(output_dir=output_dir, level=level)
    LoggerSetup.log_system_info(logger)
    return logger
