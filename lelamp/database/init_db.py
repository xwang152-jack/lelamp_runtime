#!/usr/bin/env python3
"""
Database initialization script.

This script initializes the LeLamp database by creating all tables.
It can be run standalone or imported as a module.

Usage:
    python -m lelamp.database.init_db
"""
import sys
import logging

from lelamp.database.base import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    """
    Initialize the database.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        logger.info("Initializing LeLamp database...")
        init_db()
        logger.info("Database initialized successfully!")
        return 0
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
