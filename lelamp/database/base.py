"""
Database base configuration and engine setup.

This module provides the SQLAlchemy Base, database engine, and session management
for the LeLamp runtime persistence layer.
"""
import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base

# Database URL from environment variable or default to SQLite
DATABASE_URL = os.getenv(
    "LELAMP_DATABASE_URL", "sqlite:///./lelamp.db"
)

# Create database engine
# For SQLite, we need to check_same_thread=False for FastAPI compatibility
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,  # Set to True for SQL query logging in development
    )
else:
    engine = create_engine(
        DATABASE_URL,
        echo=False,
    )

# Create SessionLocal class for database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for ORM models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function for FastAPI to get database sessions.

    Yields:
        Session: SQLAlchemy database session

    Example:
        @app.get("/conversations/")
        def read_conversations(db: Session = Depends(get_db)):
            return db.query(Conversation).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database tables.

    Creates all tables defined in ORM models. Use this for
    initial database setup or migrations.
    """
    Base.metadata.create_all(bind=engine)


def drop_db() -> None:
    """
    Drop all database tables.

    WARNING: This will delete all data. Use only for testing or development.
    """
    Base.metadata.drop_all(bind=engine)
