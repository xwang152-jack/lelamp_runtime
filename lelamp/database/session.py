"""
Database session management utilities.

Provides session management functions for database operations
with proper error handling and cleanup.
"""
from typing import Generator

from sqlalchemy import exc as sqlalchemy_exc
from sqlalchemy.orm import Session

from lelamp.database.base import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    Get database session with proper cleanup.

    This is a dependency function for FastAPI that yields a database session
    and ensures it's properly closed after use.

    Yields:
        Session: SQLAlchemy database session

    Raises:
        sqlalchemy_exc.SQLAlchemyError: On database errors

    Example:
        @app.get("/conversations/")
        def read_conversations(db: Session = Depends(get_db)):
            return db.query(Conversation).all()
    """
    db = SessionLocal()
    try:
        yield db
    except sqlalchemy_exc.SQLAlchemyError as e:
        db.rollback()
        raise e
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Get a database session for manual management.

    Use this when you need to manage the session lifecycle yourself
    (e.g., in background tasks, scripts, or non-FastAPI code).

    Returns:
        Session: SQLAlchemy database session

    Example:
        db = get_db_session()
        try:
            result = db.query(Conversation).all()
        finally:
            db.close()
    """
    return SessionLocal()
