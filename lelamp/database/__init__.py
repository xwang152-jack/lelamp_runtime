"""
LeLamp database persistence layer.

This module provides database models, CRUD operations, and session management
for the LeLamp runtime system using SQLAlchemy ORM.

Example usage:
    from lelamp.database import Base, engine, get_db
    from lelamp.database.models import Conversation, OperationLog, DeviceState, UserSettings
    from lelamp.database import crud

    # Initialize database
    Base.metadata.create_all(bind=engine)

    # Use in FastAPI
    @app.get("/conversations/")
    def read_conversations(db: Session = Depends(get_db)):
        return crud.get_conversations_by_lamp_id(db, lamp_id="lelamp", limit=10)
"""

from lelamp.database.base import Base, engine, get_db, init_db, drop_db
from lelamp.database.models import (
    Conversation,
    OperationLog,
    DeviceState,
    UserSettings,
)
from lelamp.database.session import get_db, get_db_session
from lelamp.database import crud

__all__ = [
    # Base and engine
    "Base",
    "engine",
    "get_db",
    "get_db_session",
    "init_db",
    "drop_db",
    # Models
    "Conversation",
    "OperationLog",
    "DeviceState",
    "UserSettings",
    # CRUD module
    "crud",
]
