import datetime as dt
from sqlalchemy import Column, Integer, String, DateTime, Boolean

from ragflow.api.database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True, unique=True)
    hashed_password = Column(String, index=False, unique=False)
    email = Column(String, index=True, unique=True)
    date_created = Column(DateTime, default=dt.datetime.now(dt.UTC))
    is_active = Column(Boolean, default=True)
    role = Column(String, default="user", index=True)
