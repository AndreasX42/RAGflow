import datetime as dt
from pydantic import BaseModel

from typing import Optional


class UserBase(BaseModel):
    username: str
    email: str


class UserFromDB(UserBase):
    id: int
    date_created: dt.datetime
    role: str
    is_active: bool

    class Config:
        from_attributes = True


class UpdateUserRequest(UserBase):
    password: Optional[str] = None


class CreateUserRequest(UserBase):
    password: str
