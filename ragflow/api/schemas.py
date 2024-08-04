import datetime as dt
from typing import Annotated
from pydantic import BaseModel, EmailStr, StringConstraints

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
    username: Annotated[str, StringConstraints(min_length=4, max_length=64)]
    password: Annotated[str, StringConstraints(min_length=8, max_length=128)]
    email: EmailStr
