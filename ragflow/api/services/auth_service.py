from typing import Union

from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from starlette import status
import sqlalchemy.orm as orm
from jose import JWTError, jwt
from passlib.context import CryptContext

from datetime import datetime, timedelta
import os

from ragflow.api.schemas import UserFromDB
from ragflow.api.services import user_service
from ragflow.api.services import get_db

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
HASH_ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


async def authenticate_user(
    username: str, password: str, db_session: orm.Session
) -> Union[UserFromDB, bool]:
    user = await user_service.get_user_by_name(username=username, db_session=db_session)

    if user is None:
        return False

    if not verify_password(password, user.hashed_password):
        return False

    return user


def create_access_token(
    subject: dict, expires_delta: timedelta = None
) -> tuple[str, datetime]:
    to_encode = subject.copy()

    if expires_delta is not None:
        exp = datetime.utcnow() + expires_delta

    else:
        exp = datetime.utcnow() + timedelta(minutes=15)

    to_encode |= {"exp": exp}
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, HASH_ALGORITHM)

    return encoded_jwt, exp


async def get_current_user(
    request: Request,
    db_session: orm.Session = Depends(get_db),
) -> UserFromDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = request.cookies.get("access_token")

        if token is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not authenticated",
            )

        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[HASH_ALGORITHM])
        username = payload.get("sub")
        user_id = payload.get("user_id")
        user_role = payload.get("user_role")

        if username is None or user_id is None or user_role is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    user = await user_service.get_user_by_id(user_id=user_id, db_session=db_session)

    if user is None:
        raise credentials_exception

    return UserFromDB.model_validate(user)


async def get_current_active_user(
    current_user: UserFromDB = Depends(get_current_user),
) -> UserFromDB:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    return current_user
