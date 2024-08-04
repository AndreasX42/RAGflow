import logging

from ragflow.api.services import auth_service
from ragflow.api.schemas import UserFromDB, CreateUserRequest, UpdateUserRequest
from ragflow.api.models import User
from ragflow.api.database import Session

from fastapi import HTTPException

logger = logging.getLogger(__name__)


async def create_user(user_data: CreateUserRequest, db_session: Session) -> UserFromDB:
    user_data_dict = user_data.model_dump()

    user_data_dict["hashed_password"] = auth_service.get_password_hash(
        user_data_dict.pop("password")
    )

    # Check if a user with the same username or email already exists
    existing_user = (
        db_session.query(User)
        .filter(
            (User.username == user_data_dict.get("username"))
            | (User.email == user_data_dict.get("email"))
        )
        .first()
    )

    if existing_user:
        # Determine which attribute (username or email) already exists in db
        duplicate_field = (
            "username"
            if existing_user.username == user_data_dict.get("username")
            else "email"
        )
        raise HTTPException(
            status_code=400,
            detail=f"User with provided {duplicate_field} already exists!",
        )

    user = User(**user_data_dict)

    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    return UserFromDB.model_validate(user)


async def get_all_users(db_session: Session) -> list[UserFromDB]:
    users = db_session.query(User).all()
    return list(map(UserFromDB.model_validate, users))


async def get_user_by_id(user_id: int, db_session: Session) -> User:
    user = db_session.query(User).filter(User.id == user_id).first()

    return user


async def get_user_by_name(username: str, db_session: Session) -> User:
    user = db_session.query(User).filter(User.username == username).first()

    return user


async def delete_user(user: User, db_session: Session) -> None:
    db_session.delete(user)
    db_session.commit()


async def update_user(
    user: User,
    user_data: UpdateUserRequest,
    db_session: Session,
) -> UserFromDB:
    user.username = user_data.username
    user.email = user_data.email

    if user_data.password:
        user.hashed_password = auth_service.get_password_hash(user_data.password)

    db_session.commit()
    db_session.refresh(user)

    return UserFromDB.model_validate(user)
