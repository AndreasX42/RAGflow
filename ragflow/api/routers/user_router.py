from fastapi import APIRouter, Depends, HTTPException
import sqlalchemy.orm as orm
from starlette import status

from ragflow.api.services import get_db
import ragflow.api.services.user_service as user_service
import ragflow.api.services.auth_service as auth_service

from ragflow.api.schemas import CreateUserRequest, UpdateUserRequest, UserFromDB

router = APIRouter(
    tags=["User endpoints"],
)

import logging

logger = logging.getLogger(__name__)


@router.post(
    "/users",
    response_model=UserFromDB,
    status_code=status.HTTP_201_CREATED,
)
async def create_user(
    user_data: CreateUserRequest,
    db_session: orm.Session = Depends(get_db),
):
    return await user_service.create_user(user_data=user_data, db_session=db_session)


@router.get(
    "/users",
    response_model=list[UserFromDB],
    status_code=status.HTTP_200_OK,
)
async def get_all_users(
    user: UserFromDB = Depends(auth_service.get_current_active_user),
    db_session: orm.Session = Depends(get_db),
):
    if user.role != "admin":
        raise HTTPException(status_code=401, detail="User not authorized")

    return await user_service.get_all_users(db_session=db_session)


@router.get(
    "/users/{user_id}",
    response_model=UserFromDB,
    status_code=status.HTTP_200_OK,
)
async def get_user_by_id(
    user_id: int,
    user: UserFromDB = Depends(auth_service.get_current_active_user),
    db_session: orm.Session = Depends(get_db),
):
    if user.id != user_id and user.role != "admin":
        raise HTTPException(status_code=401, detail="User not authorized")

    return await user_service.get_user_by_id(user_id=user_id, db_session=db_session)


@router.delete(
    "/users/{user_id}",
    response_model=UserFromDB,
    status_code=status.HTTP_200_OK,
)
async def delete_user_by_id(
    user_id: int,
    user: UserFromDB = Depends(auth_service.get_current_active_user),
    db_session: orm.Session = Depends(get_db),
):
    if user.id != user_id and user.role != "admin":
        raise HTTPException(status_code=401, detail="User not authorized")

    user_to_delete = await user_service.get_user_by_id(
        user_id=user_id, db_session=db_session
    )

    await user_service.delete_user(user=user_to_delete, db_session=db_session)

    return user_to_delete


@router.put(
    "/users/{user_id}",
    response_model=UserFromDB,
    status_code=status.HTTP_200_OK,
)
async def update_user_by_id(
    user_id: int,
    user_data: UpdateUserRequest,
    user: UserFromDB = Depends(auth_service.get_current_active_user),
    db_session: orm.Session = Depends(get_db),
):
    if user.id != user_id and user.role != "admin":
        raise HTTPException(status_code=401, detail="User not authorized")

    user_to_change = await user_service.get_user_by_id(
        user_id=user_id, db_session=db_session
    )

    return await user_service.update_user(
        user=user_to_change, user_data=user_data, db_session=db_session
    )
