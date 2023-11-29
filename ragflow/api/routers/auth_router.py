from fastapi import APIRouter, Depends, HTTPException, Response, Request
from starlette import status
import sqlalchemy.orm as orm
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta

from ragflow.api.schemas import UserFromDB
from ragflow.api.services import get_db
from ragflow.api.services.auth_service import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication endpoints"],
)

ACCESS_TOKEN_EXPIRATION_IN_MINUTES = 60 * 24 * 7  # 7 days


@router.get("/user", response_model=UserFromDB, status_code=status.HTTP_200_OK)
async def get_authenticated_user(
    user: UserFromDB = Depends(get_current_active_user),
):
    return user


@router.post("/login", response_model=UserFromDB, status_code=status.HTTP_200_OK)
async def login_for_access_token(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db_session: orm.Session = Depends(get_db),
):
    user = await authenticate_user(
        username=form_data.username, password=form_data.password, db_session=db_session
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )

    jwt, exp = create_access_token(
        subject={"sub": user.username, "user_id": user.id, "user_role": user.role},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRATION_IN_MINUTES),
    )

    response.set_cookie(
        key="access_token",
        value=jwt,
        expires=int(exp.timestamp()),
        httponly=True,
    )

    return UserFromDB.model_validate(user)


@router.get("/logout", response_model=dict, status_code=status.HTTP_200_OK)
async def logout(response: Response):
    response.delete_cookie(key="access_token")

    return {"message": "Logged out successfully"}
