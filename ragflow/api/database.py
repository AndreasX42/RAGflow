from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

import os

DRIVER = os.environ.get("POSTGRES_DRIVER")
HOST = os.environ.get("POSTGRES_HOST", "localhost")
PORT = os.environ.get("POSTGRES_PORT")
DB = os.environ.get("POSTGRES_DATABASE")
USER = os.environ.get("POSTGRES_USER")
PWD = os.environ.get("POSTGRES_PASSWORD")


if HOST == "localhost":
    HOST += f":{PORT}"

DATABASE_URL = f"postgresql+{DRIVER}://{USER}:{PWD}@{HOST}/{DB}"

import logging

logger = logging.getLogger(__name__)
logger.error(f"\n HI {DATABASE_URL}\n")


engine = create_engine(DATABASE_URL)

Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
