from ragflow.api.database import Session


def get_db():
    db_session = Session()
    try:
        yield db_session
    finally:
        db_session.close()
