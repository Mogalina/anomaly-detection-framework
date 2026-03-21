from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base

from utils.config import get_config
from utils.logger import get_logger


logger = get_logger(__name__)


Base = declarative_base()


_engine = None
_SessionLocal = None


def _get_database_url() -> str:
    """
    Build a PostgreSQL connection URL from configuration settings.

    Returns:
        The connection URL as a string
    """
    config = get_config()
    
    postgres_config = config.get('database.postgres', {})
    host = postgres_config.get('host', 'localhost')
    port = postgres_config.get('port', 5432)
    db = postgres_config.get('database', 'anomaly_detection')
    user = postgres_config.get('user', 'adf_user')
    pwd = postgres_config.get('password', 'adf_password')
    
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


def get_engine():
    """Return the shared SQLAlchemy engine."""
    global _engine
    
    if _engine is None:
        url = _get_database_url()
        
        _engine = create_engine(
            url,
            pool_size=postgres_config.get('pool_size', 10),
            max_overflow=postgres_config.get('max_overflow', 20),
            pool_pre_ping=True,
        )

        logger.debug(f"Database engine created: {url.split('@')[1]}")
    
    return _engine


def init_db() -> None:
    """Create all tables. Tables are idempotent so that it is safe to call on every startup."""
    import utils.db_models

    Base.metadata.create_all(bind=get_engine())
    logger.info("Database schema initialised")


def get_session_factory():
    """Return the session factory."""
    global _SessionLocal

    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )
    
    return _SessionLocal


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager that yields a database session and handles commits, rollbacks 
    and closes automatically.

    Returns:
        A database session
    """
    factory = get_session_factory()
    session: Session = factory()
    
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
