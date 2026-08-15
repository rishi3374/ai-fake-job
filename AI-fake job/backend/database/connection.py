"""
Database Connection Module
Handles PostgreSQL database connection and session management
"""

import logging
from typing import Optional
from pathlib import Path

# Optional SQLAlchemy imports
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import StaticPool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection manager"""
    
    def __init__(self, database_url: str = "sqlite:///./fake_job_detector.db"):
        """
        Initialize database manager
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection"""
        if not SQLALCHEMY_AVAILABLE:
            logger.warning("SQLAlchemy not available, database functionality will be limited")
            return
        
        try:
            # Create engine
            if self.database_url.startswith("sqlite"):
                self.engine = create_engine(
                    self.database_url,
                    connect_args={"check_same_thread": False},
                    poolclass=StaticPool
                )
            else:
                self.engine = create_engine(self.database_url)
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            logger.info(f"Database connection established: {self.database_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def get_session(self) -> Optional[Session]:
        """
        Get database session
        
        Returns:
            SQLAlchemy Session or None if not available
        """
        if not SQLALCHEMY_AVAILABLE or self.SessionLocal is None:
            logger.warning("Database session not available")
            return None
        
        return self.SessionLocal()
    
    def create_tables(self):
        """Create all database tables"""
        if not SQLALCHEMY_AVAILABLE or self.engine is None:
            logger.warning("Cannot create tables - database not available")
            return
        
        try:
            from backend.database.models import Base
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
    
    def drop_tables(self):
        """Drop all database tables"""
        if not SQLALCHEMY_AVAILABLE or self.engine is None:
            logger.warning("Cannot drop tables - database not available")
            return
        
        try:
            from backend.database.models import Base
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


# Global database manager instance
db_manager = None


def get_database_manager(database_url: str = "sqlite:///./fake_job_detector.db") -> DatabaseManager:
    """
    Get or create database manager instance
    
    Args:
        database_url: Database connection URL
        
    Returns:
        DatabaseManager instance
    """
    global db_manager
    
    if db_manager is None:
        db_manager = DatabaseManager(database_url)
    
    return db_manager


def get_session() -> Optional[Session]:
    """
    Get database session (convenience function)
    
    Returns:
        SQLAlchemy Session or None
    """
    manager = get_database_manager()
    return manager.get_session()


def init_database(database_url: str = "sqlite:///./fake_job_detector.db"):
    """
    Initialize database with tables
    
    Args:
        database_url: Database connection URL
    """
    manager = get_database_manager(database_url)
    manager.create_tables()


if __name__ == "__main__":
    # Test database connection
    print("=== Database Connection Test ===")
    
    # Initialize with SQLite (default)
    manager = get_database_manager()
    
    if SQLALCHEMY_AVAILABLE:
        print("SQLAlchemy available - testing database connection")
        
        # Create tables
        manager.create_tables()
        
        # Test session
        session = manager.get_session()
        if session:
            print("Database session created successfully")
            session.close()
        else:
            print("Failed to create database session")
        
        # Close connection
        manager.close()
    else:
        print("SQLAlchemy not available - install with: pip install sqlalchemy")
