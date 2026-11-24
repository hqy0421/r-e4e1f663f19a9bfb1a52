import logging
from datetime import datetime
from typing import Any, List

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    Float,
    ForeignKey,
    Index,
    MetaData,
    event,
    inspect,
)
from sqlalchemy.schema import CreateTable
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session

logger = logging.getLogger(__name__)

Base = declarative_base()

# Length of the source-scoped entry key. SQLite does not enforce VARCHAR
# lengths, but keeping it explicit documents the intended maximum:
# a decimal feed id + ":" + a guid of up to 512 characters.
ENTRY_KEY_MAX_LENGTH = 640


def make_entry_key(feed_id: int, guid: str) -> str:
    """Build the stable, source-scoped identity key for an article.

    The same short GUID emitted by two different subscriptions (a common
    practice among public feeds) must map to two distinct stored entries.
    Prefixing the GUID with the owning feed id gives every source its own
    namespace, while remaining deterministic so re-ingestion can look an
    entry up without consulting URLs.

    Args:
        feed_id: Database id of the owning feed (the source identity).
        guid: The entry GUID exactly as published by the source.

    Returns:
        A key of the form ``"<feed_id>:<guid>"``.
    """
    return f"{feed_id}:{guid}"


class Feed(Base):
    """Model for RSS feed subscriptions."""
    __tablename__ = "feeds"

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    url = Column(String(512), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    website_url = Column(String(512), nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    last_checked = Column(DateTime, default=datetime.utcnow)
    muted = Column(Boolean, default=False)
    muted_reason = Column(Text, nullable=True)  # Reason for muting the feed
    peer_through = Column(Boolean, default=False)  # Whether to peer through aggregator to origin article
    error_count = Column(Integer, default=0)  # Count of consecutive errors
    no_entries_count = Column(Integer, default=0)  # Count of consecutive checks with no entries
    last_error = Column(Text, nullable=True)
    
    # Statistics
    article_count = Column(Integer, default=0)  # Total count of ingested articles
    avg_quality_score = Column(Float, nullable=True)  # Average quality score of articles
    quality_tier_counts = Column(Text, nullable=True)  # JSON string of quality tier counts {"S": 5, "A": 10, ...}
    
    # Relationships
    articles = relationship("Article", back_populates="feed", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Feed(id={self.id}, title='{self.title}', muted={self.muted}, peer_through={self.peer_through})>"


class Article(Base):
    """Model for articles from RSS feeds."""
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True)
    feed_id = Column(Integer, ForeignKey("feeds.id"), nullable=False)
    title = Column(String(512), nullable=False)
    url = Column(String(512), nullable=False)
    # GUID as published by the source. NOT globally unique: independent
    # subscriptions may reuse the same short GUID. See entry_key.
    guid = Column(String(512), nullable=False)
    # Source-scoped identity: "<feed_id>:<guid>". This is the persisted
    # deduplication key; two feeds sharing a GUID (or a URL) stay distinct.
    entry_key = Column(String(ENTRY_KEY_MAX_LENGTH), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    author = Column(String(255), nullable=True)
    published_at = Column(DateTime, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime, nullable=True)
    summary = Column(Text, nullable=True)
    quality_tier = Column(String(1), nullable=True)  # S, A, B, C, D
    quality_score = Column(Integer, nullable=True)  # 1-100
    labels = Column(String(255), nullable=True)
    embedding_generated = Column(Boolean, default=False)
    embedding_vector = Column(Text, nullable=True)  # JSON-serialized embedding vector
    word_count = Column(Integer, nullable=True)
    jina_enhanced = Column(Boolean, default=False)  # Flag to indicate if content was fetched from Jina.ai
    
    # Relationships
    feed = relationship("Feed", back_populates="articles")

    __table_args__ = (
        # Non-unique: two different sources legitimately carry the same
        # article URL. The index makes "which sources reported this URL?"
        # lookups cheap without collapsing those records.
        Index("ix_articles_url", "url"),
    )

    def __repr__(self) -> str:
        return f"<Article(id={self.id}, title='{self.title[:30]}...', quality_tier='{self.quality_tier or 'None'}')>"


@event.listens_for(Article, "before_insert")
def _populate_entry_key(mapper, connection, target: Article) -> None:
    """Derive entry_key from the owning feed and GUID when not set explicitly.

    Callers that assign ``feed=...`` rather than ``feed_id=...`` are covered
    too, so the key can never be persisted as NULL and no insert path can
    bypass source-scoped identity.
    """
    if target.entry_key:
        return
    feed_id = target.feed_id
    if feed_id is None and target.feed is not None:
        feed_id = target.feed.id
    if feed_id is None or target.guid is None:
        return
    target.entry_key = make_entry_key(feed_id, target.guid)


def _missing_columns(existing: List[str]) -> List[Any]:
    """Return model columns that do not exist in the physical table."""
    return [col for col in Article.__table__.columns if col.name not in existing]


def _single_column_unique_indexes(engine, table: str) -> set:
    """Return the set of columns that alone form a UNIQUE index on ``table``.

    Reads SQLite's ``PRAGMA index_list``/``index_info`` directly because
    SQLAlchemy reflection can miss inline ``UNIQUE`` column constraints
    (reporting an empty constraint list) depending on how the table was
    created. Both auto-index forms (``INTEGER PRIMARY KEY`` and column-level
    UNIQUE) are covered.
    """
    columns = set()
    with engine.connect() as conn:
        indexes = conn.exec_driver_sql(f'PRAGMA index_list("{table}")').fetchall()
        for index_row in indexes:
            # Layout: seq, name, unique, origin, partial
            index_name, is_unique = index_row[1], index_row[2]
            if not is_unique:
                continue
            info = conn.exec_driver_sql(f'PRAGMA index_info("{index_name}")').fetchall()
            if len(info) == 1:
                columns.add(info[0][2])
    return columns


def _schema_needs_rebuild(engine, existing: List[str], missing: List[Any]) -> bool:
    """Decide whether ``articles`` must be rebuilt rather than altered.

    Rebuild is required when:
    - ``entry_key`` is absent or any required column is missing (it cannot be
      added in place with a value for existing rows), or
    - ``guid`` still carries a single-column UNIQUE constraint. That
      constraint is the bug: it collapses distinct sources that reuse a short
      GUID, so it must go.
    """
    if any(col.name == "entry_key" for col in missing):
        return True
    if any(not col.nullable for col in missing):
        return True

    return "guid" in _single_column_unique_indexes(engine, "articles")


def _rebuild_articles_table(engine) -> None:
    """Rebuild ``articles`` while preserving primary keys and row data.

    Used for legacy databases that predate ``entry_key`` (and possibly other
    columns). Rebuilding, rather than an in-place ALTER, is required because
    the legacy schema carries ``guid UNIQUE``, which is exactly the global
    constraint that collapsed distinct sources sharing a short GUID.

    Existing ``id`` values are copied verbatim: article ids are referenced by
    the Annoy vector index and by external links, so renumbering would break
    both. The rebuild happens in one transaction; the temporary table is
    dropped if the row count does not match, leaving the original intact.
    """
    insp = inspect(engine)
    existing = [col["name"] for col in insp.get_columns("articles")]
    target_names = [col.name for col in Article.__table__.columns]

    tmp_name = "articles__entry_key_migration"
    md = MetaData()
    Feed.__table__.to_metadata(md)
    tmp_table = Article.__table__.to_metadata(md, name=tmp_name)

    def source_expr(name: str) -> str:
        if name == "entry_key":
            if "entry_key" in existing:
                # Preserve an already-assigned key; only derive it for rows
                # that have none. Overwriting would silently re-key entries.
                return (
                    'COALESCE(NULLIF("entry_key", \'\'), '
                    "CAST(feed_id AS TEXT) || ':' || guid)"
                )
            # Derive source-scoped identity from the existing row.
            return "CAST(feed_id AS TEXT) || ':' || guid"
        if name in existing:
            return f'"{name}"'
        return "NULL"

    column_list = ", ".join(f'"{name}"' for name in target_names)
    select_list = ", ".join(source_expr(name) for name in target_names)

    with engine.begin() as conn:
        conn.exec_driver_sql(f'DROP TABLE IF EXISTS "{tmp_name}"')
        conn.exec_driver_sql(str(CreateTable(tmp_table).compile(dialect=conn.dialect)))
        conn.exec_driver_sql(
            f'INSERT INTO "{tmp_name}" ({column_list}) SELECT {select_list} FROM articles'
        )
        old_count = conn.exec_driver_sql("SELECT COUNT(*) FROM articles").scalar()
        new_count = conn.exec_driver_sql(f'SELECT COUNT(*) FROM "{tmp_name}"').scalar()
        if old_count != new_count:
            raise RuntimeError(
                f"Article table migration copied {new_count} of {old_count} rows; aborting"
            )
        conn.exec_driver_sql("DROP TABLE articles")
        conn.exec_driver_sql(f'ALTER TABLE "{tmp_name}" RENAME TO articles')
        # Recreate the indexes declared on the model (URL lookup, etc.).
        for index in Article.__table__.indexes:
            index.create(conn, checkfirst=True)

    logger.info("Migrated articles table to source-scoped entry keys (%s rows)", new_count)


def _ensure_article_schema(engine) -> None:
    """Bring an existing database up to the current Article schema.

    Idempotent, and safe to call on every connection: fresh databases fall
    out after ``create_all``; already-migrated ones only pay for reflection.
    """
    Base.metadata.create_all(engine)

    insp = inspect(engine)
    if "articles" not in insp.get_table_names():
        return

    existing = [col["name"] for col in insp.get_columns("articles")]
    missing = _missing_columns(existing)

    if _schema_needs_rebuild(engine, existing, missing):
        _rebuild_articles_table(engine)
    elif missing:
        # Only nullable additions (e.g. embedding_vector) are needed.
        with engine.begin() as conn:
            for col in missing:
                col_type = col.type.compile(dialect=conn.dialect)
                conn.exec_driver_sql(
                    f'ALTER TABLE articles ADD COLUMN "{col.name}" {col_type}'
                )

    # Ensure declared indexes exist on pre-existing tables. create_all()
    # does not add indexes to an existing table, and older schemas had no
    # index on url at all.
    with engine.begin() as conn:
        for index in Article.__table__.indexes:
            index.create(conn, checkfirst=True)


def _open_session(db_path: str) -> Session:
    """Open a session against ``db_path``, migrating the schema if needed."""
    engine = create_engine(f"sqlite:///{db_path}")
    _ensure_article_schema(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def init_db(db_path: str) -> Session:
    """Initialize database and return session."""
    return _open_session(db_path)


def get_db_session(db_path: str) -> Session:
    """Get a database session."""
    return _open_session(db_path)