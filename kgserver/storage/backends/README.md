# Storage Backends

This directory contains concrete implementations of the storage interfaces for different database backends.

## Available Backends

### SQLite (`sqlite.py`)

SQLite implementation for development, testing, and small-scale deployments.

**Features:**
- In-memory or file-based storage
- No external dependencies
- Fast for small datasets
- Stores generic `Entity` and `Relationship` models

**Use Cases:**
- Local development
- Automated testing
- CI/CD ingests
- Small projects
- Prototyping

**Connection Examples:**
```python
from storage.backends.sqlite import SQLiteStorage

# Persistent file database
storage = SQLiteStorage("data/knowledge_graph.db")
```

### PostgreSQL (`postgres.py`)

PostgreSQL implementation for production deployments.

**Features:**
- Full ACID compliance
- Concurrent read/write operations
- Stores generic `Entity` and `Relationship` models
- Optimized for large datasets

**Use Cases:**
- Production deployments
- Large datasets
- Multi-user applications
- When you need robust concurrent access

**Connection Examples:**
```python
from storage.backends.postgres import PostgresStorage
from sqlmodel import create_engine, Session

# Example with connection string
engine = create_engine("postgresql://user:password@localhost:5432/mydatabase")
session = Session(engine)
storage = PostgresStorage(session)
```

## Performance Characteristics

| Feature | SQLite | PostgreSQL |
|---------|--------|------------|
| Setup | Easy | Moderate |
| Small datasets | Excellent | Good |
| Large datasets | Good (with care) | Excellent |
| Concurrent writes | Poor | Excellent |
| Concurrent reads | Good | Excellent |
| Memory usage | Low | Moderate |
| Disk usage | Low | Moderate |

## Connection String Formats

### SQLite
```
:memory:                    # In-memory database
relative/path/db.sqlite     # Relative path
/absolute/path/db.sqlite    # Absolute path
```

### PostgreSQL
```
postgresql://[user[:password]@][host][:port][/dbname][?param1=value1&...]

Examples:
postgresql://localhost/mydatabase
postgresql://user:pass@localhost:5432/mydatabase
```

## Troubleshooting

### SQLite Issues

**"Database is locked" error:**
- Ensure you're closing connections properly
- Consider PostgreSQL for concurrent access

**Slow performance:**
- Add appropriate indexes
- Use transactions for bulk inserts

### PostgreSQL Issues

**Connection refused:**
- Check PostgreSQL is running
- Verify connection parameters
- Check firewall settings

**Slow queries:**
- Add indexes on frequently queried columns
- Use `EXPLAIN ANALYZE` to debug queries
- Adjust PostgreSQL settings

## Further Reading

- [Storage Layer Overview](../README.md)
- [Database Schema Documentation](../models/README.md)