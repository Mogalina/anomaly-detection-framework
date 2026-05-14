"""
Redis cache client for ephemeral coordinator state.

Stores in-flight round updates so the coordinator can survive a process restart 
mid-round without losing updates that were already received.

Falls back to a simple in-memory dictionary if Redis is unreachable so the 
coordinator can still run in environments without Redis.
"""

import json
from typing import Any, Dict, Optional

from utils.config import get_config
from utils.logger import get_logger


logger = get_logger(__name__)


_redis_client = None
_fallback_store: Dict[str, str] = {}
_redis_available = False
_redis_checked = False


def _try_connect() -> None:
    """Attempt to connect to Redis and fall back to in-memory cache if unavailable."""
    global _redis_client, _redis_available, _redis_checked
    
    _redis_checked = True

    try:
        import redis as _redis_lib

        redis_config = get_config().get('database.redis', {})

        _redis_client = _redis_lib.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            decode_responses=True,
            socket_connect_timeout=2,
        )
        _redis_client.ping()

        _redis_available = True
        logger.info("Redis connected successfully")
    except Exception as exc:
        _redis_available = False
        _redis_client = None
        logger.warning(f"Redis not available, falling back to in-memory cache: {exc}")


def get_redis() -> Optional["redis.Redis"]:
    """
    Return the Redis client if available.

    Returns:
        The Redis client if available, otherwise None
    """
    global _redis_client, _redis_checked

    if not _redis_checked:
        _try_connect()

    return _redis_client if _redis_available else None


def _key(round_num: int, client_id: str) -> str:
    """
    Return the key for a client update.

    Args:
        round_num: Current FL round number
        client_id: The submitting client

    Returns:
        The key for the client update
    """ 
    return f"round:{round_num}:client:{client_id}"


def cache_client_update(
    round_num: int,
    client_id: str,
    update: Dict[str, Any],
    ttl: int = 3600
) -> None:
    """
    Persist a client update dictionary in Redis, or in-memory fallback.

    Args:
        round_num: Current Federated Learning round number
        client_id: The submitting client
        update: The full update dictionary
        ttl: Optional time-to-live in seconds
    """
    key  = _key(round_num, client_id)
    blob = json.dumps(update, default=str)

    redis_client = get_redis()
    if redis_client is not None:
        redis_client.setex(key, ttl, blob)
    else:
        _fallback_store[key] = blob


def get_client_update(
    round_num: int,
    client_id: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a cached client update dictionary.

    Args:
        round_num: Current Federated Learning round number
        client_id: The submitting client

    Returns:
        The update dictionary, or None if not found
    """
    key = _key(round_num, client_id)

    redis_client = get_redis()
    blob = redis_client.get(key) if redis_client is not None else _fallback_store.get(key)

    return json.loads(blob) if blob else None


def delete_client_update(round_num: int, client_id: str) -> None:
    """
    Remove a cached update (called after successful aggregation).

    Args:
        round_num: Current Federated Learning round number
        client_id: The submitting client
    """
    key = _key(round_num, client_id)

    redis_client = get_redis()
    if redis_client is not None:
        redis_client.delete(key)
    else:
        _fallback_store.pop(key, None)


def list_cached_updates(round_num: int) -> Dict[str, Dict[str, Any]]:
    """
    Return all cached updates for *round_num*.

    Args:
        round_num: Current Federated Learning round number

    Returns:
        Dictionary mapping client_id to update dict
    """
    prefix = f"round:{round_num}:client:"
    result: Dict[str, Dict] = {}

    redis_client = get_redis()
    if redis_client is not None:
        keys = redis_client.keys(f"{prefix}*")
        for key in keys:
            blob = redis_client.get(key)
            if blob:
                client_id = key[len(prefix):]
                result[client_id] = json.loads(blob)
    else:
        for key, blob in _fallback_store.items():
            if key.startswith(prefix):
                client_id = key[len(prefix):]
                result[client_id] = json.loads(blob)

    return result


def is_redis_available() -> bool:
    """
    Check if Redis is connected and available.

    Returns:
        True if Redis is connected and available, False otherwise
    """
    redis_client = get_redis()
    return redis_client is not None
