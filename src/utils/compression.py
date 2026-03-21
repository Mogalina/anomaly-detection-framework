from __future__ import annotations

import enum
import io
import torch
from typing import Dict, Any
from utils.config import get_config


class CompressionType(enum.IntEnum):
    # No compression
    NONE = 0

    # A fast, lossless compression algorithm that is suitable for real-time 
    # compression of model weights. It is a good choice for edge devices with 
    # limited CPU resources.
    LZ4  = 1

    # A high-ratio, lossless compression algorithm that achieves significantly
    # smaller payloads than LZ4 by using entropy coding (Huffman + dictionary).
    # Ideal for nodes with spare CPU cycles and bandwidth-constrained networks
    # (e.g. cellular or WAN links), especially with Top-K sparse gradients.
    ZSTD = 2


# Cache for payload compression configuration
_payload_config_cache = None


def _get_payload_config() -> Dict[str, Any]:
    """
    Get and cache the payload compression configuration.

    Returns:
        A dictionary containing compression configuration.
    """
    global _payload_config_cache

    if _payload_config_cache is None:
        cfg = get_config().get('federated.client.payload_compression', {})
        _payload_config_cache = {
            'high_cpu': cfg.get('cpu_usage_percent', {}).get('high_cpu_threshold', 85.0),
            'moderate_cpu': cfg.get('cpu_usage_percent', {}).get('moderate_cpu_threshold', 60.0),
            'zstd_moderate': cfg.get('zstd_level', {}).get('moderate', 1),
            'zstd_idle': cfg.get('zstd_level', {}).get('idle', 3),
        }

    return _payload_config_cache


def choose_algorithm(cpu_usage_percent: float) -> CompressionType:
    """
    Choose the best compression algorithm for payload transmission based on
    current CPU load.

    Args:
        cpu_usage_percent: Current CPU utilization in range [0, 100]

    Returns:
        The compression algorithm to use for this payload
    """
    cfg = _get_payload_config()

    if cpu_usage_percent >= cfg['high_cpu']:
        return CompressionType.LZ4
    
    return CompressionType.ZSTD


def _zstd_level(cpu_usage_percent: float) -> int:
    """
    Return the Zstd compression level appropriate for the current CPU load.

    Args:
        cpu_usage_percent: Current CPU utilization in range [0, 100]

    Returns:
        The Zstd compression level to use for this payload
    """
    cfg = _get_payload_config()

    if cpu_usage_percent >= cfg['moderate_cpu']:
        return cfg['zstd_moderate']

    return cfg['zstd_idle']


def serialize_state_dict(state_dict: Dict[str, Any]) -> bytes:
    """
    Serialize a PyTorch state dictionary to raw bytes.

    Args:
        state_dict: PyTorch state dictionary (or any dictionary of tensors)

    Returns:
        Raw bytes representing the serialized state dictionary
    """
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def deserialize_state_dict(raw: bytes) -> Dict[str, Any]:
    """
    Deserialize raw bytes back into a PyTorch state dict.

    Args:
        raw: Bytes representing the serialized state dictionary

    Returns:
        PyTorch state dictionary 
    """
    buffer = io.BytesIO(raw)
    return torch.load(buffer, map_location="cpu", weights_only=True)


def compress(
    raw: bytes,
    algorithm: CompressionType,
    cpu_usage_percent: float = 0.0,
) -> bytes:
    """
    Compress raw bytes with the given algorithm.

    Args:
        raw: Bytes to compress
        algorithm: Compression algorithm to use
        cpu_usage_percent: CPU usage percentage in range [0, 100]

    Returns:
        Compressed bytes
    """
    if algorithm == CompressionType.NONE:
        return raw

    if algorithm == CompressionType.LZ4:
        import lz4.frame
        return lz4.frame.compress(raw)

    if algorithm == CompressionType.ZSTD:
        import zstandard as zstd
        level = _zstd_level(cpu_usage_percent)
        compressor = zstd.ZstdCompressor(level=level)
        return compressor.compress(raw)

    raise ValueError(f"Unknown compression algorithm: {algorithm}")


def decompress(compressed: bytes, algorithm: CompressionType) -> bytes:
    """
    Decompress bytes that were compressed with algorithm.

    Args:
        compressed: Bytes to decompress
        algorithm:  Must match the algorithm used during compression

    Returns:
        Original raw bytes
    """
    if algorithm == CompressionType.NONE:
        return compressed

    if algorithm == CompressionType.LZ4:
        import lz4.frame
        return lz4.frame.decompress(compressed)

    if algorithm == CompressionType.ZSTD:
        import zstandard as zstd
        decompressor = zstd.ZstdDecompressor()
        return decompressor.decompress(compressed)

    raise ValueError(f"Unknown compression algorithm: {algorithm}")


def pack_state_dict(
    state_dict: Dict[str, Any],
    cpu_usage_percent: float,
) -> tuple[bytes, CompressionType]:
    """
    Serialize and adaptively compress a state dictionary.

    Args:
        state_dict: PyTorch model state dictionary
        cpu_usage_percent: CPU usage percentage in range [0, 100]

    Returns:
        Compressed state dictionary and the compression algorithm used
    """
    # Choose the best compression algorithm based on CPU usage
    algorithm = choose_algorithm(cpu_usage_percent)

    # Serialize and compress the state dictionary
    raw = serialize_state_dict(state_dict)
    compressed = compress(raw, algorithm, cpu_usage_percent)

    return compressed, algorithm


def unpack_state_dict(
    compressed: bytes,
    algorithm: CompressionType,
) -> Dict[str, Any]:
    """
    Decompress and deserialize a state dictionary.

    Args:
        compressed: Bytes from compressed state dictionary
        algorithm: Compression algorithm used

    Returns:
        PyTorch state dictionary
    """
    # Decompress and deserialize the state dictionary
    raw = decompress(compressed, algorithm) 
    return deserialize_state_dict(raw)
