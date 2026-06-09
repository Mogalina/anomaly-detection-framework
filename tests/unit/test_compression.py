import pytest
import torch


class TestCompressionRoundtrip:
    """
    Validates byte-level compression and decompression roundtrips across all formats.
    """

    def test_lz4_roundtrip(self):
        """
        Verify that LZ4 compression followed by decompression yields identical bytes.
        """
        from utils.compression import compress, decompress, CompressionType
        data = b"hello world " * 1000
        compressed = compress(data, CompressionType.LZ4)
        assert decompress(compressed, CompressionType.LZ4) == data

    def test_zstd_roundtrip(self):
        """
        Verify that Zstd compression followed by decompression yields identical bytes.
        """
        from utils.compression import compress, decompress, CompressionType
        data = b"hello world " * 1000
        compressed = compress(data, CompressionType.ZSTD)
        assert decompress(compressed, CompressionType.ZSTD) == data

    def test_none_roundtrip(self):
        """
        Verify that using no compression (NONE) behaves as an identity map.
        """
        from utils.compression import compress, decompress, CompressionType
        data = b"test data"
        assert compress(data, CompressionType.NONE) == data
        assert decompress(data, CompressionType.NONE) == data

    def test_lz4_actually_compresses(self):
        """
        Verify that LZ4 successfully reduces the byte footprint of repetitive data.
        """
        from utils.compression import compress, CompressionType
        data = b"repetitive data! " * 5000
        compressed = compress(data, CompressionType.LZ4)
        assert len(compressed) < len(data)

    def test_zstd_actually_compresses(self):
        """
        Verify that Zstd successfully reduces the byte footprint of repetitive data.
        """
        from utils.compression import compress, CompressionType
        data = b"repetitive data! " * 5000
        compressed = compress(data, CompressionType.ZSTD)
        assert len(compressed) < len(data)

    def test_unknown_algorithm_raises(self):
        """
        Verify that requesting compression for an unregistered integer code raises ValueError.
        """
        from utils.compression import compress
        with pytest.raises(ValueError):
            compress(b"test", 99)


class TestAdaptiveSelection:
    """
    Validates adaptive selection logic choosing between LZ4 and Zstd based on CPU usage.
    """

    def test_high_cpu_selects_lz4(self):
        """
        Confirm that high CPU utilization chooses LZ4 compression for low CPU overhead.
        """
        from utils.compression import choose_algorithm, CompressionType
        assert choose_algorithm(95.0) == CompressionType.LZ4

    def test_low_cpu_selects_zstd(self):
        """
        Confirm that low CPU utilization chooses Zstd compression for high compression ratio.
        """
        from utils.compression import choose_algorithm, CompressionType
        assert choose_algorithm(30.0) == CompressionType.ZSTD

    def test_moderate_cpu_selects_zstd(self):
        """
        Confirm that moderate CPU utilization defaults to Zstd compression.
        """
        from utils.compression import choose_algorithm, CompressionType
        assert choose_algorithm(70.0) == CompressionType.ZSTD

    def test_boundary_cpu(self):
        """
        Confirm boundary CPU utilization thresholds select the correct compression type.
        """
        from utils.compression import choose_algorithm, CompressionType
        assert choose_algorithm(85.0) == CompressionType.LZ4

    def test_zstd_higher_ratio_than_lz4(self):
        """
        Confirm that Zstd achieves a higher compression ratio than LZ4 on typical telemetry.
        """
        from utils.compression import compress, CompressionType
        data = b"some model weights data " * 5000
        lz4_out = compress(data, CompressionType.LZ4)
        zstd_out = compress(data, CompressionType.ZSTD, cpu_usage_percent=30.0)
        assert len(zstd_out) <= len(lz4_out)


class TestStateDictSerialization:
    """
    Validates model parameter serialization and deserialization routines.
    """

    def test_serialize_deserialize(self):
        """
        Verify that serializing and deserializing PyTorch parameter state dicts retains precision.
        """
        from utils.compression import serialize_state_dict, deserialize_state_dict
        sd = {"layer.weight": torch.randn(4, 4), "layer.bias": torch.randn(4)}
        raw = serialize_state_dict(sd)
        assert isinstance(raw, bytes)
        restored = deserialize_state_dict(raw)
        for key in sd:
            assert torch.allclose(sd[key], restored[key])


class TestPackUnpack:
    """
    Validates full package wrapping, compression, and unpacking routines.
    """

    def test_pack_unpack_preserves_model(self, lstm_model):
        """
        Verify that wrapping and unpacking preserves all LSTM model parameters.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
        """
        from utils.compression import pack_state_dict, unpack_state_dict
        original = lstm_model.state_dict()
        payload, alg = pack_state_dict(original, cpu_usage_percent=50.0)
        restored = unpack_state_dict(payload, alg)
        for key in original:
            assert torch.allclose(original[key], restored[key], atol=1e-6)

    def test_pack_returns_bytes_and_algorithm(self, lstm_model):
        """
        Verify that packing returns wrapped bytes alongside the selected algorithm enum.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
        """
        from utils.compression import pack_state_dict, CompressionType
        payload, alg = pack_state_dict(lstm_model.state_dict(), cpu_usage_percent=50.0)
        assert isinstance(payload, bytes)
        assert isinstance(alg, CompressionType)

    def test_pack_high_cpu(self, lstm_model):
        """
        Verify that packing chooses LZ4 when CPU usage is high.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
        """
        from utils.compression import pack_state_dict, CompressionType
        _, alg = pack_state_dict(lstm_model.state_dict(), cpu_usage_percent=95.0)
        assert alg == CompressionType.LZ4

    def test_pack_low_cpu(self, lstm_model):
        """
        Verify that packing chooses Zstd when CPU usage is low.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
        """
        from utils.compression import pack_state_dict, CompressionType
        _, alg = pack_state_dict(lstm_model.state_dict(), cpu_usage_percent=20.0)
        assert alg == CompressionType.ZSTD
