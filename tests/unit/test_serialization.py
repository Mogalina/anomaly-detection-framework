import os
import tempfile
import pytest
import torch


class TestModelSerialization:
    """
    Validates model weight and optimizer state serialization routines.
    """

    def test_save_and_load(self, lstm_model):
        """
        Verify that model parameters and custom round/epoch metadata save and reload.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
        """
        from utils.serialization import save_model, load_model
        from edge.models import LSTMAnomalyDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            save_model(lstm_model, path, metadata={"round": 5}, epoch=10)

            loaded = LSTMAnomalyDetector(input_size=10, hidden_size=32, num_layers=1, dropout=0.0)
            result = load_model(loaded, path)

            assert result["metadata"]["round"] == 5
            assert result["epoch"] == 10
            for key in lstm_model.state_dict():
                assert torch.allclose(lstm_model.state_dict()[key], loaded.state_dict()[key])

    def test_save_creates_parent_dirs(self, lstm_model):
        """
        Verify saving to a nested directory path auto-creates parent folders.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
        """
        from utils.serialization import save_model

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "model.pt")
            save_model(lstm_model, path)
            assert os.path.exists(path)

    def test_save_with_optimizer(self, lstm_model):
        """
        Verify saving and reloading training optimizer state parameters.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
        """
        from utils.serialization import save_model, load_model
        from edge.models import LSTMAnomalyDetector

        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            save_model(lstm_model, path, optimizer=optimizer)

            loaded = LSTMAnomalyDetector(input_size=10, hidden_size=32, num_layers=1, dropout=0.0)
            loaded_opt = torch.optim.Adam(loaded.parameters(), lr=0.001)
            result = load_model(loaded, path, load_optimizer=True, optimizer=loaded_opt)
            assert result.get("optimizer_loaded") is True


class TestObjectSerialization:
    """
    Validates generic pickle object serialization.
    """

    def test_save_and_load_object(self):
        """
        Verify saving and loading dict object state.
        """
        from utils.serialization import save_object, load_object

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "obj.pkl")
            data = {"key": [1, 2, 3], "nested": {"a": "b"}}
            save_object(data, path)
            loaded = load_object(path)
            assert loaded == data


class TestJsonSerialization:
    """
    Validates JSON serialization.
    """

    def test_save_and_load_json(self):
        """
        Verify saving and loading JSON dict configurations.
        """
        from utils.serialization import save_json, load_json

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.json")
            data = {"key": "value", "number": 42}
            save_json(data, path)
            loaded = load_json(path)
            assert loaded == data

    def test_json_creates_parent_dirs(self):
        """
        Verify saving JSON creates missing target path parent directories.
        """
        from utils.serialization import save_json

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "data.json")
            save_json({"test": True}, path)
            assert os.path.exists(path)
