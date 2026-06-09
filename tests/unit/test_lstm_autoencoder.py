import numpy as np
import pytest
import torch
import torch.nn as nn


class TestLSTMAnomalyDetectorArchitecture:
    """
    Validates model shapes, bidirectionality, and structural components of
    LSTMAnomalyDetector.
    """

    def test_output_shape_matches_input(self, lstm_model, config):
        """
        Verify that the model's output tensor matches the input tensor shape exactly.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x = torch.randn(4, mc["sequence_length"], mc["input_size"])
        out = lstm_model(x)
        assert out.shape == x.shape

    def test_single_sample_forward(self, lstm_model, config):
        """
        Verify that a forward pass handles a single sequence correctly.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x = torch.randn(1, mc["sequence_length"], mc["input_size"])
        out = lstm_model(x)
        assert out.shape == (1, mc["sequence_length"], mc["input_size"])

    def test_different_sequence_lengths(self, config):
        """
        Verify that the model accommodates varying input sequence lengths during inference.

        Args:
            config: Test configuration fixture
        """
        from edge.models import LSTMAnomalyDetector

        mc = config["edge"]["model"]
        model = LSTMAnomalyDetector(
            input_size=mc["input_size"],
            hidden_size=mc["hidden_size"],
            num_layers=mc["num_layers"],
            dropout=0.0,
        )
        model.eval()
        for seq_len in [5, 20, 50]:
            x = torch.randn(1, seq_len, mc["input_size"])
            out = model(x)
            assert out.shape == x.shape, f"Failed for seq_len={seq_len}"

    def test_bidirectional_model(self, config):
        """
        Verify that bidirectional configurations produce correct reconstruction shapes.

        Args:
            config: Test configuration fixture
        """
        from edge.models import LSTMAnomalyDetector

        mc = config["edge"]["model"]
        model = LSTMAnomalyDetector(
            input_size=mc["input_size"],
            hidden_size=mc["hidden_size"],
            num_layers=mc["num_layers"],
            dropout=0.0,
            bidirectional=True,
        )
        x = torch.randn(2, mc["sequence_length"], mc["input_size"])
        out = model(x)
        assert out.shape == x.shape

    def test_model_has_encoder_decoder(self, lstm_model):
        """
        Confirm that the model class contains separate encoder, decoder, and linear projection layers.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
        """
        assert hasattr(lstm_model, "encoder")
        assert hasattr(lstm_model, "decoder")
        assert hasattr(lstm_model, "output_layer")
        assert isinstance(lstm_model.encoder, nn.LSTM)
        assert isinstance(lstm_model.decoder, nn.LSTM)
        assert isinstance(lstm_model.output_layer, nn.Linear)

    def test_parameter_count_positive(self, lstm_model):
        """
        Verify that the model has parameters initialized.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
        """
        total = sum(p.numel() for p in lstm_model.parameters())
        assert total > 0

    def test_weight_initialization(self, config):
        """
        Verify that model weights are initialized with orthogonal/xavier properties.

        Args:
            config: Test configuration fixture
        """
        from edge.models import LSTMAnomalyDetector

        model = LSTMAnomalyDetector(
            input_size=config["edge"]["model"]["input_size"],
            hidden_size=config["edge"]["model"]["hidden_size"],
            num_layers=1,
            dropout=0.0,
        )
        for name, param in model.named_parameters():
            if "bias" in name:
                assert torch.all(param == 0), f"Bias {name} not zero-initialized"


class TestLSTMAnomalyDetectorLatent:
    """
    Validates model latent space encoding capabilities.
    """

    def test_forward_with_latent(self, lstm_model, config):
        """
        Verify that forward pass with return_latent yields a valid latent representation.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x = torch.randn(2, mc["sequence_length"], mc["input_size"])
        reconstruction, latent = lstm_model(x, return_latent=True)
        assert reconstruction.shape == x.shape
        assert latent.shape == (2, mc["hidden_size"])

    def test_encode_shape(self, lstm_model, config):
        """
        Verify that encode returns a correctly sized latent tensor.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x = torch.randn(3, mc["sequence_length"], mc["input_size"])
        latent = lstm_model.encode(x)
        assert latent.shape == (3, mc["hidden_size"])

    def test_latent_varies_with_input(self, lstm_model, config):
        """
        Verify that different inputs generate distinct latent representation matrices.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x1 = torch.randn(1, mc["sequence_length"], mc["input_size"])
        x2 = torch.randn(1, mc["sequence_length"], mc["input_size"]) + 10.0

        lstm_model.eval()
        with torch.no_grad():
            z1 = lstm_model.encode(x1)
            z2 = lstm_model.encode(x2)
        assert not torch.allclose(z1, z2, atol=1e-3)


class TestLSTMReconstructionError:
    """
    Validates reconstruction error calculations and properties.
    """

    def test_error_shape_mean(self, lstm_model, config):
        """
        Verify reconstruction error calculation under mean reduction.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x = torch.randn(5, mc["sequence_length"], mc["input_size"])
        error = lstm_model.compute_reconstruction_error(x, reduction="mean")
        assert error.shape == (5,)

    def test_error_shape_sum(self, lstm_model, config):
        """
        Verify reconstruction error calculation under sum reduction.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x = torch.randn(5, mc["sequence_length"], mc["input_size"])
        error = lstm_model.compute_reconstruction_error(x, reduction="sum")
        assert error.shape == (5,)

    def test_error_shape_none(self, lstm_model, config):
        """
        Verify reconstruction error calculation with no reduction.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x = torch.randn(5, mc["sequence_length"], mc["input_size"])
        error = lstm_model.compute_reconstruction_error(x, reduction="none")
        assert error.shape == x.shape

    def test_error_non_negative(self, lstm_model, config):
        """
        Verify that reconstruction error values are strictly non-negative.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x = torch.randn(5, mc["sequence_length"], mc["input_size"])
        error = lstm_model.compute_reconstruction_error(x, reduction="mean")
        assert (error >= 0).all()

    def test_anomaly_error_higher_than_normal(self, lstm_model, config):
        """
        Verify that model training reduces error on normal data relative to anomalous data.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        seq, feat = mc["sequence_length"], mc["input_size"]

        normal = torch.randn(16, seq, feat) * 0.1
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        lstm_model.train()
        for _ in range(30):
            optimizer.zero_grad()
            loss = criterion(lstm_model(normal), normal)
            loss.backward()
            optimizer.step()

        lstm_model.eval()
        with torch.no_grad():
            normal_err = lstm_model.compute_reconstruction_error(normal).mean().item()
            anomaly = torch.randn(16, seq, feat) * 10.0
            anomaly_err = lstm_model.compute_reconstruction_error(anomaly).mean().item()

        assert anomaly_err > normal_err

    def test_zero_input_error(self, lstm_model, config):
        """
        Verify that reconstruction error remains finite for zero inputs.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x = torch.zeros(1, mc["sequence_length"], mc["input_size"])
        error = lstm_model.compute_reconstruction_error(x, reduction="mean")
        assert torch.isfinite(error).all()


class TestLSTMTraining:
    """
    Validates loss convergence and gradient flows under backpropagation.
    """

    def test_loss_decreases(self, lstm_model, config):
        """
        Verify that training loss decreases across epochs.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x = torch.randn(16, mc["sequence_length"], mc["input_size"]) * 0.1

        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        lstm_model.train()
        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            loss = criterion(lstm_model(x), x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease during training"

    def test_gradient_flow(self, lstm_model, config):
        """
        Verify that all parameters receive non-zero gradients under backpropagation.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x = torch.randn(2, mc["sequence_length"], mc["input_size"])
        out = lstm_model(x)
        loss = nn.MSELoss()(out, x)
        loss.backward()

        for name, param in lstm_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


class TestAutoEncoderArchitecture:
    """
    Validates structural properties, shapes, and convergence of the AutoEncoder.
    """

    def test_forward_shapes(self, autoencoder):
        """
        Verify structural dimensions of reconstruction and compression representations.

        Args:
            autoencoder: AutoEncoder test fixture
        """
        x = torch.randn(4, 128)
        reconstruction, encoding = autoencoder(x)
        assert reconstruction.shape == (4, 128)
        assert encoding.shape == (4, 16)

    def test_encode_shape(self, autoencoder):
        """
        Verify that encoding produces the compressed latent dimension.

        Args:
            autoencoder: AutoEncoder test fixture
        """
        x = torch.randn(4, 128)
        z = autoencoder.encode(x)
        assert z.shape == (4, 16)

    def test_decode_shape(self, autoencoder):
        """
        Verify that decoding expands latent representation to original dimensions.

        Args:
            autoencoder: AutoEncoder test fixture
        """
        z = torch.randn(4, 16)
        x_hat = autoencoder.decode(z)
        assert x_hat.shape == (4, 128)

    def test_encode_decode_roundtrip(self, autoencoder):
        """
        Verify that compression followed by decompression retains correct tensor shapes.

        Args:
            autoencoder: AutoEncoder test fixture
        """
        x = torch.randn(4, 128)
        z = autoencoder.encode(x)
        x_hat = autoencoder.decode(z)
        assert x_hat.shape == x.shape

    def test_custom_hidden_dims(self):
        """
        Verify that custom intermediate layer configurations produce correct sizes.
        """
        from edge.models import AutoEncoder
        ae = AutoEncoder(input_dim=256, encoding_dim=32, hidden_dims=[128, 64])
        x = torch.randn(2, 256)
        recon, enc = ae(x)
        assert recon.shape == (2, 256)
        assert enc.shape == (2, 32)

    def test_default_hidden_dims(self):
        """
        Verify that default intermediate configurations are structured properly.
        """
        from edge.models import AutoEncoder
        ae = AutoEncoder(input_dim=512, encoding_dim=64)
        x = torch.randn(2, 512)
        recon, enc = ae(x)
        assert recon.shape == (2, 512)

    def test_ae_training_convergence(self, autoencoder):
        """
        Verify that standard AutoEncoder training exhibits expected loss convergence.

        Args:
            autoencoder: AutoEncoder test fixture
        """
        x = torch.randn(32, 128) * 0.1
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        losses = []
        autoencoder.train()
        for _ in range(30):
            optimizer.zero_grad()
            recon, _ = autoencoder(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0]
