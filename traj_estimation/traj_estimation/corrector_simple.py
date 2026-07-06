import torch
from torch import nn

class RNNCorrectionModel(nn.Module):
    """
    One-block RNN correction model for EKF improvement.

    Architecture
    ------------
    Input sequence:
        x: shape (B, T, 13)

    RNN block:
        LSTM or GRU

    Readout:
        fully connected MLP

    Output:
        correction: shape (B, 6)

    Output components:
        [d_lat, d_lon, d_alt, d_rot_x, d_rot_y, d_rot_z]
    """

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 128,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0.0,
        readout_hidden_dim: int = 64,
        output_dim: int = 6,
    ):
        super().__init__()

        rnn_type = rnn_type.lower()

        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(
                f"Unsupported rnn_type={rnn_type!r}. Expected 'lstm' or 'gru'."
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.output_dim = output_dim

        rnn_dropout = dropout if num_layers > 1 else 0.0

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, readout_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, 13).

        Returns
        -------
        correction : torch.Tensor
            Shape (B, 6).
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input x with shape (B, T, input_dim), got {x.shape}."
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got x.shape[-1]={x.shape[-1]}."
            )

        rnn_out, hidden = self.rnn(x)

        # Use final timestep hidden representation.
        #
        # rnn_out shape: (B, T, hidden_dim)
        final_features = rnn_out[:, -1, :]

        correction = self.readout(final_features)

        return correction


def make_lstm_correction_model(
    input_dim: int = 13,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.0,
    readout_hidden_dim: int = 64,
) -> RNNCorrectionModel:
    return RNNCorrectionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        rnn_type="lstm",
        dropout=dropout,
        readout_hidden_dim=readout_hidden_dim,
        output_dim=6,
    )


def make_gru_correction_model(
    input_dim: int = 13,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.0,
    readout_hidden_dim: int = 64,
) -> RNNCorrectionModel:
    return RNNCorrectionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        rnn_type="gru",
        dropout=dropout,
        readout_hidden_dim=readout_hidden_dim,
        output_dim=6,
    )

