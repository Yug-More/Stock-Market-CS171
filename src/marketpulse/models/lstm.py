"""A one-layer sequence LSTM trained on CPU with validation early stopping."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from marketpulse.evaluation.metrics import rmse


@dataclass(frozen=True)
class LSTMTrainingResult:
    """A restored best-validation model and the loss recorded at each epoch."""

    model: nn.Module
    hidden_size: int
    epochs_trained: int
    best_epoch: int
    history: tuple[dict[str, float], ...]
    validation_rmse: float


class ReturnLSTM(nn.Module):
    """LSTM over a lookback window, read out at the final timestep."""

    def __init__(self, n_features: int, hidden_size: int) -> None:
        super().__init__()
        if hidden_size < 1 or n_features < 1:
            raise ValueError("n_features and hidden_size must be positive")
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(sequences)
        return self.head(output[:, -1, :]).squeeze(-1)


def train_return_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    hidden_size: int,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
) -> LSTMTrainingResult:
    """Train on training sequences and restore the weights with the best validation MSE.

    Batch order inside the training set is shuffled. The chronological split
    is unchanged. PyTorch does not guarantee bitwise CPU reproducibility
    across library versions even after seeding.
    """

    _set_seed(seed)
    model = ReturnLSTM(n_features=x_train.shape[-1], hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    generator = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    best_epoch = 0
    waits = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = generator.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            prediction = model(_tensor(x_train[batch]))
            loss = loss_function(prediction, _tensor(y_train[batch]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        validation_mse = _mean_squared_error(model, x_validation, y_validation)
        history.append({"epoch": float(epoch), "validation_mse": validation_mse})
        if validation_mse < best_validation:
            best_validation = validation_mse
            best_epoch = epoch
            waits = 0
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
        else:
            waits += 1
            if waits >= patience:
                break

    if best_state is None:
        raise RuntimeError("LSTM training did not record a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    validation_prediction = predict_return_lstm(model, x_validation)
    return LSTMTrainingResult(
        model=model,
        hidden_size=hidden_size,
        epochs_trained=len(history),
        best_epoch=best_epoch,
        history=tuple(history),
        validation_rmse=rmse(y_validation, validation_prediction),
    )


def predict_return_lstm(model: nn.Module, sequences: np.ndarray) -> np.ndarray:
    """Predict in evaluation mode without updating the model."""

    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequences), 1024):
            prediction = model(_tensor(sequences[start : start + 1024]))
            chunks.append(prediction.detach().cpu().numpy())
    return np.concatenate(chunks)


def select_return_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    hidden_sizes: tuple[int, ...],
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
) -> tuple[LSTMTrainingResult, dict[int, float]]:
    """Train each predeclared hidden size and keep the best validation RMSE."""

    if len(hidden_sizes) == 0:
        raise ValueError("hidden_sizes must contain at least one candidate")
    trained: list[LSTMTrainingResult] = []
    for hidden_size in hidden_sizes:
        trained.append(
            train_return_lstm(
                x_train,
                y_train,
                x_validation,
                y_validation,
                hidden_size=hidden_size,
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed,
            )
        )
    scores = {result.hidden_size: result.validation_rmse for result in trained}
    selected = min(trained, key=lambda result: (result.validation_rmse, result.hidden_size))
    return selected, scores


def _mean_squared_error(model: nn.Module, sequences: np.ndarray, targets: np.ndarray) -> float:
    prediction = predict_return_lstm(model, sequences)
    return float(np.mean((prediction - targets) ** 2))


def _tensor(values: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
