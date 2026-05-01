"""
Model training utilities.

Provides sequence windowing/scaling, DL training loop with early stopping,
DL inference, and time-series cross-validation for tree-based models.
"""

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader

from src.config import DEVICE, HAS_LIGHTGBM
from src.models.sequence_models import SequenceBundle, WindowedDataset

# Conditional import for LightGBM
if HAS_LIGHTGBM:
    from lightgbm import LGBMRegressor


def build_scaled_sequences(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_col: str,
    train_size: int,
    val_size: int,
    sequence_length: int,
) -> SequenceBundle:
    """Scale features/targets and build sliding-window sequences for DL models."""
    from sklearn.preprocessing import StandardScaler

    full_X = frame[feature_columns].to_numpy(dtype=float)
    full_y = frame[[target_col]].to_numpy(dtype=float)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(full_X[:train_size])
    target_scaler.fit(full_y[:train_size])

    X_all_scaled = feature_scaler.transform(full_X)
    y_all_scaled = target_scaler.transform(full_y).reshape(-1)

    X_seq, y_seq, positions = [], [], []
    for end_idx in range(sequence_length - 1, len(frame)):
        start_idx = end_idx - sequence_length + 1
        X_seq.append(X_all_scaled[start_idx : end_idx + 1])
        y_seq.append(y_all_scaled[end_idx])
        positions.append(end_idx)

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.float32)
    positions = np.asarray(positions, dtype=int)

    return SequenceBundle(
        X_all_scaled=X_all_scaled,
        y_all_scaled=y_all_scaled,
        X_sequences=X_seq,
        y_sequences=y_seq,
        sequence_target_positions=positions,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        train_mask=positions < train_size,
        val_mask=(positions >= train_size) & (positions < train_size + val_size),
        test_mask=positions >= train_size + val_size,
    )


def train_sequence_model(
    model: nn.Module,
    bundle: SequenceBundle,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
) -> Tuple[nn.Module, pd.DataFrame]:
    """Train a DL model with early stopping and return training history."""
    train_dataset = WindowedDataset(
        bundle.X_sequences[bundle.train_mask], bundle.y_sequences[bundle.train_mask],
    )
    val_dataset = WindowedDataset(
        bundle.X_sequences[bundle.val_mask], bundle.y_sequences[bundle.val_mask],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0
    history_rows = []

    for epoch in range(1, max_epochs + 1):
        # Training pass
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation pass
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(val_losses))
        history_rows.append({
            "epoch": epoch, "train_loss": mean_train_loss, "val_loss": mean_val_loss,
        })

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history_rows)


def predict_sequence_model(model: nn.Module, X_seq: np.ndarray) -> np.ndarray:
    """Run inference on a batch of sequences."""
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(X_seq, dtype=torch.float32, device=DEVICE)
        preds = model(tensor).detach().cpu().numpy()
    return preds


def time_series_cv_search(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_col: str,
    param_grid: Sequence[Dict],
    random_seed: int,
) -> Tuple[object, pd.DataFrame]:
    """Time-series cross-validation hyperparameter search for tree models."""
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X = train_val_df[feature_columns]
    y = train_val_df[target_col]
    tscv = TimeSeriesSplit(n_splits=4)
    cv_rows = []
    best_params = None
    best_rmse = float("inf")

    def build_model(params: Dict):
        if model_name == "Random Forest":
            return RandomForestRegressor(random_state=random_seed, n_jobs=-1, **params)
        if model_name == "LightGBM":
            if not HAS_LIGHTGBM:
                raise ImportError("LightGBM is not installed.")
            return LGBMRegressor(
                objective="regression", random_state=random_seed,
                verbose=-1, n_jobs=-1, **params,
            )
        raise ValueError(f"Unsupported tree model: {model_name}")

    for params in param_grid:
        fold_rmses = []
        for fold_id, (fit_idx, hold_idx) in enumerate(tscv.split(X), start=1):
            model = build_model(params)
            model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            fold_pred = model.predict(X.iloc[hold_idx])
            try:
                fold_rmse = mean_squared_error(y.iloc[hold_idx], fold_pred, squared=False)
            except TypeError:
                fold_rmse = np.sqrt(mean_squared_error(y.iloc[hold_idx], fold_pred))
            fold_rmses.append(fold_rmse)
            cv_rows.append({
                "model": model_name, "fold": fold_id, "params": params, "rmse": fold_rmse,
            })

        avg_rmse = float(np.mean(fold_rmses))
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_params = params

    best_model = build_model(best_params)
    best_model.fit(X, y)
    return best_model, pd.DataFrame(cv_rows)
