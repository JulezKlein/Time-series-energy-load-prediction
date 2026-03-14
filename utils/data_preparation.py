from dotenv import load_dotenv
import os
from datetime import date
from pathlib import Path

import pandas as pd
from utils.get_features import get_matched_weather_load_data
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import torch
import numpy as np
from typing import Union

def prepare_data_for_modeling(
    features: list,
    target: Union[str, list],
    scale_features: list,
    save_scaler: bool = True,
    save_data: bool = True,
    reprocess_data: bool = False,
    train_start_date: Union[date, str] = date(2018, 1, 1),
    train_end_date: Union[date, str] = date(2023, 12, 31),
    val_start_date: Union[date, str] = date(2024, 1, 1),
    val_end_date: Union[date, str] = date(2024, 12, 31),
    test_start_date: Union[date, str] = date(2025, 1, 1),
    test_end_date: Union[date, str] = date(2025, 12, 31),
    production_data: bool = False
):
    load_dotenv()

    entsoe_api_key = os.getenv("ENTSOE_API_KEY")

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    models_dir = project_root / "models"

    def _to_date(value: Union[date, str], name: str) -> date:
        resolved_date = pd.Timestamp(value).date()
        if resolved_date is None:
            raise ValueError(f"Invalid value for {name}: {value}")
        return resolved_date

    train_start = _to_date(train_start_date, "train_start_date")
    train_end = _to_date(train_end_date, "train_end_date")
    val_start = _to_date(val_start_date, "val_start_date")
    val_end = _to_date(val_end_date, "val_end_date")
    test_start = _to_date(test_start_date, "test_start_date")
    test_end = _to_date(test_end_date, "test_end_date")

    if train_start > train_end:
        raise ValueError("train_start_date must be before or equal to train_end_date")
    if val_start > val_end:
        raise ValueError("val_start_date must be before or equal to val_end_date")
    if test_start > test_end:
        raise ValueError("test_start_date must be before or equal to test_end_date")

    train_out_path = data_dir / f"train_data_{train_start:%Y%m%d}_{train_end:%Y%m%d}.parquet"
    val_out_path = data_dir / f"val_data_{val_start:%Y%m%d}_{val_end:%Y%m%d}.parquet"
    test_out_path = data_dir / f"test_data_{test_start:%Y%m%d}_{test_end:%Y%m%d}.parquet"

    if not train_out_path.exists() or reprocess_data:
        train_df = get_matched_weather_load_data(start_date=train_start, end_date=train_end,
            country_code="DE", locations=3, api_key=entsoe_api_key,
            align_calendar_to_target_day=True, production_data=production_data)
        if save_data:
            train_df.to_parquet(train_out_path)
    else:
        train_df = pd.read_parquet(train_out_path)

    if not val_out_path.exists() or reprocess_data:
        val_df = get_matched_weather_load_data(start_date=val_start, end_date=val_end,
            country_code="DE", locations=3, api_key=entsoe_api_key,
            align_calendar_to_target_day=True, production_data=production_data)
        if save_data:
            val_df.to_parquet(val_out_path)
    else:
        val_df = pd.read_parquet(val_out_path)

    if not test_out_path.exists() or reprocess_data:
        test_df = get_matched_weather_load_data(start_date=test_start, end_date=test_end,
            country_code="DE", locations=3, api_key=entsoe_api_key,
            align_calendar_to_target_day=True, production_data=production_data)
        if save_data:
            test_df.to_parquet(test_out_path)
    else:
        test_df = pd.read_parquet(test_out_path)

    if isinstance(target, str):
        target_columns = [target]
        return_series_target = True
    elif isinstance(target, list):
        if not target:
            raise ValueError("target list cannot be empty")
        target_columns = target
        return_series_target = False
    else:
        raise TypeError("target must be a string column name or a list of column names")

    # Features + Target (actual load data of the day in MW)
    X_train = train_df[features]
    y_train = train_df[target_columns]

    X_val = val_df[features]
    y_val = val_df[target_columns]

    X_test = test_df[features]
    y_test = test_df[target_columns]

    # Backward compatibility: keep Series output when a single target string is provided.
    if return_series_target:
        y_train = y_train.iloc[:, 0]
        y_val = y_val.iloc[:, 0]
        y_test = y_test.iloc[:, 0]
    
    scaler = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), scale_features)
    ],
    remainder="passthrough"
)

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, models_dir / "feature_scaler.joblib") if save_scaler else None
    return {'train_df': train_df, 'val_df': val_df, 'test_df': test_df, 'X_train_scaled': X_train_scaled, 'X_train': X_train, 'y_train': y_train, 'X_val_scaled': X_val_scaled, 'X_val': X_val, 'y_val': y_val, 'X_test_scaled': X_test_scaled, 'X_test': X_test, 'y_test': y_test}, {'scaler_mean': scaler.named_transformers_["num"].mean_, 'scaler_std': scaler.named_transformers_["num"].scale_}


def create_torch_dataset(X: np.ndarray, y: np.ndarray, window_size: int):
    """Transform a time series into a prediction dataset
    
    Args:
        X: A numpy array of features, first dimension is the time steps
        y: A numpy array of targets, first dimension is the time steps
        window_size: Size of window for prediction
    """
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    if y_np.ndim == 1:
        single_target = True
    elif y_np.ndim == 2:
        single_target = False
    else:
        raise ValueError("y must be a 1D array-like or a 2D array-like for multi-target data")

    X_windows, y_windows = [], []
    for i in range(len(X_np) - window_size + 1):
        feature_window = X_np[i:i + window_size]
        if single_target:
            target_value = y_np[i + window_size - 1]
        else:
            target_value = y_np[i + window_size - 1, :]
        X_windows.append(feature_window)
        y_windows.append(target_value)

    X_windows = np.asarray(X_windows)
    y_windows = np.asarray(y_windows)

    return torch.tensor(X_windows, dtype=torch.float32), torch.tensor(y_windows, dtype=torch.float32)