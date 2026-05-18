import os
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.graph_objects as go

from utils.get_features import get_matched_weather_load_data
from utils.lstm_model import LSTMForecaster


DEFAULT_FEATURES_TODAY = ["Temp", "Min Temp", "Max Temp", "load"]
DEFAULT_FEATURES_TARGET_TIME = ["is_holiday", "dow_sin", "dow_cos", "month_sin", "month_cos"]
DEFAULT_FEATURES = DEFAULT_FEATURES_TODAY + DEFAULT_FEATURES_TARGET_TIME
DEFAULT_TARGETS = [f"load_t+{i}" for i in range(1, 8)]
DEFAULT_WINDOW_SIZE = 21
DEFAULT_LOCATION_COUNT = 3

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


@st.cache_resource(show_spinner=False)
def load_checkpoint(model_path: Path) -> dict:
    checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        return {
            "model_state_dict": checkpoint,
            "target_scaled": False,
        }
    return checkpoint


@st.cache_resource(show_spinner=False)
def load_scaler(scaler_path: Path):
    return joblib.load(scaler_path)


def infer_model_shape(checkpoint: dict, features: list[str], targets: list[str]) -> tuple[int, int, int, int]:
    state_dict = checkpoint["model_state_dict"]

    if "fc.weight" not in state_dict:
        raise KeyError("Checkpoint does not contain fc.weight.")

    fc_weight = state_dict["fc.weight"]
    if isinstance(fc_weight, torch.Tensor):
        output_size, hidden_size = fc_weight.shape
    else:
        output_size, hidden_size = np.asarray(fc_weight).shape

    num_layers = sum(1 for k in state_dict.keys() if k.startswith("lstm.weight_ih_l"))
    input_size = len(features)

    if output_size != len(targets):
        st.warning(
            f"Checkpoint output size ({output_size}) differs from TARGETS length ({len(targets)}). "
            "Using checkpoint output size and truncating/expanding labels automatically."
        )

    return input_size, hidden_size, max(num_layers, 1), output_size


@st.cache_resource(show_spinner=False)
def build_model(model_path: Path, _checkpoint: dict, features: list[str], targets: list[str]) -> tuple[LSTMForecaster, int, int]:
    input_size, hidden_size, num_layers, output_size = infer_model_shape(_checkpoint, features, targets)

    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=0.0,
    )

    model.load_state_dict(_checkpoint["model_state_dict"])
    model.eval()

    window_size = int(_checkpoint.get("window_size", DEFAULT_WINDOW_SIZE))
    return model, window_size, output_size


def get_latest_production_frame(
    start_date: date,
    end_date: date,
    locations: int,
) -> pd.DataFrame:
    frame = get_matched_weather_load_data(
        start_date=start_date,
        end_date=end_date,
        country_code="DE",
        locations=locations,
        api_key=os.getenv("ENTSOE_API_KEY"),
        align_calendar_to_target_day=True,
        production_data=True,
    )
    frame = frame.sort_values("time").reset_index(drop=True)
    # train_out_path = PROJECT_ROOT / "data" / "latest_production_frame.parquet"
    # frame.to_parquet(train_out_path)
    return frame


def load_local_production_frame(
    parquet_path: Path,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Local parquet file not found: {parquet_path}")

    frame = pd.DataFrame(pd.read_parquet(parquet_path))
    if "time" not in frame.columns:
        raise ValueError("Local parquet file must include a 'time' column.")

    frame["time"] = pd.DatetimeIndex(pd.to_datetime(frame["time"])).normalize()

    if start_date is not None:
        frame = frame.loc[frame["time"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        frame = frame.loc[frame["time"] <= pd.Timestamp(end_date)]

    sort_idx = np.argsort(frame["time"].to_numpy())
    frame = frame.iloc[sort_idx].reset_index(drop=True)
    return frame


def run_forecast(
    model: LSTMForecaster,
    scaler,
    data_df: pd.DataFrame,
    features: list[str],
    window_size: int,
    output_size: int,
    checkpoint: dict,
    target_names: list[str],
) -> pd.DataFrame:
    missing_features = [col for col in features if col not in data_df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    if len(data_df) < window_size:
        raise ValueError(f"Need at least {window_size} rows for prediction, but got {len(data_df)} rows.")

    X_scaled = scaler.transform(data_df[features])
    X_window = X_scaled[-window_size:]

    X_tensor = torch.tensor(X_window, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().reshape(-1)

    if bool(checkpoint.get("target_scaled", False)):
        mean_vals = np.asarray(checkpoint.get("target_mean", [0.0] * output_size), dtype=float)
        std_vals = np.asarray(checkpoint.get("target_std", [1.0] * output_size), dtype=float)
        if mean_vals.shape[0] != output_size or std_vals.shape[0] != output_size:
            raise ValueError("Checkpoint target scaling metadata shape does not match model output size.")
        y_pred = (y_pred * std_vals) + mean_vals

    last_timestamp = pd.to_datetime(data_df["time"].iloc[-1])
    last_date = last_timestamp.floor("D")
    forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=output_size, freq="D")

    if len(target_names) != output_size:
        target_names = [f"load_t+{i}" for i in range(1, output_size + 1)]

    return pd.DataFrame(
        {
            "time": forecast_dates,
            "target": target_names,
            "predicted_load": y_pred,
        }
    )


def make_extended_plot(
    last_month_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    last_week_hypothetical_df: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=last_month_df["time"],
            y=last_month_df["load"],
            mode="lines+markers",
            name="Historical load",
            line={"color": "#1f77b4", "width": 2},
        )
    )

    anchor_x = [last_month_df["time"].iloc[-1]] + forecast_df["time"].tolist()
    anchor_y = [last_month_df["load"].iloc[-1]] + forecast_df["predicted_load"].tolist()

    fig.add_trace(
        go.Scatter(
            x=anchor_x,
            y=anchor_y,
            mode="lines+markers",
            name="Predicted load",
            line={"color": "#ff7f0e", "width": 3, "dash": "dash"},
        )
    )

    if not last_week_hypothetical_df.empty:
        fig.add_trace(
            go.Scatter(
                x=last_week_hypothetical_df["time"],
                y=last_week_hypothetical_df["hypothetical_predicted_load"],
                mode="markers+lines",
                name="Hypothetical prediction (last week)",
                line={"color": "#2ca02c", "width": 2, "dash": "dot"},
                marker={"size": 8, "symbol": "diamond"},
            )
        )

    fig.update_layout(
        title="Last Month Load + Multi-Day Forecast",
        xaxis_title="Date",
        yaxis_title="Load (MW)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )

    return fig


def make_last_week_hypothetical_table(
    model: LSTMForecaster,
    scaler,
    data_df: pd.DataFrame,
    features: list[str],
    window_size: int,
    checkpoint: dict,
) -> pd.DataFrame:
    data_df = data_df.sort_values("time").reset_index(drop=True)
    needed_columns = ["time", "load"] + features
    missing = [col for col in needed_columns if col not in data_df.columns]
    if missing:
        raise ValueError(f"Missing columns for hypothetical predictions: {missing}")

    if len(data_df) < (window_size + 7):
        # We need one prior day window per hypothetical next-day prediction.
        usable = max(0, len(data_df) - window_size)
        days_to_compute = min(7, usable)
    else:
        days_to_compute = 7

    if days_to_compute == 0:
        return pd.DataFrame(columns=["time", "actual_load", "hypothetical_predicted_load", "abs_error"])

    X_scaled = scaler.transform(data_df[features])

    output_size = len(checkpoint.get("target_mean", [])) if bool(checkpoint.get("target_scaled", False)) else None
    mean_vals = np.asarray(checkpoint.get("target_mean", []), dtype=float)
    std_vals = np.asarray(checkpoint.get("target_std", []), dtype=float)

    load_numeric = pd.Series(pd.to_numeric(data_df["load"], errors="coerce"), index=data_df.index)
    rows = []
    start_target_idx = len(data_df) - days_to_compute
    for target_idx in range(start_target_idx, len(data_df)):
        window_end_idx = target_idx - 1
        window_start_idx = window_end_idx - window_size + 1
        if window_start_idx < 0:
            continue

        X_window = X_scaled[window_start_idx : window_end_idx + 1]
        X_tensor = torch.tensor(X_window, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_vec = model(X_tensor).cpu().numpy().reshape(-1)

        pred_next_day = float(pred_vec[0])
        if bool(checkpoint.get("target_scaled", False)):
            if output_size is None or mean_vals.size == 0 or std_vals.size == 0:
                raise ValueError("Checkpoint target scaling metadata is missing.")
            if mean_vals.shape[0] != pred_vec.shape[0] or std_vals.shape[0] != pred_vec.shape[0]:
                raise ValueError("Checkpoint target scaling metadata shape does not match model output size.")
            pred_next_day = float((pred_vec[0] * std_vals[0]) + mean_vals[0])

        actual_value = float(load_numeric.iloc[target_idx])
        rows.append(
            {
                "time": pd.to_datetime(data_df.loc[target_idx, "time"]),
                "actual_load": actual_value,
                "hypothetical_predicted_load": pred_next_day,
                "abs_error": abs(actual_value - pred_next_day),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Energy Multi-Day Forecast", layout="wide")
    st.title("Energy Load Forecast Dashboard")
    st.caption("Uses production feature data and a trained multi-day LSTM checkpoint.")

    today = date.today()
    default_end = today - timedelta(days=1)
    default_start = default_end - timedelta(days=120)

    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input(
            "Model path",
            str(MODELS_DIR / "production_lstm_multiday_load_forecaster.pt"),
        )
        scaler_path = st.text_input(
            "Scaler path",
            str(MODELS_DIR / "best_lstm_multiday_feature_scaler.joblib"),
        )

        data_source = st.radio(
            "Production data source",
            options=["ENTSO-E API available", "Load local parquet file"],
            index=1,
        )

        local_parquet_path = st.text_input(
            "Local parquet path",
            str(PROJECT_ROOT / "data" / "latest_production_frame.parquet"),
        )

        start_date = st.date_input("Data start", value=default_start)
        end_date = st.date_input("Data end", value=default_end)
        location_count = st.number_input("Number of weather locations", min_value=1, max_value=6, value=DEFAULT_LOCATION_COUNT)

        run_btn = st.button("Load data and predict", type="primary")

    if not run_btn:
        st.info("Select settings and click 'Load data and predict'.")
        return

    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        return
    if not scaler_path.exists():
        st.error(f"Scaler file not found: {scaler_path}")
        return

    with st.spinner("Loading model, scaler, and production data..."):
        checkpoint = load_checkpoint(model_path)

        features = checkpoint.get("features", DEFAULT_FEATURES)
        target_names = checkpoint.get("targets", DEFAULT_TARGETS)

        model, window_size, output_size = build_model(model_path, checkpoint, features, target_names)
        scaler = load_scaler(scaler_path)

        if data_source == "Load local parquet file":
            production_df = load_local_production_frame(
                parquet_path=Path(local_parquet_path),
                start_date=start_date,
                end_date=end_date,
            )
        else:
            production_df = get_latest_production_frame(
                start_date=start_date,
                end_date=end_date,
                locations=int(location_count),
            )

        forecast_df = run_forecast(
            model=model,
            scaler=scaler,
            data_df=production_df,
            features=features,
            window_size=window_size,
            output_size=output_size,
            checkpoint=checkpoint,
            target_names=target_names,
        )

        last_week_hypothetical_df = make_last_week_hypothetical_table(
            model=model,
            scaler=scaler,
            data_df=production_df,
            features=features,
            window_size=window_size,
            checkpoint=checkpoint,
        )

    one_month_ago = pd.Timestamp(end_date) - pd.Timedelta(days=30)
    prediction_columns = ["time", "load"] + [c for c in features if c != "load"]
    recent_mask = production_df["time"] >= one_month_ago
    last_month_df = pd.DataFrame(production_df.loc[recent_mask, prediction_columns]).copy()
    last_month_df = last_month_df.sort_values("time")

    st.subheader("Forecast for Next Days")
    st.dataframe(forecast_df, width="stretch", hide_index=True)

    st.subheader("Historical Load Extended with Forecast")
    st.plotly_chart(
        make_extended_plot(
            last_month_df=last_month_df,
            forecast_df=forecast_df,
            last_week_hypothetical_df=last_week_hypothetical_df,
        ),
        width="stretch",
    )

    st.subheader("Last Month Input Data (Prediction Features)")
    st.dataframe(last_month_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
