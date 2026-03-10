import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_and_plot_model_sklearn(model, X_test_scaled, y_test, test_df):
    y_pred_test = model.predict(X_test_scaled)

    plot_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred_test
    }, index=test_df.loc[y_test.index, "time"] if "time" in test_df.columns else y_test.index)
    plot_df = plot_df.sort_index()

    pred_std = float(np.std(y_pred_test))
    if pred_std < 1e-6:
        print("Warning: model predictions are nearly constant. The model may be underfit or checkpoint/data may be mismatched.")

    print(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.2f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")

    plt.figure(figsize=(14, 5))
    plt.plot(plot_df.index, plot_df["actual"], label="Actual", linewidth=1.8)
    plt.plot(plot_df.index, plot_df["predicted"],
             label="Predicted", linewidth=1.5, alpha=0.85)
    plt.title("Test Set: Predicted vs Actual Load")
    plt.xlabel("Date")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(6, 6))
    # plt.scatter(plot_df["actual"], plot_df["predicted"], alpha=0.35)
    # axis_min = min(plot_df["actual"].min(), plot_df["predicted"].min())
    # axis_max = max(plot_df["actual"].max(), plot_df["predicted"].max())
    # plt.plot([axis_min, axis_max], [axis_min, axis_max], "r--", linewidth=1.2)
    # plt.title("Predicted vs Actual Scatter (Test)")
    # plt.xlabel("Actual Load (MW)")
    # plt.ylabel("Predicted Load (MW)")
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.show()


def evaluate_and_plot_model_torch(
    model,
    test_loader,
    y_test,
    test_df,
    device,
    window_size=1,
    target_mean=None,
    target_std=None,
):
    model.eval()
    y_pred_batches = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_pred_batches.append(model(X_batch).squeeze(-1).detach().cpu().numpy())

    if not y_pred_batches:
        raise ValueError("Test loader produced no batches.")

    y_pred_values = np.concatenate(y_pred_batches)
    y_true_aligned = y_test.iloc[window_size - 1:]

    if target_mean is not None and target_std is not None:
        if target_std <= 0:
            raise ValueError("target_std must be > 0 for inverse scaling.")
        y_pred_values = (y_pred_values * target_std) + target_mean
        y_true_values = (y_true_aligned.values * target_std) + target_mean
    else:
        y_true_values = y_true_aligned.values

    y_pred_test = pd.Series(data=y_pred_values, index=y_true_aligned.index)

    plot_df = pd.DataFrame(
        {
            "actual": y_true_values,
            "predicted": y_pred_test.values,
        },
        index=test_df.loc[y_true_aligned.index, "time"] if "time" in test_df.columns else y_true_aligned.index,
    ).sort_index()

    pred_std = float(plot_df["predicted"].std())
    if pred_std < 1e-6:
        print("Warning: model predictions are nearly constant. The loaded checkpoint appears collapsed to a mean prediction.")

    print(f"Test MAE: {mean_absolute_error(plot_df['actual'], plot_df['predicted']):.2f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(plot_df['actual'], plot_df['predicted'])):.2f}")
    # print("\nPredictions vs actual values (first 20 rows):")
    # print(plot_df[["actual", "predicted"]].head(20).to_string())

    plt.figure(figsize=(14, 5))
    plt.plot(plot_df.index, plot_df["actual"], label="Actual", linewidth=1.8)
    plt.plot(plot_df.index, plot_df["predicted"], label="Predicted", linewidth=1.5, alpha=0.85)
    plt.title("Test Set: Predicted vs Actual Load")
    plt.xlabel("Date")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return plot_df