import torch
import numpy as np


def train_one_epoch_multiday(
    epoch,
    best_val_mse,
    best_epoch,
    epochs_without_improvement,
    model,
    loss_fn,
    optimizer,
    window_size,
    features,
    targets,
    best_model_path,
    target_mean,
    target_std,
    scheduler,
    min_improvement,
    device,
    early_stopping_patience,
    validate_every,
    training_loader,
    validation_loader,
):

    model.train()
    train_mse_sum = 0.0
    train_mae_sum = 0.0
    train_elements = 0

    for X_batch, y_batch in training_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_elements = y_batch.numel()
        train_elements += batch_elements
        train_mse_sum += torch.sum((y_pred - y_batch) ** 2).item()
        train_mae_sum += torch.sum(torch.abs(y_pred - y_batch)).item()

    train_mse = train_mse_sum / train_elements
    train_rmse = np.sqrt(train_mse)
    train_mae = train_mae_sum / train_elements

    if epoch % validate_every != 0:
        return best_val_mse, best_epoch, epochs_without_improvement, False

    model.eval()
    val_mse_sum = 0.0
    val_mae_sum = 0.0
    val_elements = 0
    val_pred_sum = 0.0
    val_pred_sq_sum = 0.0

    with torch.no_grad():
        for X_batch, y_batch in validation_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)

            batch_elements = y_batch.numel()
            val_elements += batch_elements
            val_mse_sum += torch.sum((y_pred - y_batch) ** 2).item()
            val_mae_sum += torch.sum(torch.abs(y_pred - y_batch)).item()
            val_pred_sum += torch.sum(y_pred).item()
            val_pred_sq_sum += torch.sum(y_pred**2).item()

    val_mse = val_mse_sum / val_elements
    val_rmse = np.sqrt(val_mse)
    val_mae = val_mae_sum / val_elements

    val_pred_mean = val_pred_sum / val_elements
    val_pred_var = max((val_pred_sq_sum / val_elements) - (val_pred_mean**2), 0.0)
    val_pred_std = np.sqrt(val_pred_var)

    scheduler.step(val_mse)

    improved = val_mse < (best_val_mse - min_improvement)
    if improved:
        best_val_mse = val_mse
        best_epoch = epoch
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "target_scaled": True,
                "target_mean": target_mean.to_numpy().tolist(),
                "target_std": target_std.to_numpy().tolist(),
                "window_size": window_size,
                "features": features,
                "targets": targets,
            },
            best_model_path,
        )
        epochs_without_improvement = 0
        checkpoint_note = " [saved best]"
    else:
        epochs_without_improvement += 1
        checkpoint_note = ""

    print(
        f"Epoch {epoch}: train RMSE {train_rmse:.4f}, train MAE {train_mae:.4f}, "
        f"val RMSE {val_rmse:.4f}, val MAE {val_mae:.4f}, val MSE {val_mse:.4f}, "
        f"val pred std {val_pred_std:.2f}, lr {optimizer.param_groups[0]['lr']:.2e}{checkpoint_note}"
    )

    if val_pred_std < 0.05:
        print("Warning: validation prediction std is very low; model may be collapsing to near-constant output.")

    should_stop = epochs_without_improvement >= early_stopping_patience
    if should_stop:
        print(f"Early stopping at epoch {epoch}. No sufficient improvement for {early_stopping_patience} validation checks.")

    return best_val_mse, best_epoch, epochs_without_improvement, should_stop
