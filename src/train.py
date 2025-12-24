import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import config
from dataset import FlashFloodDataset
from model import EALSTM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class WeightedNSELoss(nn.Module):
    """
    Weighted NSE Loss Function.

    Objective:
    Minimize the weighted squared error normalized by the basin's natural variance.
    This effectively maximizes the NSE score while prioritizing high-flow events.

    Formula:
        Loss = Mean( (Weights * (Pred - Obs)^2) / (Basin_Variance + epsilon) )

    Where:
        Weights = clamp(1 + Observed, min=0.1)
    """

    def __init__(self):
        super(WeightedNSELoss, self).__init__()
        self.eps = 1e-6

    def forward(self, predictions, targets, basin_variances):
        """
        Args:
            predictions: Model outputs [Batch_Size]
            targets: Observed values [Batch_Size]
            basin_variances: Variance of the target variable for each basin [Batch_Size]
        """
        weights = torch.clamp(1 + targets, min=0.1)

        squared_errors = (targets - predictions) ** 2
        weighted_errors = weights * squared_errors

        safe_variance = torch.clamp(basin_variances, min=0.01)
        safe_variance = torch.nan_to_num(safe_variance, nan=1.0)

        normalized_errors = weighted_errors / (safe_variance + self.eps)

        return torch.mean(normalized_errors)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    Updates model weights based on the Weighted NSE Loss.
    """
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)

    for x_dyn, x_stat, y, basin_var, _ in pbar:
        x_dyn, x_stat = x_dyn.to(device), x_stat.to(device)
        y, basin_var = y.to(device), basin_var.to(device)

        optimizer.zero_grad()

        preds = model(x_dyn, x_stat).squeeze()
        loss = criterion(preds, y, basin_var)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    """
    Evaluates model performance on validation set.
    Calculates Loss and Hydrological Metrics (CSI, POD, FAR) using
    basin-specific flood thresholds (Q_5y).
    """
    model.eval()
    total_loss = 0.0

    hits = 0
    misses = 0
    false_alarms = 0

    with torch.no_grad():
        for x_dyn, x_stat, y, basin_var, threshold in loader:
            x_dyn, x_stat = x_dyn.to(device), x_stat.to(device)
            y, basin_var = y.to(device), basin_var.to(device)
            threshold = threshold.to(device)

            preds = model(x_dyn, x_stat).squeeze()

            loss = criterion(preds, y, basin_var)
            total_loss += loss.item()

            pred_flood = preds > threshold
            true_flood = y > threshold

            hits += (pred_flood & true_flood).sum().item()
            misses += (~pred_flood & true_flood).sum().item()
            false_alarms += (pred_flood & ~true_flood).sum().item()

    epsilon = 1e-6
    pod = hits / (hits + misses + epsilon)
    far = false_alarms / (hits + false_alarms + epsilon)
    csi = hits / (hits + misses + false_alarms + epsilon)

    avg_loss = total_loss / len(loader)

    return {
        "loss": avg_loss,
        "csi": csi,
        "pod": pod,
        "far": far
    }

def plot_training_history(history, save_dir):
    """
    Plots the training progression:
    1. Loss (Train vs Val)
    2. Hydrological Metrics (CSI, POD, FAR)
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue', marker='o')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='orange', marker='o')
    ax1.set_title('Training vs Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (Weighted NSE)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['csi'], label='CSI (Success)', color='green', marker='s')
    ax2.plot(epochs, history['pod'], label='POD (Detection)', color='purple', marker='^')
    ax2.plot(epochs, history['far'], label='FAR (False Alarm)', color='red', marker='x')
    ax2.set_title('Hydrological Metrics (Validation)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score (0-1)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    save_path = save_dir / "training_history.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Training graph saved to: {save_path}")


def main():
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    EPOCHS = 30
    HIDDEN_DIM = 256
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Starting training on device: {DEVICE}")

    train_ds = FlashFloodDataset(mode='train', seq_length=270)
    val_ds = FlashFloodDataset(mode='val', seq_length=270)

    sample_x_dyn, sample_x_stat, _, _, _ = train_ds[0]
    input_dim_dyn = sample_x_dyn.shape[1]
    input_dim_stat = sample_x_stat.shape[0]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = EALSTM(input_dim_dyn, input_dim_stat, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = WeightedNSELoss()

    best_val_loss = float('inf')

    history = {
        'train_loss': [],
        'val_loss': [],
        'csi': [],
        'pod': [],
        'far': []
    }

    logger.info("Starting Training Loop...")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        metrics = validate(model, val_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(metrics['loss'])
        history['csi'].append(metrics['csi'])
        history['pod'].append(metrics['pod'])
        history['far'].append(metrics['far'])

        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['loss']:.4f} | "
            f"CSI: {metrics['csi']:.3f} | POD: {metrics['pod']:.3f} | FAR: {metrics['far']:.3f}"
        )

        if metrics['loss'] < best_val_loss:
            best_val_loss = metrics['loss']
            torch.save(model.state_dict(), config.MODELS_DIR / "best_model.pth")
            logger.info(f"New best model saved! (CSI: {metrics['csi']:.3f})")

    plot_training_history(history, config.PROCESSED_DATA_DIR)

    logger.info("Training complete.")
if __name__ == "__main__":
    main()