import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import config
from dataset import FlashFloodDataset
from model import EALSTM
from preprocessing import DataScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_test_set():
    """
    Loads the best trained model and generates predictions for the TEST set (2005-2015).
    Saves the results to 'test_results.csv'.
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 256
    HIDDEN_DIM = 256

    logger.info(f"Running inference on device: {DEVICE}")

    test_ds = FlashFloodDataset(mode='test', seq_length=270)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    sample_x_dyn, sample_x_stat, _, _, _ = test_ds[0]
    input_dim_dyn = sample_x_dyn.shape[1]
    input_dim_stat = sample_x_stat.shape[0]

    model = EALSTM(input_dim_dyn, input_dim_stat, HIDDEN_DIM).to(DEVICE)
    model_path = config.MODELS_DIR / "best_model.pth"

    if not model_path.exists():
        logger.error("No trained model found! Run train.py first.")
        return

    logger.info(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    scaler = DataScaler()
    scaler.load_scaler()

    if config.TARGET_COL not in scaler.means:
        logger.error("Target column not found in scaler stats. Cannot un-normalize.")
        return

    target_mean = scaler.means[config.TARGET_COL]
    target_std = scaler.stds[config.TARGET_COL]

    logger.info(f"Scaler params for target: Mean={target_mean:.4f}, Std={target_std:.4f}")

    results = []

    logger.info("Generating predictions...")
    with torch.no_grad():
        for x_dyn, x_stat, y, _, _ in tqdm(test_loader, desc="Testing"):
            x_dyn, x_stat = x_dyn.to(DEVICE), x_stat.to(DEVICE)

            preds_norm = model(x_dyn, x_stat).squeeze()

            preds_norm = preds_norm.cpu().numpy()
            y_norm = y.cpu().numpy()

            preds_cms = (preds_norm * target_std) + target_mean
            obs_cms = (y_norm * target_std) + target_mean

            results.append(pd.DataFrame({
                'q_obs_norm': y_norm,
                'q_sim_norm': preds_norm,
                'q_obs_cms': obs_cms,
                'q_sim_cms': preds_cms
            }))

    if results:
        final_df = pd.concat(results, ignore_index=True)

        final_df['q_sim_cms'] = final_df['q_sim_cms'].clip(lower=0.0)

        output_path = config.PROCESSED_DATA_DIR / "test_results.csv"
        final_df.to_csv(output_path, index=False)

        logger.info(f"Saved predictions to {output_path}")
        logger.info("Summary Statistics:")
        print(final_df.describe())
    else:
        logger.warning("No results generated!")


if __name__ == "__main__":
    evaluate_test_set()