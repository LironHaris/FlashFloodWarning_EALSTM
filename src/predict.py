import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import logging
import config
from dataset import FlashFloodDataset
from model import EALSTM
from preprocessing import DataScaler
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_hydrograph(df, save_dir):
    """
    Plots Observed vs Simulated discharge with Q2 Threshold and Performance Stats (Hits/False Alarms).
    """
    unique_basins = df['basin_id'].unique()
    if len(unique_basins) == 0:
        logger.warning("No basins found in results to plot.")
        return
    rp_path = config.PROCESSED_DATA_DIR / 'return_periods.csv'
    rp_df = None
    if rp_path.exists():
        rp_df = pd.read_csv(rp_path)
        rp_df['gauge_id'] = rp_df['gauge_id'].astype(str)
    else:
        logger.warning(f"Return periods file not found at {rp_path}. Thresholds will not be plotted.")

    chosen_basin = random.choice(unique_basins)
    basin_data = df[df['basin_id'] == chosen_basin].sort_values('date')

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(basin_data['date'], basin_data['q_obs_cms'], label='Observed (Target)', color='blue', alpha=0.6,
            linewidth=2)
    ax.plot(basin_data['date'], basin_data['q_sim_cms'], label='Simulated (EA-LSTM)', color='orange', alpha=0.9,
            linestyle='--')

    stats_text = "Threshold Stats:\nN/A"

    if rp_df is not None:
        basin_rp_row = rp_df[rp_df['gauge_id'] == str(chosen_basin)]

        if not basin_rp_row.empty and 'Q_2y' in basin_rp_row.columns:
            q2_val = basin_rp_row.iloc[0]['Q_2y']

            obs = basin_data['q_obs_cms'].values
            sim = basin_data['q_sim_cms'].values

            obs_flood = obs > q2_val
            sim_flood = sim > q2_val

            hits = np.sum(obs_flood & sim_flood)
            false_alarms = np.sum(~obs_flood & sim_flood)
            misses = np.sum(obs_flood & ~sim_flood)

            stats_text = (f"Threshold (Q2): {q2_val:.2f} cms\n"
                          f"---------------------------\n"
                          f"Hits (TP): {hits}\n"
                          f"False Alarms (FP): {false_alarms}\n"
                          f"Misses (FN): {misses}")
            ax.axhline(y=q2_val, color='red', linestyle='--', linewidth=1.5, label=f'Threshold Q2 ({q2_val:.2f})')
        else:
            logger.warning(f"Q2 threshold not found for basin {chosen_basin}")

    ax.set_title(f'Flash Flood Event Analysis: Basin {chosen_basin}', fontsize=14)
    ax.set_ylabel('Discharge ($m^3/s$)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.grid(True, alpha=0.3)

    ax.legend(loc='upper right')
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontname='Monospace')

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plot_path = save_dir / f"hydrograph_basin_{chosen_basin}_stats.png"
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Hydrograph with stats saved to: {plot_path}")
def evaluate_test_set():
    """
    Loads the best trained model and generates predictions for the TEST set.
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 256
    HIDDEN_DIM = 256

    logger.info(f"Running inference on device: {DEVICE}")

    test_ds = FlashFloodDataset(mode='test', seq_length=270)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    sample_x_dyn, sample_x_stat, _, _, _, _, _ = test_ds[0]
    input_dim_dyn = sample_x_dyn.shape[1]
    input_dim_stat = sample_x_stat.shape[0]

    model = EALSTM(input_dim_dyn, input_dim_stat, HIDDEN_DIM).to(DEVICE)
    model_path = config.MODELS_DIR / "best_model.pth"

    if not model_path.exists():
        logger.warning(f"Best model not found at {model_path}. Checking for checkpoint...")
        model_path = config.MODELS_DIR / "latest_checkpoint.pth"

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

    results = []

    logger.info("Generating predictions...")
    with torch.no_grad():
        for x_dyn, x_stat, y, _, _, basin_ids, dates in tqdm(test_loader, desc="Testing"):
            x_dyn, x_stat = x_dyn.to(DEVICE), x_stat.to(DEVICE)

            preds_norm = model(x_dyn, x_stat).squeeze()
            preds_norm = preds_norm.cpu().numpy()
            y_norm = y.cpu().numpy()

            preds_cms = (preds_norm * target_std) + target_mean
            obs_cms = (y_norm * target_std) + target_mean

            results.append(pd.DataFrame({
                'basin_id': basin_ids,
                'date': dates,
                'q_obs_norm': y_norm,
                'q_sim_norm': preds_norm,
                'q_obs_cms': obs_cms,
                'q_sim_cms': preds_cms
            }))

    if results:
        final_df = pd.concat(results, ignore_index=True)

        final_df['q_sim_cms'] = final_df['q_sim_cms'].clip(lower=0.0)
        final_df['date'] = pd.to_datetime(final_df['date'])

        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

        output_path = config.PROCESSED_DATA_DIR / "test_results.csv"
        final_df.to_csv(output_path, index=False)

        logger.info(f"Saved predictions to {output_path}")

        plot_hydrograph(final_df, config.PROCESSED_DATA_DIR)

        logger.info("Summary Statistics:")
        print(final_df[['q_obs_cms', 'q_sim_cms']].describe())
    else:
        logger.warning("No results generated!")


if __name__ == "__main__":
    evaluate_test_set()