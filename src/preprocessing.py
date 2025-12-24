import pandas as pd
import numpy as np
import json
import config
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataScaler:
    """
    Handles data normalization and feature engineering specifically for hydrological time series.
    Ensures no data leakage by calculating statistics solely on the training set.
    """

    def __init__(self, output_file='scaler_params.json'):
        self.output_file = config.PROCESSED_DATA_DIR / output_file
        self.means = {}
        self.stds = {}
        self.feature_names = []

    def fit(self, train_start_date, train_end_date):
        """
        Computes Mean and Standard Deviation for all dynamic features and the target variable.
        Crucially, it only considers data within the specified [train_start_date, train_end_date] range
        to prevent data leakage from the test set.

        Args:
            train_start_date (str): Start date for training data (YYYY-MM-DD).
            train_end_date (str): End date for training data (YYYY-MM-DD).
        """
        logger.info(f"Computing statistics on Training Set: {train_start_date} to {train_end_date}")

        all_data = []
        ts_files = list(config.TIMESERIES_DIR.glob('*.csv'))

        features_to_calc = config.DYNAMIC_FEATURES + [config.TARGET_COL]
        self.feature_names = features_to_calc

        for f in tqdm(ts_files, desc="Scanning Training Data"):
            try:
                df = pd.read_csv(f)

                if config.DATE_COL not in df.columns:
                    continue
                df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])

                mask = (df[config.DATE_COL] >= pd.to_datetime(train_start_date)) & \
                       (df[config.DATE_COL] <= pd.to_datetime(train_end_date))

                train_df = df.loc[mask, features_to_calc]
                train_df = train_df.dropna()

                if not train_df.empty:
                    all_data.append(train_df)

            except Exception as e:
                logger.warning(f"Error reading {f.name}: {e}")

        if not all_data:
            raise ValueError("No training data found in the specified date range!")

        full_train_df = pd.concat(all_data, axis=0)

        logger.info("Calculating Mean and Std...")
        for col in features_to_calc:
            self.means[col] = float(full_train_df[col].mean())
            self.stds[col] = float(full_train_df[col].std())

            if self.stds[col] == 0:
                self.stds[col] = 1.0

        self.save_scaler()
        logger.info(f"Scaler saved to {self.output_file}")

    def save_scaler(self):
        """Saves the calculated statistics to a JSON file."""
        data = {
            'means': self.means,
            'stds': self.stds,
            'features': self.feature_names
        }
        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=4)

    def load_scaler(self):
        """Loads statistics from the JSON file."""
        if not self.output_file.exists():
            raise FileNotFoundError(f"Scaler file not found at {self.output_file}. Run fit() first.")

        with open(self.output_file, 'r') as f:
            data = json.load(f)
            self.means = data['means']
            self.stds = data['stds']
            self.feature_names = data['features']
        logger.info("Scaler loaded successfully.")

    def normalize(self, df):
        """
        Applies Z-Score normalization using the stored statistics.
        Also adds cyclical time encoding (sin/cos) for seasonality.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Normalized dataframe with additional time features.
        """
        df_norm = df.copy()

        for col in self.feature_names:
            if col in df_norm.columns:
                mean = self.means.get(col, 0)
                std = self.stds.get(col, 1)
                df_norm[col] = (df_norm[col] - mean) / std

        if config.DATE_COL in df_norm.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_norm[config.DATE_COL]):
                df_norm[config.DATE_COL] = pd.to_datetime(df_norm[config.DATE_COL])

            day_of_year = df_norm[config.DATE_COL].dt.dayofyear

            df_norm['sin_day'] = np.sin(2 * np.pi * day_of_year / 366)
            df_norm['cos_day'] = np.cos(2 * np.pi * day_of_year / 366)

        return df_norm

if __name__ == "__main__":
    TRAIN_START = "1980-01-01"
    TRAIN_END = "2005-09-30"

    scaler = DataScaler()
    scaler.fit(TRAIN_START, TRAIN_END)

    print("\n--- Calibration Results (First 3 features) ---")
    keys = list(scaler.means.keys())[:3]
    for k in keys:
        print(f"{k}: Mean={scaler.means[k]:.4f}, Std={scaler.stds[k]:.4f}")