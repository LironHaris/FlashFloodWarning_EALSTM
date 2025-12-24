import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import config
from preprocessing import DataScaler
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FlashFloodDataset(Dataset):
    def __init__(self, mode='train', seq_length=270):
        """
        PyTorch Dataset for Hydrological Time Series (Caravan Data).

        This class handles:
        1. Loading static basin attributes from multiple files (HydroATLAS, Caravan, etc.).
        2. Merging attributes and normalizing them (Z-score).
        3. Loading dynamic time-series data (Precipitation, etc.) for each basin.
        4. Creating sliding window samples for the LSTM model.

        Args:
            mode (str): Data split mode. Options: 'train', 'val', 'test'.
            seq_length (int): Lookback window size (number of past days fed to the model).
        """
        self.mode = mode
        self.seq_length = seq_length

        self.scaler = DataScaler()
        self.scaler.load_scaler()


        dfs = []
        for csv_name in config.ATTRIBUTES_FILES:
            file_path = config.ATTRIBUTES_DIR / csv_name
            if file_path.exists():
                temp_df = pd.read_csv(file_path)

                if 'gauge_id' in temp_df.columns:
                    temp_df['gauge_id'] = temp_df['gauge_id'].astype(str)
                    temp_df.set_index('gauge_id', inplace=True)

                dfs.append(temp_df)
            else:
                logger.warning(f"Attribute file not found: {file_path}")

        if dfs:
            self.attributes_df = pd.concat(dfs, axis=1)
            self.attributes_df = self.attributes_df.loc[:, ~self.attributes_df.columns.duplicated()]
            self.attributes_df.reset_index(inplace=True)
        else:
            raise FileNotFoundError("No attribute files found! Check config.ATTRIBUTES_FILES.")


        numeric_cols = self.attributes_df.select_dtypes(include=[np.number]).columns.tolist()
        self.static_feature_cols = [c for c in numeric_cols if c != 'gauge_id']

        self.attributes_df[self.static_feature_cols] = self.attributes_df[self.static_feature_cols].fillna(0)

        self.attributes_df[self.static_feature_cols] = (
                                                               self.attributes_df[self.static_feature_cols] -
                                                               self.attributes_df[self.static_feature_cols].mean()
                                                       ) / (self.attributes_df[self.static_feature_cols].std() + 1e-6)

        self.attributes_df.set_index('gauge_id', inplace=True)


        self.static_data_map = self.attributes_df[self.static_feature_cols].T.to_dict('list')

        self.samples = []
        self._load_data()

    def _load_data(self):
        """
        Iterates over all CSV files, filters by date based on the mode,
        normalizes dynamic features, and stores valid basins in memory.
        """
        if self.mode == 'train':
            start_date = pd.to_datetime("1980-01-01")
            end_date = pd.to_datetime("2000-09-30")
        elif self.mode == 'val':
            start_date = pd.to_datetime("2000-10-01")
            end_date = pd.to_datetime("2005-09-30")
        else:
            start_date = pd.to_datetime("2005-10-01")
            end_date = pd.to_datetime("2015-09-30")

        ts_files = list(config.TIMESERIES_DIR.glob('*.csv'))

        logger.info(f"Loading {self.mode.upper()} data ({start_date.date()} - {end_date.date()})...")

        for f in tqdm(ts_files, desc=f"Processing {self.mode}"):
            try:
                basin_id = str(f.stem)

                if basin_id not in self.static_data_map:
                    continue

                df = pd.read_csv(f)
                df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])

                mask = (df[config.DATE_COL] >= start_date) & (df[config.DATE_COL] <= end_date)
                df = df[mask].copy()

                if len(df) < self.seq_length + 1:
                    continue

                df = self.scaler.normalize(df)

                features = config.DYNAMIC_FEATURES + ['sin_day', 'cos_day']
                target = config.TARGET_COL

                data_matrix = torch.tensor(df[features].values, dtype=torch.float32)
                target_array = torch.tensor(df[target].values, dtype=torch.float32)

                self.samples.append({
                    'basin_id': basin_id,
                    'data_matrix': data_matrix,
                    'target_array': target_array
                })

            except Exception:
                pass

        self.index_map = []
        for i, sample in enumerate(self.samples):
            num_rows = len(sample['data_matrix'])
            num_windows = num_rows - self.seq_length

            for start_idx in range(num_windows):
                self.index_map.append((i, start_idx))

        logger.info(f"Created {len(self.index_map)} samples for {self.mode}.")

    def __len__(self):
        """Returns the total number of samples (windows) available."""
        return len(self.index_map)

    def __getitem__(self, idx):
        """
        Retrieves a single sample given an index.

        Returns:
            x_dyn (Tensor): Dynamic features sequence [seq_length, num_features]
            x_stat (Tensor): Static features vector [num_static_features]
            y (Tensor): Target value (Discharge) at the end of the sequence
        """
        sample_idx, start_row = self.index_map[idx]
        sample_data = self.samples[sample_idx]

        basin_id = sample_data['basin_id']

        x_dyn = sample_data['data_matrix'][start_row: start_row + self.seq_length]

        y = sample_data['target_array'][start_row + self.seq_length - 1]

        x_stat = torch.tensor(self.static_data_map[basin_id], dtype=torch.float32)

        return x_dyn, x_stat, y


if __name__ == "__main__":
    for m in ['train', 'val']:
        print(f"\n--- Checking {m.upper()} Dataset ---")
        ds = FlashFloodDataset(mode=m, seq_length=270)

        if len(ds) > 0:
            print(f"Total Samples: {len(ds)}")
            x_d, x_s, y = ds[0]
            print(f"Sample 0 Shapes:")
            print(f"  Dynamic (X): {x_d.shape}  [Seq_Length, Dyn_Features]")
            print(f"  Static  (A): {x_s.shape}  [Static_Features]")
            print(f"  Target  (Y): {y.shape}    [Scalar]")
        else:
            print("Dataset is empty! Check data paths and date ranges.")