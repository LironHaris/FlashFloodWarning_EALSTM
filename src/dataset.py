import torch
from torch.utils.data import Dataset
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
        PyTorch Dataset for Hydrological Time Series.
        Handles loading static attributes, dynamic forcing data, and flood thresholds.
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

        if dfs:
            self.attributes_df = pd.concat(dfs, axis=1)
            self.attributes_df = self.attributes_df.loc[:, ~self.attributes_df.columns.duplicated()]
            self.attributes_df.reset_index(inplace=True)
        else:
            raise FileNotFoundError("No attribute files found!")

        numeric_cols = self.attributes_df.select_dtypes(include=[np.number]).columns.tolist()
        self.static_feature_cols = [c for c in numeric_cols if c != 'gauge_id']

        self.attributes_df[self.static_feature_cols] = self.attributes_df[self.static_feature_cols].fillna(0)
        self.attributes_df[self.static_feature_cols] = (
                                                               self.attributes_df[self.static_feature_cols] -
                                                               self.attributes_df[self.static_feature_cols].mean()
                                                       ) / (self.attributes_df[self.static_feature_cols].std() + 1e-6)

        self.attributes_df.set_index('gauge_id', inplace=True)
        self.static_data_map = self.attributes_df[self.static_feature_cols].T.to_dict('list')

        rp_path = config.PROCESSED_DATA_DIR / 'return_periods.csv'
        self.basin_thresholds_cms = {}
        self.basin_areas = {}

        if rp_path.exists():
            rp_df = pd.read_csv(rp_path)
            rp_df['gauge_id'] = rp_df['gauge_id'].astype(str)
            self.basin_thresholds_cms = rp_df.set_index('gauge_id')['Q_5y'].to_dict()
            self.basin_areas = rp_df.set_index('gauge_id')['area_km2'].to_dict()
        else:
            logger.warning(f"Return periods file not found at {rp_path}")

        self.samples = []
        self._load_data()

    def _load_data(self):
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
        logger.info(f"Loading {self.mode.upper()} data...")

        conversion_factor = (1000 * 1000) / (1000 * 86400)  # ≈ 0.011574 (mm/day * area -> CMS)

        for f in tqdm(ts_files, desc=f"Processing {self.mode}"):
            try:
                basin_id = str(f.stem)
                if basin_id not in self.static_data_map:
                    continue

                df = pd.read_csv(f)
                df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
                mask = (df[config.DATE_COL] >= start_date) & (df[config.DATE_COL] <= end_date)
                df = df[mask].copy()

                df = df.interpolate(method='linear', limit=5).dropna()

                if len(df) < self.seq_length + 1:
                    continue

                orig_mean = df[config.TARGET_COL].mean()
                orig_std = df[config.TARGET_COL].std()

                df = self.scaler.normalize(df)
                basin_var = df[config.TARGET_COL].var()
                norm_threshold = 9999.0

                if basin_id in self.basin_thresholds_cms and basin_id in self.basin_areas:
                    q5_cms = self.basin_thresholds_cms[basin_id]
                    area = self.basin_areas[basin_id]

                    if area > 0:
                        q5_mm_day = q5_cms / (area * conversion_factor)
                        norm_threshold = (q5_mm_day - orig_mean) / (orig_std + 1e-6)

                features = config.DYNAMIC_FEATURES + ['sin_day', 'cos_day']
                target = config.TARGET_COL

                data_matrix = torch.tensor(df[features].values, dtype=torch.float32)
                target_array = torch.tensor(df[target].values, dtype=torch.float32)

                self.samples.append({
                    'basin_id': basin_id,
                    'data_matrix': data_matrix,
                    'target_array': target_array,
                    'basin_var': basin_var,
                    'threshold': norm_threshold
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
        return len(self.index_map)

    def __getitem__(self, idx):
        sample_idx, start_row = self.index_map[idx]
        sample = self.samples[sample_idx]

        basin_id = sample['basin_id']

        x_dyn = sample['data_matrix'][start_row: start_row + self.seq_length]
        y = sample['target_array'][start_row + self.seq_length - 1]
        x_stat = torch.tensor(self.static_data_map[basin_id], dtype=torch.float32)

        basin_var = torch.tensor(sample['basin_var'], dtype=torch.float32)
        threshold = torch.tensor(sample['threshold'], dtype=torch.float32)

        return x_dyn, x_stat, y, basin_var, threshold