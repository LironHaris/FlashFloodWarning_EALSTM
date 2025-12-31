import pandas as pd
import numpy as np
from scipy.stats import gumbel_r
from tqdm import tqdm
import config
import logging
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_basin_area(gauge_id: str, df_attributes: pd.DataFrame) -> Optional[float]:
    """
    Retrieves the area of a specific basin from the attributes DataFrame.
    """
    row = df_attributes[df_attributes['gauge_id'].astype(str) == str(gauge_id)]
    if len(row) > 0:
        return float(row.iloc[0]['area'])
    return None


def calculate_return_periods():
    """
    Main function to calculate discharge thresholds for various return periods
    using the Gumbel distribution on Annual Maxima Series (AMS).
    """
    logger.info("Starting Return Period Calculation (Strict Mode for 20y Data)...")

    attr_path = config.ATTRIBUTES_DIR / 'attributes_other_il.csv'
    if not attr_path.exists():
        logger.error(f"Attributes file not found at {attr_path}")
        return

    df_attr = pd.read_csv(attr_path)

    return_periods = [2, 5, 10, 25, 50, 100]
    results = []

    ts_files = list(config.TIMESERIES_DIR.glob('*.csv'))
    logger.info(f"Found {len(ts_files)} basins to process.")

    processed_count = 0
    skipped_count = 0

    for f in tqdm(ts_files, desc="Processing Basins"):
        basin_id = f.stem

        area = get_basin_area(basin_id, df_attr)
        if area is None:
            skipped_count += 1
            continue

        try:
            df = pd.read_csv(f)
            df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
        except Exception as e:
            logger.error(f"Failed to read {f.name}: {e}")
            skipped_count += 1
            continue

        df = df.dropna(subset=[config.TARGET_COL])

        if len(df) < 365 * 5:
            skipped_count += 1
            continue

        df['discharge_cms'] = df[config.TARGET_COL].values
        df['water_year'] = df[config.DATE_COL].dt.year + (df[config.DATE_COL].dt.month >= 10).astype(int)
        annual_max = df.groupby('water_year')['discharge_cms'].max()

        if len(annual_max) < 10:
            skipped_count += 1
            continue

        try:
            loc, scale = gumbel_r.fit(annual_max.values)
        except Exception:
            skipped_count += 1
            continue

        basin_result = {'gauge_id': basin_id, 'area_km2': area}

        for rp in return_periods:
            p = 1 - (1.0 / rp)
            q_val = gumbel_r.ppf(p, loc=loc, scale=scale)
            basin_result[f'Q_{rp}y'] = max(0.0, float(q_val))

        results.append(basin_result)
        processed_count += 1

    if results:
        res_df = pd.DataFrame(results)
        output_path = config.PROCESSED_DATA_DIR / 'return_periods.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        res_df.to_csv(output_path, index=False)
        logger.info(f"Successfully calculated return periods for {processed_count} basins.")
        logger.info(f"Results saved to: {output_path}")
    else:
        logger.warning("No basins were successfully processed.")

    logger.info(f"Skipped {skipped_count} basins due to insufficient history.")

if __name__ == "__main__":
    calculate_return_periods()