import pandas as pd
from pathlib import Path

def export_to_csv():
    processed_dir = Path("data/processed")
    export_dir = Path("data/exports_csv")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Export Labeled Training Data (2,610 points)
    train_parquet = processed_dir / "split_train_v0.parquet"
    if train_parquet.exists():
        df_train = pd.read_parquet(train_parquet)
        df_train_friendly = df_train.rename(columns={
            "t": "dimensionless_time_t",
            "r": "dimensionless_radius_r",
            "I": "dimensionless_current_I",
            "T_amb": "dimensionless_ambient_temp",
            "voltage": "target_voltage",
            "concentration": "target_particle_concentration",
            "temperature": "target_cell_temperature"
        })
        df_train_friendly.head(1000).to_csv(export_dir / "training_data_sample_1000rows.csv", index=False)
        df_train_friendly.to_csv(export_dir / "training_data_full.csv", index=False)
        print(f"Exported training data: {len(df_train)} rows.")
    else:
        print("Training parquet file not found.")

    # 2. Export Unlabeled Collocation Points (36,000 points)
    colloc_parquet = processed_dir / "collocation_train_v0.parquet"
    if colloc_parquet.exists():
        df_col = pd.read_parquet(colloc_parquet)
        df_col_friendly = df_col.rename(columns={
            "t": "collocation_time_t",
            "r": "collocation_radius_r",
            "I": "collocation_current_I",
            "T_amb": "collocation_ambient_temp"
        })
        df_col_friendly.head(2000).to_csv(export_dir / "collocation_points_sample_2000rows.csv", index=False)
        df_col_friendly.to_csv(export_dir / "collocation_points_full.csv", index=False)
        print(f"Exported collocation points: {len(df_col)} rows.")
    else:
        print("Collocation parquet file not found.")

if __name__ == "__main__":
    export_to_csv()
