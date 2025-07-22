import os
import pandas as pd
from sklearn.datasets import fetch_openml

class DataIngestor:
    def __init__(self, cfg):
        self.raw_path = cfg.data.raw_data_path

    def run(self):
        os.makedirs(os.path.dirname(self.raw_path), exist_ok=True)
        print("Fetching Adult dataset from OpenML...")
        dataset = fetch_openml(name="adult", version=2, as_frame=True)
        df = pd.concat([dataset.data, dataset.target.rename("class")], axis=1)
        df.to_csv(self.raw_path, index=False)
        print(f"✔ Raw data saved to {self.raw_path}")
