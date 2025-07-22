import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeatureEngineer:
    def __init__(self, cfg):
        self.raw_path = cfg.data.raw_data_path
        self.processed_path = cfg.data.processed_data_path

    def run(self):
        print(f"Loading raw data from {self.raw_path} …")
        df = pd.read_csv(self.raw_path)
        # basic cleaning
        df = df.replace('?', pd.NA).dropna()
        # binary target
        df['class'] = (df['class'] == '>50K').astype(int)

        # train/test split
        X = df.drop('class', axis=1)
        y = df['class']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # numeric vs categorical
        num_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
        cat_cols = X_train.select_dtypes(include=['object','category']).columns.tolist()

        # preprocess
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train[num_cols])
        X_test_num  = scaler.transform(X_test[num_cols])

        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        X_train_cat = encoder.fit_transform(X_train[cat_cols])
        X_test_cat  = encoder.transform(X_test[cat_cols])

        X_train_proc = np.hstack([X_train_num, X_train_cat])
        X_test_proc  = np.hstack([X_test_num, X_test_cat])

        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        joblib.dump({
            "X_train": X_train_proc,
            "X_test": X_test_proc,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler,
            "encoder": encoder,
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols
        }, self.processed_path)

        print(f"✔ Processed data + encoders saved to {self.processed_path}")
