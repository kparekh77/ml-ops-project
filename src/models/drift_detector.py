import joblib
import pandas as pd
from scipy.stats import ks_2samp

class DriftDetector:
    def __init__(self, cfg):
        self.cfg     = cfg
        self.raw     = cfg.data.raw_data_path
        self.proc    = cfg.data.processed_data_path
        self.thresh  = cfg.drift.threshold
        self.flag_fp = "drift_flag.txt"

    def run(self):
        print("Loading baseline test split …")
        art = joblib.load(self.proc)
        X_test = art["X_test"]

        # for demo, we treat the entire fresh raw data as 'production'
        df = pd.read_csv(self.raw).replace('?', pd.NA).dropna()
        df['class'] = (df['class']=='>50K').astype(int)
        # rebuild same preprocessing
        num_cols = art["numeric_cols"]
        cat_cols = art["categorical_cols"]
        scaler   = art["scaler"]
        encoder  = art["encoder"]

        Xp_num = scaler.transform(df[num_cols])
        Xp_cat = encoder.transform(df[cat_cols])
        import numpy as np
        X_prod = np.hstack([Xp_num, Xp_cat])

        # KS test on each dimension
        drift = False
        print("Running KS tests …")
        for i in range(X_test.shape[1]):
            stat, _ = ks_2samp(X_test[:,i], X_prod[:,i])
            if stat > self.thresh:
                drift = True
                print(f" ↳ Drift on dimension {i}, KS={stat:.3f}")
                break

        # write flag
        with open(self.flag_fp, "w") as f:
            f.write(str(drift))
        print(f"✔ Drift detected = {drift} (wrote '{self.flag_fp}')")
        return drift
