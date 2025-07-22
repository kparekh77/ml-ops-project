import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, raw_path, report_dir="reports", fig_dir="reports/figures"):
        self.raw_path  = raw_path
        self.report_dir = report_dir
        self.fig_dir   = fig_dir
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

    def run(self):
        df = pd.read_csv(self.raw_path)
        # summary table
        desc = df.describe(include='all')
        desc.to_csv(os.path.join(self.report_dir, "eda_summary.csv"))

        # missingness
        miss = df.isna().sum()
        miss.to_csv(os.path.join(self.report_dir, "eda_missing.csv"))

        # numeric distributions
        num_cols = df.select_dtypes(include=['int64','float64']).columns
        for col in num_cols:
            plt.figure()
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution: {col}")
            plt.savefig(os.path.join(self.fig_dir, f"{col}_dist.png"))
            plt.close()

        # categorical counts
        cat_cols = df.select_dtypes(include=['object','category']).columns
        for col in cat_cols:
            plt.figure(figsize=(8,4))
            sns.countplot(y=df[col], order=df[col].value_counts().index)
            plt.title(f"Counts: {col}")
            plt.savefig(os.path.join(self.fig_dir, f"{col}_counts.png"))
            plt.close()

        # correlation heatmap
        plt.figure(figsize=(10,8))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(self.fig_dir, "correlation_heatmap.png"))
        plt.close()

        print(f"✔ EDA complete. Reports in '{self.report_dir}' and figures in '{self.fig_dir}'.")
