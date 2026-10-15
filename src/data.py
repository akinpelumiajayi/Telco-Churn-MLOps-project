from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/pmservice/wml-sample-models/master/spark/customer-satisfaction-prediction/data/WA_Fn%20UseC_%20Telco%20Customer%20Churn.csv"
OUT = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def download():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        urlretrieve(DATA_URL, OUT)
    df = pd.read_csv(OUT)
    print(f"Downloaded {len(df):,} rows -> {OUT}")
    print(df["Churn"].value_counts(normalize=True))
    return df


if __name__ == "__main__":
    download()
