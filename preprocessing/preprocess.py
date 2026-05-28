import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess(path):
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path, sep=";")
    print(f"[INFO] Dataset loaded: {df.shape}")

    # --- Handle Date column ---
    if "Tanggal" in df.columns:
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], format="%d/%m/%Y")
        df["Year"] = df["Tanggal"].dt.year
        df["Month"] = df["Tanggal"].dt.month
        df["Day"] = df["Tanggal"].dt.day
        df.drop(columns=["Tanggal"], inplace=True)
        print("[INFO] Tanggal column processed into Year, Month, Day.")

    # --- Handle missing values ---
    df.dropna(inplace=True)
    print(f"[INFO] After dropping rows with missing values: {df.shape}")

    # --- Create target label: High_Value ---
    if "Total" in df.columns:
        median_val = df["Total"].median()
        print(f"[INFO] Total median (threshold): {median_val}")
        df["High_Value"] = (df["Total"] > median_val).astype(int)
        df.drop(columns=["Total"], inplace=True)
        print(f"[INFO] High_Value distribution:\n{df['High_Value'].value_counts()}")

    # --- Encode categorical columns ---
    categorical_cols = ["Jenis Produk"]
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"[INFO] Encoded column: {col}")

    # --- Scale numerical columns ---
    num_cols = ["Jumlah Order", "Harga", "Year", "Month", "Day"]
    scaler = StandardScaler()
    existing_num_cols = [c for c in num_cols if c in df.columns]
    if existing_num_cols:
        df[existing_num_cols] = scaler.fit_transform(df[existing_num_cols])
        print(f"[INFO] Scaled numerical columns: {existing_num_cols}")

    print("[INFO] Preprocessing complete.")
    return df


if __name__ == "__main__":
    output_dir = "preprocessing/data_penjualan_preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "preprocessed.csv")
    df_ready = load_and_preprocess("data/data_penjualan.csv")
    df_ready.to_csv(output_path, index=False)
    print(f"[INFO] Preprocessed file saved to: {output_path}")
