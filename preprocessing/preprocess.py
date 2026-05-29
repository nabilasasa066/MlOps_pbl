import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(df):
    df = df.copy()

    drop_cols = ['ID_Proyek', 'Nama_Proyek']
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    categorical_cols = ['Lokasi', 'Jenis_Layanan']

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()

    numeric_cols = [
        'Anggaran_Proyek_IDR',
        'Nilai_Kontrak_Konsultan_IDR',
        'Durasi_Bulan',
        'Progress_Fisik'
    ]

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, encoders, scaler


if __name__ == "__main__":

    df = pd.read_excel("dataset/data_konstruksi.xlsx")

    df_processed, encoders, scaler = preprocess(df)

    os.makedirs(
        "Membangun_model/data_konstruksi_preprocessed",
        exist_ok=True
    )

    df_processed.to_csv(
        "Membangun_model/data_konstruksi_preprocessed/preprocessed.csv",
        index=False
    )

    print("Preprocessing selesai")