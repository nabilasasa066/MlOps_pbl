import pandas as pd
from preprocessing.preprocess import preprocess

def test_preprocess_runs():
    df = pd.read_excel("dataset/data_konstruksi.xlsx")

    df_processed, encoders, scaler = preprocess(df)

    assert len(df_processed) > 0


def test_columns_after_preprocessing():
    df = pd.read_excel("dataset/data_konstruksi.xlsx")

    df_processed, encoders, scaler = preprocess(df)

    expected_columns = [
        "Lokasi",
        "Jenis_Layanan",
        "Anggaran_Proyek_IDR",
        "Nilai_Kontrak_Konsultan_IDR",
        "Status",
        "Durasi_Bulan",
        "Progress_Fisik"
    ]

    for col in expected_columns:
        assert col in df_processed.columns


def test_id_and_name_removed():
    df = pd.read_excel("dataset/data_konstruksi.xlsx")

    df_processed, encoders, scaler = preprocess(df)

    assert "ID_Proyek" not in df_processed.columns
    assert "Nama_Proyek" not in df_processed.columns


def test_no_missing_values():
    df = pd.read_excel("dataset/data_konstruksi.xlsx")

    df_processed, encoders, scaler = preprocess(df)

    assert df_processed.isnull().sum().sum() == 0