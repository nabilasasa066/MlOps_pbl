import streamlit as st
import pandas as pd
import joblib
import os
import time
import gc as python_gc  # Modul GC bawaan Python untuk membaca statistik garbage collector
from prometheus_client import (
    Counter, Histogram, Gauge,
    push_to_gateway, CollectorRegistry,
    start_http_server
)
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# Prometheus Metrics (cached to avoid duplicate registration on hot-reload)
# Menggunakan CollectorRegistry CUSTOM (bukan global REGISTRY) agar aman
# saat Streamlit Cloud me-restart atau hot-reload: registry baru = tidak ada konflik
# ─────────────────────────────────────────────
@st.cache_resource
def create_metrics():
    # Registry baru setiap cache miss — tidak pernah bentrok dengan sesi sebelumnya
    registry = CollectorRegistry()

    prediction_total      = Counter("prediction_total",              "Total number of predictions made",             registry=registry)
    high_value_total      = Counter("prediction_high_value_total",   "Total High Value predictions",                 registry=registry)
    low_value_total       = Counter("prediction_low_value_total",    "Total Low Value predictions",                  registry=registry)
    prediction_latency    = Histogram("prediction_latency_seconds",  "Prediction latency in seconds",
                                      buckets=[.001, .005, .01, .025, .05, .1, .25, .5, 1.0],
                                      registry=registry)
    last_latency          = Gauge("prediction_last_latency_seconds", "Last recorded prediction latency",            registry=registry)
    model_accuracy        = Gauge("model_accuracy",                  "Loaded model training accuracy",              registry=registry)
    app_requests          = Counter("app_requests_total",            "Total Streamlit app page loads",              registry=registry)
    # --- Inference specific metrics ---
    inference_total       = Counter("inference_requests_total",      "Total inference API calls",                   registry=registry)
    inference_errors      = Counter("inference_errors_total",        "Total failed inference calls",                registry=registry)
    inference_latency     = Histogram("inference_latency_seconds",   "End-to-end inference latency",
                                      buckets=[.001, .005, .01, .025, .05, .1, .25, .5, 1.0],
                                      registry=registry)
    # --- Model quality metrics ---
    model_confidence_high = Gauge("model_confidence_high_value",     "Last prediction confidence for High Value class", registry=registry)
    model_confidence_low  = Gauge("model_confidence_low_value",      "Last prediction confidence for Low Value class",  registry=registry)
    # --- GC Metrics (dibaca dari modul gc Python, di-push via Pushgateway) ---
    # Menggunakan Gauge berlabel 'generation' (0, 1, 2) — meniru format asli prometheus_client
    gc_objects_collected   = Gauge("python_gc_objects_collected_total",
                                   "Total objects collected by Python GC per generation",
                                   ["generation"], registry=registry)
    gc_objects_uncollect   = Gauge("python_gc_objects_uncollectable_total",
                                   "Total uncollectable objects found by Python GC per generation",
                                   ["generation"], registry=registry)
    gc_collections         = Gauge("python_gc_collections_total",
                                   "Total number of GC collection runs per generation",
                                   ["generation"], registry=registry)
    return (
        registry,
        prediction_total, high_value_total, low_value_total,
        prediction_latency, last_latency, model_accuracy, app_requests,
        inference_total, inference_errors, inference_latency,
        model_confidence_high, model_confidence_low,
        gc_objects_collected, gc_objects_uncollect, gc_collections
    )

(
    METRICS_REGISTRY,
    PREDICTION_TOTAL, HIGH_VALUE_TOTAL, LOW_VALUE_TOTAL,
    PREDICTION_LATENCY, LAST_LATENCY, MODEL_ACCURACY, APP_REQUESTS,
    INFERENCE_TOTAL, INFERENCE_ERRORS, INFERENCE_LATENCY,
    MODEL_CONFIDENCE_HIGH, MODEL_CONFIDENCE_LOW,
    GC_OBJECTS_COLLECTED, GC_OBJECTS_UNCOLLECT, GC_COLLECTIONS
) = create_metrics()

# ─────────────────────────────────────────────
# Start Prometheus HTTP Server (Pull Mode — port 8000)
# Dilindungi session_state agar tidak restart setiap Streamlit hot-reload
# ─────────────────────────────────────────────
if not st.session_state.get("prometheus_server_started", False):
    try:
        start_http_server(8000, registry=METRICS_REGISTRY)
        st.session_state["prometheus_server_started"] = True
    except OSError:
        # Port sudah dipakai — server sudah jalan dari sesi sebelumnya
        st.session_state["prometheus_server_started"] = True

# Konfigurasi URL Pushgateway (Ngrok) untuk Mendorong (Push) Metrik ke Lokal
# PENTING: Ganti URL ini dengan URL Ngrok Anda setiap kali Ngrok direstart!
NGROK_PUSHGATEWAY_URL = "https://craziness-donut-trickster.ngrok-free.dev" 

def update_gc_metrics():
    """Baca statistik GC Python secara real-time lalu update Gauge metrics.
    gc.get_stats() mengembalikan list 3 dict (untuk generasi 0, 1, 2):
      [{'collections': N, 'collected': N, 'uncollectable': N}, ...]
    """
    stats = python_gc.get_stats()  # Snapshot GC saat ini
    for gen, stat in enumerate(stats):
        GC_OBJECTS_COLLECTED.labels(generation=str(gen)).set(stat["collected"])
        GC_OBJECTS_UNCOLLECT.labels(generation=str(gen)).set(stat["uncollectable"])
        GC_COLLECTIONS.labels(generation=str(gen)).set(stat["collections"])

def push_metrics_to_local():
    """Update GC metrics dari Python runtime, lalu push semua metrics ke Pushgateway."""
    update_gc_metrics()  # Snapshot GC terbaru sebelum push
    try:
        push_to_gateway(NGROK_PUSHGATEWAY_URL, job="sales-model-streamlit-cloud", registry=METRICS_REGISTRY)
    except Exception:
        pass  # Abaikan error jika Ngrok mati agar app tidak crash

# ─────────────────────────────────────────────
# Load Model & Accuracy
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "Workflow-CI", "MLProject", "outputs", "rf_model.pkl"
)

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_resource
def get_fitted_scaler():
    """Builds a StandardScaler fitted to historical data so user inputs are scaled like during training"""
    DATA_PATH_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "data_penjualan.csv")
    if os.path.exists(DATA_PATH_RAW):
        df = pd.read_csv(DATA_PATH_RAW, sep=";")
        df.dropna(inplace=True)
        if "Tanggal" in df.columns:
            df["Tanggal"] = pd.to_datetime(df["Tanggal"], format="%d/%m/%Y")
            df["Year"] = df["Tanggal"].dt.year
            df["Month"] = df["Tanggal"].dt.month
            df["Day"] = df["Tanggal"].dt.day
        
        num_cols = ["Jumlah Order", "Harga", "Year", "Month", "Day"]
        scaler = StandardScaler()
        # Clean dataframe to make sure columns exist
        df_num = df[[c for c in num_cols if c in df.columns]]
        if not df_num.empty:
            scaler.fit(df_num)
            return scaler
    return None

model = load_model()
scaler = get_fitted_scaler()

# Set statis untuk simulasi akurasi model dari data testing sebelumnya
MODEL_ACCURACY.set(0.986)

# ─────────────────────────────────────────────
# App Layout
# ─────────────────────────────────────────────
APP_REQUESTS.inc()
push_metrics_to_local()
st.set_page_config(page_title="Sales Value Dashboard", layout="wide")
st.title("📈 Sales Value Classification Dashboard")
st.caption("Prediksi apakah suatu transaksi bernilai **High Value** atau **Low Value**")

tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 EDA Dataset", "📈 Model Performance"])

# ─────────────────────────────────────────────
# TAB 1 — Prediksi
# ─────────────────────────────────────────────
with tab1:
    st.subheader("Input Data Penjualan")

    col1, col2 = st.columns(2)
    with col1:
        tanggal      = st.date_input("Tanggal Transaksi", value=pd.to_datetime("2022-08-05"))
        jenis_produk = st.selectbox("Jenis Produk", ["Foodpak260", "FoodpakMatte245", "CraftLaminasi290", "Other"])
    with col2:
        jumlah_order = st.number_input("Jumlah Order", min_value=1, max_value=1000000, value=1000)
        harga        = st.number_input("Harga", min_value=1, max_value=1000000, value=1800)

    if st.button("🔍 Prediksi Value Level"):
        if model is None:
            st.error("Model belum tersedia. Jalankan pipeline training terlebih dahulu.")
        else:
            # Simple encoding for demo (same as LabelEncoder)
            # You should ideally save and load local LabelEnoder
            jenis_map = {"CraftLaminasi290": 0, "Foodpak260": 1, "FoodpakMatte245": 2, "Other": 3}
            
            input_data = pd.DataFrame([{
                "Jenis Produk": jenis_map.get(jenis_produk, 3),
                "Jumlah Order": jumlah_order,
                "Harga":        harga,
                "Year":         tanggal.year,
                "Month":        tanggal.month,
                "Day":          tanggal.day
            }])

            # IMPORTANT: Scale the input using the same scaler fitted from dataset
            if scaler is not None:
                num_cols = ["Jumlah Order", "Harga", "Year", "Month", "Day"]
                # Cek jika kolom ada
                cols_to_scale = [c for c in num_cols if c in input_data.columns]
                input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

            # Susun urutan kolom persis seperti training:
            input_data = input_data[["Jenis Produk", "Jumlah Order", "Harga", "Year", "Month", "Day"]]

            start_time = time.time()
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            latency = time.time() - start_time

            # Update Prometheus metrics
            PREDICTION_TOTAL.inc()
            PREDICTION_LATENCY.observe(latency)
            LAST_LATENCY.set(latency)  # Simpan kecepatan terakhir
            INFERENCE_TOTAL.inc()
            INFERENCE_LATENCY.observe(latency)
            MODEL_CONFIDENCE_HIGH.set(float(probability[1]))
            MODEL_CONFIDENCE_LOW.set(float(probability[0]))
            if prediction == 1:
                HIGH_VALUE_TOTAL.inc()
            else:
                LOW_VALUE_TOTAL.inc()

            # Dorong metrik ke Pushgateway Lokal via Ngrok
            push_metrics_to_local()

            # Display result
            if prediction == 1:
                st.success(f"🟢 **HIGH VALUE** — Prediksi total penjualan tinggi")
            else:
                st.warning(f"🔴 **LOW VALUE** — Prediksi total penjualan rendah")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Low Value Probability",  f"{probability[0]:.2%}")
            col_b.metric("High Value Probability", f"{probability[1]:.2%}")
            col_c.metric("Latency",               f"{latency*1000:.1f} ms")

# ─────────────────────────────────────────────
# TAB 2 — EDA Dataset
# ─────────────────────────────────────────────
with tab2:
    DATA_PATH = os.path.join(os.path.dirname(__file__),
                             "..", "data", "data_penjualan.csv")
    if os.path.exists(DATA_PATH):
        import matplotlib.pyplot as plt
        import seaborn as sns

        df_raw = pd.read_csv(DATA_PATH, sep=";")
        st.subheader("Preview Dataset Penjualan")
        st.dataframe(df_raw.head(20), width='stretch')

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows",    df_raw.shape[0])
            st.metric("Total Columns", df_raw.shape[1])

        st.subheader("Distribusi Total (Sales Value)")
        fig, ax = plt.subplots(figsize=(8, 4))
        df_raw["Total"].dropna().hist(bins=40, ax=ax, color="#3B82F6", edgecolor="white")
        ax.axvline(df_raw["Total"].median(), color="navy", linestyle="--",
                   label=f"Median = {df_raw['Total'].median():.1f}")
        ax.set_xlabel("Total Penjualan")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Rata-rata Total per Jenis Produk")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        df_raw.groupby("Jenis Produk")["Total"].mean().sort_values().plot(
            kind="barh", ax=ax2, color="#10B981")
        ax2.set_xlabel("Avg Total Penjualan")
        st.pyplot(fig2)

    else:
        st.warning("Dataset data_penjualan.csv tidak ditemukan di folder data/. Copy file terlebih dahulu.")

# ─────────────────────────────────────────────
# TAB 3 — Model Performance
# ─────────────────────────────────────────────
with tab3:
    st.subheader("Model Info")
    if model is not None:
        st.success("✅ Model berhasil dimuat")
        st.json({
            "type":         type(model).__name__,
            "n_estimators": model.n_estimators,
            "max_depth":    str(model.max_depth),
            "n_features":   model.n_features_in_,
        })

        st.subheader("Feature Importances")
        feature_names = [
            "Jenis Produk", "Jumlah Order", "Harga",
            "Year", "Month", "Day"
        ]
        importances = pd.Series(model.feature_importances_, index=feature_names)
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        importances.sort_values().plot(kind="barh", ax=ax4, color="#10B981")
        ax4.set_xlabel("Importance")
        st.pyplot(fig4)
    else:
        st.error("Model belum tersedia. Jalankan training pipeline terlebih dahulu.")

    st.subheader("Prometheus Metrics Endpoint")
    st.info(f"Metrik di-*push* secara aktif ke: `{NGROK_PUSHGATEWAY_URL}`")
    st.info("""
    ✅ GC metrics (`python_gc_objects_collected_total`, `python_gc_collections_total`)
    sekarang di-push via Pushgateway — tersedia di Prometheus meskipun app berjalan di Streamlit Cloud.
    Nilai GC dibaca langsung dari runtime Python (`gc.get_stats()`) setiap kali tombol Prediksi ditekan.
    """)

    st.subheader("📋 Daftar Query Prometheus (PromQL)")
    st.code("""
# ── GC (Garbage Collector) Metrics ────────────────────────
# Total objek Python yang berhasil di-GC (per generasi)
python_gc_objects_collected_total

# Total objek yang TIDAK BISA di-GC (memory leak indicator)
python_gc_objects_uncollectable_total

# Jumlah GC collection yang terjadi per generasi
python_gc_collections_total

# ── Inference Metrics ─────────────────────────────────────
# Total prediksi yang sudah dibuat
prediction_total

# Total inference API calls
inference_requests_total

# Latency prediksi (histogram — rata-rata)
rate(inference_latency_seconds_sum[5m]) / rate(inference_latency_seconds_count[5m])

# Latency prediksi terakhir (gauge)
prediction_last_latency_seconds

# Persentil 95 latency (5 menit terakhir)
histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))

# ── Target / Prediction Distribution ──────────────────────
# Total prediksi High Value
prediction_high_value_total

# Total prediksi Low Value
prediction_low_value_total

# Confidence terakhir untuk High Value
model_confidence_high_value

# Confidence terakhir untuk Low Value
model_confidence_low_value

# ── Model & App Metrics ───────────────────────────────────
# Akurasi model yang sedang aktif
model_accuracy

# Total request ke dashboard Streamlit
app_requests_total

# ── Process Metrics (otomatis dari Python) ─────────────────
# Memory RSS yang digunakan proses (bytes)
process_resident_memory_bytes

# CPU seconds yang digunakan
process_cpu_seconds_total
    """, language="promql")

    st.subheader("🎯 Cara Membuka Prometheus GUI")
    st.markdown("""
    1. Jalankan `prometheus.exe --config.file=prometheus.yml` di folder Prometheus lokal
    2. Buka browser → **`http://localhost:9090`**
    3. Di kolom **Expression**, ketik salah satu query di atas lalu klik **Execute**
    4. Pilih tab **Graph** untuk melihat grafik waktu, atau **Table** untuk nilai sekarang
    5. Buka **`http://localhost:9090/targets`** untuk melihat status scrape endpoint
    """)
