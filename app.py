import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta

# KONFIGURASI HALAMAN

st.set_page_config(
    page_title="Dashboard Klasifikasi Gangguan Rantai Pasok",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_random_forest_gangguan.pkl")
FITUR_PATH = os.path.join(BASE_DIR, "fitur_model.pkl")
RIWAYAT_PATH = os.path.join(BASE_DIR, "riwayat_prediksi.csv")

# LOAD MODEL DAN FITUR

if not os.path.exists(MODEL_PATH):
    st.error("File model tidak ditemukan. Pastikan model_random_forest_gangguan.pkl ada di repo.")
    st.stop()

if not os.path.exists(FITUR_PATH):
    st.error("File fitur model tidak ditemukan. Pastikan fitur_model.pkl ada di repo.")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FITUR_PATH)


# FUNGSI BANTUAN

def waktu_wib():
    return (datetime.utcnow() + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S") + " WIB"


def baca_file_upload(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        return None


def ubah_ke_numeric(series):
    if series.dtype == "object":
        series = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def siapkan_data_prediksi(data_input):
    data_prediksi = data_input.copy()
    data_prediksi.columns = data_prediksi.columns.str.strip()

    if "Gangguan" in data_prediksi.columns:
        data_prediksi = data_prediksi.drop(columns=["Gangguan"])

    kolom_wajib = [
        "Tanggal", "TotalPengambilan", "TotalPesanan", "Sisa",
        "TambahPengambilan", "TambahPesanan", "suhu_rata_rata"
    ]
    missing_kolom = [col for col in kolom_wajib if col not in data_prediksi.columns]
    if len(missing_kolom) > 0:
        return None, missing_kolom, data_prediksi

    data_prediksi["Tanggal"] = pd.to_datetime(data_prediksi["Tanggal"], errors="coerce")
    data_prediksi["Tahun"] = data_prediksi["Tanggal"].dt.year
    data_prediksi["Bulan"] = data_prediksi["Tanggal"].dt.month
    data_prediksi["Hari"] = data_prediksi["Tanggal"].dt.day

    kolom_numerik_awal = [
        "TotalPengambilan", "TotalPesanan", "Sisa",
        "TambahPengambilan", "TambahPesanan", "suhu_rata_rata"
    ]
    for col in kolom_numerik_awal:
        data_prediksi[col] = ubah_ke_numeric(data_prediksi[col])

    data_prediksi["TambahPengambilan"] = data_prediksi["TambahPengambilan"].fillna(0)
    data_prediksi["TambahPesanan"] = data_prediksi["TambahPesanan"].fillna(0)

    total_pengambilan_awal = data_prediksi["TotalPengambilan"].copy()
    total_pesanan_awal = data_prediksi["TotalPesanan"].copy()

    total_pengambilan_akhir = total_pengambilan_awal + data_prediksi["TambahPengambilan"]
    total_pesanan_akhir = total_pesanan_awal + data_prediksi["TambahPesanan"]
    susut_hitung = total_pengambilan_akhir - (total_pesanan_akhir + data_prediksi["Sisa"])

    data_prediksi["TotalPengambilan"] = total_pengambilan_akhir
    data_prediksi["TotalPesanan"] = total_pesanan_akhir
    data_prediksi["Susut"] = susut_hitung

    missing_fitur_model = [col for col in feature_cols if col not in data_prediksi.columns]
    if len(missing_fitur_model) > 0:
        return None, missing_fitur_model, data_prediksi

    for col in feature_cols:
        data_prediksi[col] = ubah_ke_numeric(data_prediksi[col])

    return data_prediksi, [], data_prediksi


def prediksi_model(data_prediksi):
    X_prediksi = data_prediksi[feature_cols]
    prediksi = model.predict(X_prediksi)
    probabilitas = model.predict_proba(X_prediksi)
    kelas_model = list(model.classes_)

    prob_tidak_gangguan = probabilitas[:, kelas_model.index(0)] if 0 in kelas_model else [0] * len(prediksi)
    prob_gangguan = probabilitas[:, kelas_model.index(1)] if 1 in kelas_model else [0] * len(prediksi)

    return prediksi, prob_tidak_gangguan, prob_gangguan


def buat_tabel_hasil(data_prediksi, prediksi, prob_tidak_gangguan, prob_gangguan):
    hasil = pd.DataFrame()
    hasil["Tanggal"] = data_prediksi["Tanggal"].dt.strftime("%Y-%m-%d")
    hasil["TotalPengambilan"] = data_prediksi["TotalPengambilan"]
    hasil["TotalPesanan"] = data_prediksi["TotalPesanan"]
    hasil["Sisa"] = data_prediksi["Sisa"]
    hasil["Susut"] = data_prediksi["Susut"]
    hasil["TambahPengambilan"] = data_prediksi["TambahPengambilan"]
    hasil["TambahPesanan"] = data_prediksi["TambahPesanan"]
    hasil["Suhu Rata-rata"] = data_prediksi["suhu_rata_rata"]
    hasil["Prediksi_Gangguan"] = prediksi
    hasil["Keterangan_Prediksi"] = hasil["Prediksi_Gangguan"].map({0: "Tidak Gangguan", 1: "Gangguan"})
    hasil["Probabilitas_Tidak_Gangguan"] = prob_tidak_gangguan
    hasil["Probabilitas_Gangguan"] = prob_gangguan
    return hasil


# JUDUL DASHBOARD

st.title("Dashboard Klasifikasi Gangguan Rantai Pasok")
st.write(
    "Dashboard ini digunakan untuk mengklasifikasikan apakah data operasional harian "
    "mengalami gangguan rantai pasok atau tidak menggunakan model Random Forest."
)

st.markdown("---")

mode_prediksi = st.radio(
    "Pilih mode prediksi:",
    ["Input Manual Satu Data", "Upload Dataset Baru"],
    horizontal=True
)

# MODE 1: INPUT MANUAL SATU DATA

if mode_prediksi == "Input Manual Satu Data":

    st.subheader("Input Data Operasional Harian")

    with st.form("form_prediksi_manual"):
        col1, col2 = st.columns(2)
        with col1:
            tanggal = st.date_input("Tanggal")
        with col2:
            suhu_rata_rata = st.number_input("Suhu Rata-rata (°C)", min_value=0.0, step=0.1)

        col3, col4 = st.columns(2)
        with col3:
            total_pengambilan = st.number_input("Total Pengambilan (kg)", min_value=0.0, step=0.1)
        with col4:
            total_pesanan = st.number_input("Total Pesanan (kg)", min_value=0.0, step=0.1)

        col5, col6, col7 = st.columns(3)
        with col5:
            sisa = st.number_input("Sisa (kg)", min_value=0.0, step=0.1)
        with col6:
            tambah_pengambilan = st.number_input("Tambah Pengambilan (kg)", step=0.1)
        with col7:
            tambah_pesanan = st.number_input("Tambah Pesanan (kg)", step=0.1)

        tombol_prediksi = st.form_submit_button("Klasifikasi dan Simpan Data")

    if tombol_prediksi:
        if (total_pengambilan == 0 and total_pesanan == 0 and sisa == 0
                and tambah_pengambilan == 0 and tambah_pesanan == 0):
            st.warning("Mohon isi data operasional terlebih dahulu.")
            st.stop()

        waktu_input_wib = waktu_wib()

        data_input = pd.DataFrame([{
            "Tanggal": str(tanggal),
            "TotalPengambilan": total_pengambilan,
            "TotalPesanan": total_pesanan,
            "Sisa": sisa,
            "TambahPengambilan": tambah_pengambilan,
            "TambahPesanan": tambah_pesanan,
            "suhu_rata_rata": suhu_rata_rata
        }])

        data_prediksi, missing_cols, data_debug = siapkan_data_prediksi(data_input)

        if len(missing_cols) > 0:
            st.error("Terdapat kolom yang belum tersedia.")
            st.write(missing_cols)
            st.stop()

        jumlah_missing = data_prediksi[feature_cols].isnull().sum()
        if jumlah_missing.sum() > 0:
            st.error("Terdapat nilai kosong atau format tidak valid pada fitur model.")
            st.write(jumlah_missing[jumlah_missing > 0])
            st.stop()

        prediksi, prob_tidak_gangguan, prob_gangguan = prediksi_model(data_prediksi)
        hasil_prediksi = buat_tabel_hasil(data_prediksi, prediksi, prob_tidak_gangguan, prob_gangguan)

        st.subheader("Hasil Perhitungan Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Pengambilan", f"{data_prediksi['TotalPengambilan'].iloc[0]:.2f} kg")
        with col2:
            st.metric("Total Pesanan", f"{data_prediksi['TotalPesanan'].iloc[0]:.2f} kg")
        with col3:
            st.metric("Susut", f"{data_prediksi['Susut'].iloc[0]:.2f} kg")

        st.subheader("Hasil Klasifikasi")
        keterangan_prediksi = hasil_prediksi["Keterangan_Prediksi"].iloc[0]

        if keterangan_prediksi == "Gangguan":
            st.error("Hasil Klasifikasi: Gangguan")
        else:
            st.success("Hasil Klasifikasi: Tidak Gangguan")

        st.write("Detail hasil klasifikasi:")
        st.dataframe(hasil_prediksi, use_container_width=True)

        hasil_simpan = hasil_prediksi.copy()
        hasil_simpan.insert(0, "Waktu_Input", waktu_input_wib)

        if os.path.exists(RIWAYAT_PATH):
            data_lama = pd.read_csv(RIWAYAT_PATH)
            data_baru = pd.concat([data_lama, hasil_simpan], ignore_index=True)
            data_baru.to_csv(RIWAYAT_PATH, index=False)
        else:
            hasil_simpan.to_csv(RIWAYAT_PATH, index=False)

        st.success("Data input manual dan hasil klasifikasi berhasil disimpan.")
        st.info(
            "Catatan: di Streamlit Community Cloud, penyimpanan bersifat sementara dan bisa "
            "hilang saat aplikasi restart/sleep. Untuk riwayat permanen, download CSV di bagian bawah."
        )

# MODE 2: UPLOAD DATASET BARU

elif mode_prediksi == "Upload Dataset Baru":

    st.subheader("Upload Dataset Baru untuk Klasifikasi Gangguan")

    uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data_upload = baca_file_upload(uploaded_file)

        if data_upload is None:
            st.error("Format file tidak didukung. Gunakan CSV atau Excel.")
            st.stop()

        st.write("Preview dataset yang diupload:")
        st.dataframe(data_upload.head(), use_container_width=True)

        tombol_prediksi_file = st.button("Klasifikasi Gangguan")

        if tombol_prediksi_file:
            data_prediksi, missing_cols, data_debug = siapkan_data_prediksi(data_upload)

            if len(missing_cols) > 0:
                st.error("Dataset belum memiliki kolom yang dibutuhkan model.")
                st.write("Kolom yang belum tersedia:")
                st.write(missing_cols)
                st.stop()

            jumlah_missing = data_prediksi[feature_cols].isnull().sum()
            if jumlah_missing.sum() > 0:
                st.error("Terdapat nilai kosong atau format tidak valid pada fitur yang dibutuhkan model.")
                st.write("Jumlah nilai kosong per fitur:")
                st.write(jumlah_missing[jumlah_missing > 0])
                st.stop()

            prediksi_file, prob_tidak_gangguan_file, prob_gangguan_file = prediksi_model(data_prediksi)
            hasil_prediksi = buat_tabel_hasil(data_prediksi, prediksi_file, prob_tidak_gangguan_file, prob_gangguan_file)

            st.success("Klasifikasi gangguan berhasil dilakukan.")

            st.subheader("Ringkasan Hasil Klasifikasi Gangguan")
            jumlah_data = len(hasil_prediksi)
            jumlah_tidak_gangguan = (hasil_prediksi["Prediksi_Gangguan"] == 0).sum()
            jumlah_gangguan = (hasil_prediksi["Prediksi_Gangguan"] == 1).sum()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jumlah Data", int(jumlah_data))
            with col2:
                st.metric("Tidak Gangguan", int(jumlah_tidak_gangguan))
            with col3:
                st.metric("Gangguan", int(jumlah_gangguan))

            st.write("Hasil klasifikasi gangguan:")
            st.dataframe(hasil_prediksi, use_container_width=True)

            csv_prediksi = hasil_prediksi.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Hasil Klasifikasi Gangguan CSV",
                data=csv_prediksi,
                file_name="hasil_klasifikasi_gangguan.csv",
                mime="text/csv"
            )

# MENAMPILKAN RIWAYAT KLASIFIKASI MANUAL

st.markdown("---")
st.subheader("Riwayat Data Klasifikasi Manual")

if os.path.exists(RIWAYAT_PATH):
    riwayat = pd.read_csv(RIWAYAT_PATH)
    st.dataframe(riwayat, use_container_width=True)

    csv_download = riwayat.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Riwayat Klasifikasi Manual CSV",
        data=csv_download,
        file_name="riwayat_klasifikasi_manual.csv",
        mime="text/csv"
    )
else:
    st.info("Belum ada data klasifikasi manual yang tersimpan.")
