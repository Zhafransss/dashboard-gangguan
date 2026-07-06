# Cara Deploy ke Streamlit Community Cloud

## Isi folder ini
```
streamlit_app/
├── app.py                              <- aplikasi Streamlit (sudah disesuaikan, tanpa Colab/Drive)
├── requirements.txt                    <- daftar library
├── model_random_forest_gangguan.pkl    <- model yang sudah dilatih ulang (akurasi ~90.7%)
└── fitur_model.pkl                     <- daftar fitur model
```

## Langkah 1 — Upload ke GitHub
1. Buat akun GitHub kalau belum punya (https://github.com/join)
2. Buat repository baru (boleh **Public** atau **Private**), misal beri nama `dashboard-gangguan`
3. Upload **semua isi folder `streamlit_app/`** ini ke root repo tersebut
   (lewat web GitHub: "Add file" → "Upload files", atau pakai `git push` kalau familiar)

## Langkah 2 — Deploy di Streamlit Community Cloud
1. Buka https://share.streamlit.io/
2. Klik **"Sign in with GitHub"** dan izinkan akses
3. Klik **"Create app"** / **"New app"**
4. Isi:
   - Repository: pilih repo yang kamu buat tadi
   - Branch: `main` (default)
   - Main file path: `app.py`
5. Klik **"Deploy"** — tunggu beberapa menit, app akan otomatis online di
   `https://nama-app-kamu.streamlit.app`

## Langkah 3 (opsional) — Pakai domain sendiri dari RumahWeb
1. Di dashboard app Streamlit Cloud kamu, buka **Settings → General** untuk melihat domain default
2. Di cPanel RumahWeb, buka **Zone Editor / DNS Management** untuk domain kamu
3. Tambahkan **CNAME record**: nama subdomain (misal `dashboard`) mengarah ke domain `streamlit.app` kamu
4. Tunggu propagasi DNS (bisa sampai 24-48 jam)

## Catatan penting
- **Penyimpanan riwayat bersifat sementara** — Streamlit Community Cloud tidak punya penyimpanan
  permanen. Data di `riwayat_prediksi.csv` bisa hilang saat app di-restart atau "tidur" karena lama
  tidak diakses. Kalau butuh riwayat permanen, gunakan tombol **Download CSV** setelah setiap prediksi,
  atau pertimbangkan database eksternal (misalnya Google Sheets API) nanti kalau perlu.
- Kalau file model kamu nanti membesar (>100MB), GitHub akan menolak upload biasa — perlu **Git LFS**.
  Untuk saat ini ukurannya masih kecil jadi aman.
- Kalau ingin retrain model dengan dataset baru, jalankan ulang `train_model.py` dari paket Flask
  sebelumnya, lalu ganti file `.pkl` di repo GitHub ini.
