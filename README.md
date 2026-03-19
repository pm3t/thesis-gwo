# Sales Forecasting with Ensemble GWO

Aplikasi desktop untuk peramalan penjualan menggunakan optimasi bobot model ensemble dengan algoritma Grey Wolf Optimizer (GWO). Proyek ini dikembangkan sebagai bagian dari penelitian tesis.

## Fitur Utama

- **Load Dataset**: Memuat file CSV (kompatibel dengan dataset Store Sales Kaggle).
- **Preprocessing**: Pembersihan data, agregasi harian, penanganan outlier, dan pembagian data training/testing kronologis.
- **Visualisasi**: Plot deret waktu (Time Series), dekomposisi musiman, dan distribusi data.
- **Model Baseline**: Implementasi Moving Average (MA), Exponential Smoothing (ES), dan Linear Regression (LR).
- **Optimasi GWO**: Implementasi custom Grey Wolf Optimizer untuk mencari bobot optimal dalam penggabungan model ensemble (Weighted Ensemble).
- **Ensemble Model**: Peramalan menggunakan gabungan model dengan bobot hasil optimasi GWO.
- **Evaluasi**: Metrik lengkap (MAE, MSE, RMSE, MAPE, R²) dan perbandingan performa antar model.
- **Export**: Simpan hasil prediksi ke CSV dan simpan bobot optimal ke file teks.

## Struktur Proyek

```text
Thesis/
├── main.py                 # Titik masuk aplikasi
├── ui/                     # Komponen Antarmuka Pengguna
│   ├── main_window.py      # Layout utama dengan tab
│   ├── data_tab.py         # Tab manajemen data
│   ├── visualize_tab.py    # Tab visualisasi data
│   ├── model_tab.py        # Tab model baseline
│   ├── gwo_tab.py          # Tab optimasi GWO
│   └── ensemble_tab.py     # Tab hasil ensemble & perbandingan
├── models/                 # Implementasi Model Peramalan
│   ├── moving_average.py   # MA
│   ├── exponential_smoothing.py # ES
│   ├── linear_regression.py # LR
│   └── ensemble.py         # Weighted Ensemble
├── optimizers/             # Implementasi Algoritma Optimasi
│   └── gwo.py              # Grey Wolf Optimizer (Custom)
├── utils/                  # Fungsi Utilitas
│   ├── data_processor.py   # Preprocessing data
│   ├── metrics.py          # Metrik evaluasi
│   └── visualizer.py       # Fungsi plotting Matplotlib
└── requirements.txt        # Daftar dependensi
```

## Instalasi

1. Pastikan Anda memiliki Python 3.9 atau versi yang lebih baru.
2. Clone repository ini atau download source code.
3. Instal dependensi yang dibutuhkan menggunakan pip:

```bash
pip install -r requirements.txt
```

## Cara Penggunaan

1. Jalankan aplikasi:
   ```bash
   python main.py
   ```
2. **Tab DATA**: Klik "Load CSV" dan pilih file `train.csv`. Pilih kolom tanggal (`date`) dan target (`sales`). Klik "Preprocess" lalu "Split Data".
3. **Tab VISUALISASI**: Pilih jenis plot yang diinginkan (contoh: Time Series Plot) dan klik "Generate Plot".
4. **Tab BASELINE MODEL**: Klik "Train & Predict" pada masing-masing model (MA, ES, LR) untuk mendapatkan hasil awal.
5. **Tab OPTIMASI GWO**: Atur parameter (Populasi & Iterasi) lalu klik "Run GWO Optimization". Perhatikan kurva konvergensi hingga mencapai nilai minimum.
6. **Tab ENSEMBLE**: Klik "Run Ensemble" untuk melihat hasil penggabungan model dan perbandingannya dengan model baseline.
7. **Export**: Gunakan tombol "Export Results" untuk menyimpan hasil ke file CSV.

## Lisensi
Proyek ini dibuat untuk tujuan penelitian akademis.
# thesis-gwo
