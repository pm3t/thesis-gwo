# Panduan Prompt Draft Bab IV & Bab V Tesis

Dokumen ini berisi kumpulan **prompt siap pakai** (dalam bahasa Indonesia dengan gaya bahasa akademis formal) yang dapat Anda salin dan tempel ke AI (seperti ChatGPT, Claude, atau Gemini) untuk menulis draft awal Bab IV (Hasil dan Pembahasan) serta Bab V (Kesimpulan dan Saran) berdasarkan data dari laporan yang Anda ekspor.

---

## Cara Penggunaan
1. Buka file hasil ekspor laporan teks Anda (`Hasil_Eksperimen_Tesis.txt`).
2. Pilih prompt di bawah ini yang sesuai dengan bagian bab yang ingin Anda buat.
3. Salin prompt tersebut, lalu ganti teks penanda seperti `[PASTE_DATA_DISINI]` dengan data dari laporan teks Anda.
4. Kirim ke AI untuk menghasilkan draf tulisan akademis.

---

## 1. Prompt Bab IV: Deskripsi Data & Uji Stasioneritas (ADF)

```text
Bertindaklah sebagai Asisten Penulisan Tesis Akademik bidang Ilmu Komputer/Informatika dan Sains Data. 
Saya sedang menulis Bab IV (Hasil dan Pembahasan) Tesis saya yang berjudul "Analisis Peramalan Deret Waktu Menggunakan Model Ensemble dengan Optimasi Bobot GWO dan PSO". 

Bantu saya menulis draft sub-bab tentang "Deskripsi Dataset dan Analisis Stasioneritas Data". Tulislah dalam bahasa Indonesia akademis yang formal, objektif, dan ilmiah (gaya penulisan baku tesis Indonesia).

Berikut adalah data statistik deskriptif dan hasil uji Augmented Dickey-Fuller (ADF) dari program saya:

---
[PASTE BAGIAN 1 (STATISTIK DESKRIPTIF DATASET) DAN BAGIAN 2 (ADF TEST RESULTS) DARI FILE TXT LAPORAN ANDA DI SINI]
---

Dalam draf tulisan Anda, tolong bahas:
1. Penjelasan singkat mengenai karakteristik statistik data (rata-rata, varians/standar deviasi, nilai minimum dan maksimum).
2. Hasil analisis uji stasioneritas menggunakan metode ADF (jelaskan nilai ADF statistic, p-value, dan bandingkan dengan critical values pada signifikansi 1%, 5%, dan 10%).
3. Berikan kesimpulan apakah data tersebut stasioner atau tidak, serta apa implikasinya bagi pemodelan peramalan (jelaskan mengapa data non-stasioner membutuhkan pendekatan yang lebih adaptif seperti model ensemble yang dioptimasi).
```

---

## 2. Prompt Bab IV: Hasil Pengujian Model Baseline (MA, ES, RNN)

```text
Bertindaklah sebagai Asisten Penulisan Tesis Akademik. Saya sedang menyusun Bab IV sub-bab "Hasil Pengujian Model Baseline (Moving Average, Exponential Smoothing, dan Simple RNN)". 

Bantu saya membuat draft tulisan akademis formal yang memaparkan hasil evaluasi masing-masing model individu ini sebelum dilakukan penggabungan (ensemble).

Berikut adalah metrik performa model individu (baseline) dari eksperimen saya:

---
[PASTE BAGIAN 3 (PERFORMA MODEL INDIVIDU / BASELINE) DARI FILE TXT LAPORAN ANDA DI SINI]
---

Tolong bahas hal-hal berikut dalam draf tulisan:
1. Deskripsikan performa masing-masing model (MA, ES, RNN) berdasarkan metrik MAPE, MAE, MSE, RMSE, dan R-squared (R2).
2. Lakukan analisis komparatif sederhana: model mana yang menghasilkan error terkecil (performa terbaik) dan model mana yang paling lemah, serta berikan penjelasan teoritis singkat mengapa model tersebut lebih unggul/lemah pada dataset ini.
3. Jelaskan bahwa variasi performa dari ketiga model ini menunjukkan keunikan masing-masing model dalam menangkap pola linear (MA, ES) dan non-linear (Simple RNN), sehingga memperkuat alasan ilmiah untuk menggabungkannya ke dalam model ensemble guna meminimalkan risiko prediksi salah satu model individu.
```

---

## 3. Prompt Bab IV: Hasil Optimasi & Uji Kestabilan (GWO vs PSO)

```text
Bertindaklah sebagai Asisten Penulisan Tesis Akademik. Saya sedang menulis Bab IV sub-bab "Hasil Optimasi Bobot Ensemble dan Uji Kestabilan Algoritma: GWO vs PSO". 

Bantu saya membuat draft akademis formal yang membandingkan performa pencarian solusi optimal dan tingkat kestabilan (stability analysis) antara algoritma Grey Wolf Optimizer (GWO) dan Particle Swarm Optimization (PSO) berdasarkan pengujian berulang sebanyak 30 run (multi-run).

Berikut adalah data kestabilan 30 run dari program saya:

---
[PASTE BAGIAN 4 (HASIL OPTIMASI & UJI KESTABILAN GWO / PSO) DARI FILE TXT LAPORAN ANDA DI SINI]
---

Tolong bahas hal-hal berikut dalam draf tulisan:
1. Bandingkan performa terbaik (Best), terburuk (Worst), rata-rata (Mean), dan Standar Deviasi (Std Dev) dari nilai MAPE yang dicapai oleh GWO dan PSO selama 30 kali run acak.
2. Analisis kestabilan: jelaskan implikasi nilai Standar Deviasi (Std Dev) yang kecil pada salah satu algoritma sebagai penanda tingkat konsistensi dan kestabilan algoritma metaheuristik yang bersifat stokastik.
3. Bahas distribusi rata-rata bobot (w1 untuk MA, w2 untuk ES, w3 untuk RNN) yang dihasilkan oleh masing-masing optimizer, serta model mana yang mendapatkan kontribusi bobot paling dominan.
```

---

## 4. Prompt Bab IV: Perbandingan Ensemble & Uji Signifikansi (Wilcoxon)

```text
Bertindaklah sebagai Asisten Penulisan Tesis Akademik. Saya sedang menulis Bab IV sub-bab "Analisis Hasil Model Ensemble dan Pengujian Signifikansi Statistik".

Bantu saya menyusun draft tulisan ilmiah formal yang mengevaluasi model Ensemble teroptimasi (GWO/PSO) dibandingkan dengan metode rata-rata biasa (Equal Average) dan model baseline terbaik, serta menyajikan hasil uji statistik signifikansi Wilcoxon Signed-Rank Test.

Berikut adalah data metrik perbandingan ensemble dan hasil uji Wilcoxon dari eksperimen saya:

---
[PASTE BAGIAN 5 (PERBANDINGAN MODEL ENSEMBLE) DAN BAGIAN 6 (UJI PERBANDINGAN SIGNIFIKANSI STATISTIK) DARI FILE TXT LAPORAN ANDA DI SINI]
---

Tolong bahas hal-hal berikut dalam draf:
1. Bandingkan performa model GWO/PSO Ensemble dengan Equal Average dan model individu terbaik. Sebutkan penurunan nilai MAPE (Improvement %) secara kuantitatif.
2. Paparkan hasil uji Wilcoxon Signed-Rank Test (nilai statistik uji, p-value, dan tingkat signifikansi).
3. Berikan argumen ilmiah bahwa berdasarkan p-value (apakah p < 0.05), hipotesis nol ditolak, yang membuktikan secara statistik bahwa peningkatan akurasi dari model ensemble teroptimasi yang diusulkan adalah signifikan secara nyata, bukan merupakan hasil variasi acak (chance variation).
```

---

## 5. Prompt Bab V: Kesimpulan & Saran

```text
Bertindaklah sebagai Asisten Penulisan Tesis Akademik. Saya sedang menyusun Bab V (Kesimpulan dan Saran) untuk tesis saya.

Bantu saya membuat draft tulisan Bab V yang terstruktur rapi menjadi bagian "Kesimpulan" dan "Saran" berdasarkan keseluruhan data eksperimen berikut:

---
[PASTE SELURUH DATA DARI FILE TXT LAPORAN ANDA DI SINI]
---

Tolong susun draf dengan ketentuan:
1. Bagian Kesimpulan harus menjawab tujuan penelitian secara tegas: 
   - Efektivitas model ensemble dibandingkan model tunggal baseline.
   - Algoritma optimasi mana (GWO vs PSO) yang lebih unggul secara akurasi dan kestabilan.
   - Hasil validasi statistik (Wilcoxon Test) yang membuktikan keandalan model ensemble yang diusulkan.
2. Bagian Saran harus memberikan poin-poin konstruktif untuk pengembangan penelitian selanjutnya (misalnya: eksplorasi parameter tuning, penggunaan model penyusun ensemble yang lebih kompleks seperti LSTM/GRU/Transformer, atau pengujian pada karakteristik dataset time series yang berbeda).
```
