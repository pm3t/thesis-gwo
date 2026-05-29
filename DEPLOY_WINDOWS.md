# Panduan Deploy & Menjalankan Aplikasi di Windows

Dokumen ini berisi panduan langkah-demi-langkah untuk menyiapkan lingkungan, menginstal dependensi, dan menjalankan aplikasi **Optimasi Model Ensemble via GWO untuk Peramalan Penjualan** di Sistem Operasi Windows.

---

## 📋 Prasyarat Sistem

Sebelum memulai, pastikan sistem Windows Anda telah memenuhi persyaratan berikut:

1. **Python (Versi 3.9 hingga 3.11 direkomendasikan)**
   * Unduh installer resmi dari [python.org](https://www.python.org/downloads/).
   * > [!IMPORTANT]
     > Saat menjalankan installer, pastikan Anda mencentang opsi **"Add python.exe to PATH"** di bagian bawah jendela instalasi sebelum menekan tombol *Install Now*.
2. **Koneksi Internet** (Untuk mengunduh pustaka/library yang dibutuhkan).

---

## 🚀 Langkah-Langkah Instalasi

### Langkah 1: Buka Terminal (Command Prompt / PowerShell)
1. Tekan tombol `Windows + S`, ketik `cmd` atau `PowerShell`.
2. Arahkan direktori terminal ke dalam folder proyek Anda. Contoh:
   ```cmd
   cd C:\Users\Username\Projects\Thesis
   ```

### Langkah 2: Buat Lingkungan Virtual (Virtual Environment)
Disarankan untuk membuat *virtual environment* agar pustaka proyek tidak bentrok dengan instalasi Python global di sistem Anda.
```cmd
python -m venv .venv
```

### Langkah 3: Aktifkan Virtual Environment
Aktifkan lingkungan virtual yang telah dibuat berdasarkan terminal yang Anda gunakan:

* **Menggunakan Command Prompt (cmd):**
  ```cmd
  .venv\Scripts\activate
  ```
* **Menggunakan PowerShell:**
  ```powershell
  .venv\Scripts\activate.ps1
  ```
  > [!NOTE]
  > Jika Anda mendapatkan error kebijakan eksekusi (*Execution Policy*) di PowerShell, jalankan perintah berikut terlebih dahulu untuk memberikan izin eksekusi script:
  > `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`

Setelah berhasil aktif, Anda akan melihat tanda `(.venv)` di depan baris input terminal Anda.

### Langkah 4: Instal Pustaka Dependensi
Gunakan `pip` untuk menginstal seluruh pustaka yang terdaftar di `requirements.txt`:
```cmd
pip install -r requirements.txt
```
*Proses ini akan menginstal `CustomTkinter`, `Pandas`, `NumPy`, `Scikit-Learn`, `Statsmodels`, `Matplotlib`, `Seaborn`, `Pillow`, dan `TensorFlow`.*

---

## 🖥️ Menjalankan Aplikasi

Setelah semua pustaka terinstal, Anda dapat langsung menjalankan program utama dengan perintah:
```cmd
python main.py
```

---

## ⚡ Membuat Pintasan Otomatis (Desktop Shortcut)

Agar Anda tidak perlu membuka terminal dan mengetik perintah di atas setiap kali ingin menjalankan aplikasi, Anda dapat membuat file *batch* (.bat) untuk menjalankannya dengan sekali klik ganda.

1. Di dalam folder proyek Anda, buat file baru bernama **`Jalankan_Aplikasi.bat`**.
2. Klik kanan file tersebut, pilih **Edit** atau buka dengan Notepad.
3. Rekatkan kode berikut ke dalam file tersebut:
   ```bat
   @echo off
   echo Mengaktifkan Virtual Environment...
   call .venv\Scripts\activate
   echo Menjalankan Aplikasi Peramalan...
   python main.py
   pause
   ```
4. Simpan file tersebut (`Ctrl + S`).
5. **Selesai!** Sekarang Anda cukup melakukan klik ganda (double-click) pada file `Jalankan_Aplikasi.bat` untuk membuka aplikasi secara otomatis.

---

## 🔍 Penyelesaian Masalah (Troubleshooting)

### 1. `python` atau `pip` Tidak Dikenali di Terminal
* **Penyebab:** Python belum ditambahkan ke variabel sistem (`PATH`).
* **Solusi:** Instal ulang Python dan pastikan mencentang **"Add python.exe to PATH"**, atau tambahkan lokasi folder instalasi Python Anda ke variabel lingkungan (*Environment Variables*) secara manual.

### 2. Error Pustaka TensorFlow (Terkait GPU)
* **Penyebab:** Peringatan bahwa TensorFlow mencari perangkat GPU namun tidak menemukannya.
* **Solusi:** Abaikan peringatan tersebut. Program akan otomatis beralih menggunakan CPU untuk pemrosesan data, yang sudah sangat mumpuni untuk ukuran dataset yang digunakan pada program ini.

### 3. Folder `Analysis` Tidak Ditemukan
* **Penyebab:** Beberapa fungsi penyimpanan statistik memerlukan folder bernama `Analysis` di direktori utama.
* **Solusi:** Jika folder tersebut belum ada, buat folder baru secara manual bernama **`Analysis`** dan **`Dataset`** sejajar dengan file `main.py`.
