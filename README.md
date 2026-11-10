# Sunda-Jawa AI

Prototipe aplikasi identifikasi **Sunda Kasar, Sunda Lemes, Jawa, Aceh, dan Palembang** melalui teks, audio, dan foto berbasis AI.

## Kebutuhan sistem
- Windows 10/11
- Python 3.10 atau lebih baru
- VS Code (opsional)

---

## Instalasi dasar

Buka terminal pada folder proyek:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Buka alamat lokal yang ditampilkan Streamlit, biasanya http://localhost:8501.

Jika PowerShell memblokir aktivasi:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

## Fitur teks

Tersedia setelah instalasi dasar dan pelatihan model. Ketik kalimat dalam Sunda Kasar, Sunda Lemes, Jawa, Aceh, atau Palembang untuk melihat hasil klasifikasi.

---

## Fitur foto (OCR — Tesseract)

### Instalasi Tesseract OCR di Windows

1. Unduh installer **Tesseract for Windows** dari tautan resmi:  
   👉 https://github.com/UB-Mannheim/tesseract/wiki

2. Jalankan installer. Saat memilih komponen, **centang**:
   - `Additional language data (download)` → `Indonesian (ind)` dan `English (eng)`

3. Lokasi instalasi default: `C:\Program Files\Tesseract-OCR\`  
   Catat lokasi ini jika mengubahnya.

4. **Tambahkan ke PATH Windows** (opsional tapi direkomendasikan):
   - Buka **System Properties → Environment Variables**
   - Tambahkan `C:\Program Files\Tesseract-OCR` ke variabel `Path`

5. Tutup dan buka kembali terminal, lalu jalankan aplikasi.

### Aplikasi otomatis mendeteksi Tesseract

Aplikasi mencari `tesseract.exe` di urutan berikut:
1. PATH sistem
2. Variabel lingkungan `TESSERACT_CMD`
3. Lokasi umum: `C:\Program Files\Tesseract-OCR\`, `C:\Program Files (x86)\Tesseract-OCR\`, dll.

Jika Tesseract terinstal di lokasi non-standar, set variabel lingkungan:
```powershell
$env:TESSERACT_CMD = "C:\LokasiKhusus\tesseract.exe"
streamlit run app.py
```

Atau tambahkan ke `utils/ocr.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\LokasiKhusus\tesseract.exe"
```

---

## Fitur audio (Whisper + FFmpeg)

### Instalasi dependensi audio

```powershell
pip install -r requirements-audio.txt
```

### Instalasi FFmpeg di Windows

Whisper membutuhkan FFmpeg untuk memproses format audio non-WAV.

1. Unduh FFmpeg dari: https://www.gyan.dev/ffmpeg/builds/  
   (pilih `ffmpeg-release-essentials.zip`)

2. Ekstrak ke folder, contoh: `C:\ffmpeg\`

3. Tambahkan `C:\ffmpeg\bin` ke PATH Windows:
   - **System Properties → Environment Variables → Path → Edit → New**
   - Ketik: `C:\ffmpeg\bin`

4. Tutup terminal dan buka kembali. Verifikasi:
   ```powershell
   ffmpeg -version
   ```

### Catatan audio
- Izinkan mikrofon untuk `localhost` di pengaturan izin situs browser.
- Periksa input microphone Windows: **Settings → System → Sound → Input**.
- Model Whisper diunduh otomatis saat pertama kali digunakan (~74 MB untuk `tiny`, ~142 MB untuk `base`).
- Model Whisper di-cache di memori: tidak dimuat ulang setiap kali tombol ditekan.
- Gunakan model `tiny` untuk perangkat RAM rendah; `base` untuk akurasi lebih baik.

---

## Memperbaiki dataset

Dataset berada di `dataset/data_bahasa.csv` dengan kolom:
- `text`: kalimat
- `label`: `Sunda Kasar`, `Sunda Lemes`, `Jawa`, `Aceh`, atau `Palembang`

Setelah dataset berubah, latih ulang model:
```powershell
python train_model.py
```

---

## Memperbarui dependensi

```powershell
pip install --upgrade -r requirements.txt
pip install -r requirements-audio.txt
```

---

## Troubleshooting

| Masalah | Solusi |
|--------|--------|
| `tesseract is not installed or it's not in your PATH` | Instal Tesseract dari link di atas, atau set `TESSERACT_CMD` |
| `Paket Whisper belum terpasang` | Jalankan `pip install -r requirements-audio.txt` |
| `FFmpeg not found` | Instal FFmpeg dan tambahkan ke PATH |
| Audio duration = 0 | Pastikan format file benar (WAV/MP3/M4A) dan ukuran file > 1500 byte |
| `Model belum tersedia` | Jalankan `python train_model.py` |
| PowerShell blokir aktivasi venv | Jalankan `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| OCR hasilnya kosong | Gunakan gambar yang lebih tajam, terang, dan resolusi tinggi |

---

## Catatan akademik penting

Dataset bawaan hanya contoh awal untuk menguji alur aplikasi, bukan dataset penelitian yang memadai. Skor evaluasi dari dataset kecil ini tidak boleh diklaim sebagai akurasi umum bahasa Sunda/Jawa. Untuk tugas akhir/proyek kuliah:
- Perluas dataset dengan sumber yang jelas dan data bervariasi
- Validasi label oleh penutur yang kompeten
- Pisahkan data latih/uji tanpa duplikasi
- Uji juga kalimat campuran, bahasa Indonesia, teks sangat pendek, serta dialek yang berbeda

Whisper adalah sistem transkripsi, bukan pendeteksi bahasa daerah yang dijamin akurat. Transkripsi bahasa daerah bisa keliru; hasil klasifikasi audio bergantung pada kualitas teks transkripsi.
