# Drone E99 - Face Recognition & Anti-Spoofing System

Sistem pengenalan wajah dan deteksi anti-spoofing secara real-time melalui kamera drone (RTSP stream) maupun webcam lokal, dirancang untuk keperluan skripsi.

## Overview

Sistem ini mengimplementasikan pipeline end-to-end untuk face recognition dan liveness detection yang berjalan secara real-time. Alur kerjanya dimulai dari capture frame via drone RTSP stream (atau webcam lokal), buffer frame menggunakan threaded capture untuk minimalkan latensi, lalu menjalankan face detection menggunakan model InsightFace Buffalo_sc. Setelah wajah terdeteksi, sistem mengekstrak embedding dan membandingkannya dengan database wajah menggunakan cosine similarity dengan threshold tertentu. Secara paralel, setiap wajah juga dicek keasliannya melalui modul anti-spoofing berbasis MiniFASNetV2 yang dijalankan via ONNX Runtime dengan akselerasi CUDA.

Sistem menyediakan 3 mode operasi yang modular:
- **Detection Only** (`main_detect.py`) — Hanya mendeteksi wajah tanpa identifikasi
- **Recognition Only** (`main_recog.py`) — Deteksi + pengenalan wajah tanpa liveness check
- **Full Pipeline** (`main.py`) — Deteksi + pengenalan wajah + anti-spoofing

Model anti-spoofing MiniFASNetV2 digunakan dari repositori [yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing).

## Tech Stack

| Komponen | Teknologi |
|---|---|
| Bahasa | Python 3.10 |
| Face Detection & Recognition | InsightFace (Buffalo_sc) |
| Anti-Spoofing | MiniFASNetV2 via ONNX Runtime (CUDA) |
| Computer Vision | OpenCV 4.9 |
| GPU Acceleration | CUDA 12.1 + onnxruntime-gpu |
| Numerical Computing | NumPy 1.26 |
| Deep Learning Backend | PyTorch 2.5 (InsightFace dependency) |

## Key Features

- **Real-time Face Detection** — Deteksi wajah secara real-time menggunakan InsightFace dengan resolusi input 320x320
- **Face Recognition + Similarity Threshold** — Pengenalan wajah berbasis cosine similarity dengan threshold 0.45 terhadap database embedding
- **Anti-Spoofing / Liveness Check** — Deteksi wajah palsu (foto, video) menggunakan MiniFASNetV2 dengan threshold real score > 0.85
- **Multi-Camera Support** — Kompatibel dengan stream RTSP drone maupun webcam lokal
- **Frame Skip Optimization** — Proses inference hanya dilakukan setiap N frame untuk menjaga performa FPS
- **Snapshot & Recording** — Fitur capture foto dan rekam video langsung dari live feed (tekan `S` / `R`)
- **Few-Shot Face Registration** — Registrasi wajah baru hanya membutuhkan beberapa foto referensi, embedding dirata-ratakan dan di-normalisasi L2

## Project Structure

```
drone_e99_face_recognition/
├── app.py                  # UI handler: snapshot, recording, dan overlay visual
├── camera_config.py        # Threaded RTSP stream capture untuk drone
├── webcam_config.py        # Threaded capture untuk webcam lokal
├── model.py                # Wrapper InsightFace untuk detection + recognition
├── face_detection.py       # Wrapper InsightFace untuk detection saja (buffalo_s)
├── MiniFASNetV2.py         # Modul anti-spoofing berbasis ONNX Runtime
├── main.py                 # Full pipeline: detect + recognize + liveness
├── main_detect.py          # Mode detection only
├── main_recog.py           # Mode detection + recognition (tanpa liveness)
├── capture_data.py         # Capture foto wajah via drone untuk dataset
├── data_register.py        # Ekstraksi embedding dari dataset & simpan ke face_db.pkl
├── requirements.txt        # Daftar dependensi Python
├── face_db.pkl             # Database embedding wajah (gitignored)
├── models/
│   └── MiniFASNetV2.onnx   # Model ONNX anti-spoofing
├── dataset/                # Folder dataset wajah per orang (gitignored)
│   └── <nama>/
│       └── *.jpg
└── inference_output/       # Output snapshot & recording (gitignored)
```

## Installation

### Prerequisites

- **Python 3.10** — Versi lain mungkin tidak kompatibel dengan beberapa dependensi
- **NVIDIA GPU** — Diperlukan untuk akselerasi CUDA pada ONNX Runtime dan InsightFace
- **CUDA Toolkit 12.1** — Sesuaikan versi dengan PyTorch dan onnxruntime-gpu yang terinstall
- **Git** — Untuk clone repository

### 1. Clone Repository

```bash
git clone <url-repository>
cd drone_e99_face_recognition
```

### 2. Install Dependencies

Disarankan menggunakan virtual environment:

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### 3. Download Model Anti-Spoofing

Unduh model `MiniFASNetV2.onnx` dari repositori [yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing) dan letakkan di folder:

```
models/MiniFASNetV2.onnx
```

> Catatan: Model InsightFace Buffalo_sc akan otomatis terunduh saat pertama kali dijalankan.

### 4. Siapkan Dataset Wajah

Buat folder untuk setiap orang di dalam direktori `dataset/`:

```
dataset/
└── ridho/
    ├── foto1.jpg
    ├── foto2.jpg
    └── foto3.jpg
```

Atau gunakan tool capture bawaan (via drone):

```bash
python capture_data.py
```

Masukkan nama subjek, lalu tekan `S` untuk mengambil foto dan `Q` untuk selesai.

### 5. Bangun Database Embedding

Setelah dataset siap, jalankan registrasi untuk mengekstrak embedding dan menyimpannya ke `face_db.pkl`:

```bash
python data_register.py
```

### 6. Jalankan Sistem

Pilih mode sesuai kebutuhan:

```bash
# Full pipeline (detection + recognition + anti-spoofing)
python main.py

# Detection only
python main_detect.py

# Detection + recognition (tanpa anti-spoofing)
python main_recog.py
```

**Kontrol saat runtime:**
| Tombol | Fungsi |
|---|---|
| `Q` | Keluar |
| `S` | Ambil snapshot |
| `R` | Mulai/berhenti rekam video |

## Engineering Value

- **Threaded Stream Capture** — Frame dibaca di thread terpisah untuk menghindari blocking I/O, sehingga frame selalu terbaru dan latensi minimal. Konfigurasi FFMPEG menggunakan `nobuffer` dan `low_delay` untuk stream RTSP.
- **Frame Skip Strategy** — Inference hanya dijalankan setiap 5 frame (configurable), frame di antaranya tetap menampilkan bounding box dari hasil inference terakhir, menjaga FPS tetap tinggi.
- **Cosine Similarity + Threshold** — Embedding wajah dinormalisasi L2, lalu dihitung cosine similarity terhadap database. Threshold 0.45 memberikan keseimbangan antara presisi dan recall.
- **Few-Shot Embedding Averaging** — Registrasi wajah tidak membutuhkan banyak foto. Embedding dari beberapa foto dirata-ratakan dan di-normalisasi, menghasilkan representasi yang lebih robust terhadap variasi pose/pencahayaan.
- **Modular Architecture** — Tiga mode operasi (detect, recognize, full) memungkinkan pengembangan dan testing secara bertahap tanpa perlu mengubah keseluruhan codebase.
- **Separate Liveness Module** — Anti-spoofing dijalankan terpisah dari recognition pipeline, memudahkan penggantian model liveness tanpa mempengaruhi logic recognition.

## Future Improvements

- **Multi-face Parallel Recognition** — Proses recognition untuk multiple wajah secara simultan menggunakan batch inference
- **Alert System** — Integrasi notifikasi (suara / push notification) saat wajah spoof terdeteksi
- **Persistent Tracking ID** — Implementasi object tracking (misalnya DeepSORT) agar identitas wajah konsisten antar frame tanpa perlu re-recognize setiap infer step
- **Web Dashboard** — Dashboard berbasis web (Flask/FastAPI) untuk monitoring jarak jauh
- **Model Upgrade** — Migrasi ke model recognition yang lebih ringan (misalnya ArcFaceONNX) untuk edge deployment
- **Kalibrasi Threshold Otomatis** — Sistem auto-calibration untuk similarity threshold dan liveness threshold berdasarkan kondisi pencahayaan
