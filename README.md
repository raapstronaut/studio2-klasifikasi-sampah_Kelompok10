# Studio 2 — Klasifikasi Sampah · Kelompok 10

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1 | Cikal Galih Nur Arifin | 123140109 | Model Training & Evaluation |
| 2 | Muhamad Rafi Ilham | 123140173 | Development & Integration |

## Ringkasan Pendekatan

Proyek ini mengimplementasikan CNN menggunakan **full fine-tuning ResNet-18 pretrained ImageNet** untuk mengklasifikasikan lima jenis sampah: `botol_plastik`, `gelas_plastik`, `kertas_kardus`, `saset_kemasan`, dan `styrofoam`.

- **Arsitektur / backbone:** ResNet-18 pretrained ImageNet dengan classification head 5 kelas.
- **Strategi pelatihan:** Fine-tuning seluruh parameter selama 8 epoch, batch size 32, AdamW (learning rate 0.0003, weight decay 0.0001), CosineAnnealingLR, dan CrossEntropyLoss dengan label smoothing 0.1.
- **Augmentasi:** RandomResizedCrop, RandomHorizontalFlip, dan ColorJitter.
- **Dataset:** 2.753 gambar dari dataset resmi Studio 2, dengan sekitar 85% data training dan 15% validasi lokal (410 gambar).
- **Data tambahan:** Tidak ada.
- **Ukuran input & preprocessing:** 224 × 224 piksel. Validasi menggunakan Resize(256), CenterCrop(224), ToTensor, dan normalisasi ImageNet.
- **Metrik utama:** Macro-F1. Checkpoint baseline dipilih berdasarkan akurasi validasi terbaik.

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | `b17b55c` | 0.9295 | Menunggu hasil |
| Preview 2 (21 Okt) | - | - | - |
| Final (25 Okt) | - | - | - |

### Catatan Eksperimen

**Eksperimen 01 — Baseline ResNet-18**

Model dilatih selama 8 epoch menggunakan transfer learning dan memperoleh hasil evaluasi lokal:

- **Macro-F1:** 0.9295
- **Akurasi:** 0.9341
- **F1 terendah:** `kertas_kardus` (0.8738)
- **Waktu evaluasi:** 4.6 detik pada GPU Tesla T4

Submission berhasil diuji menggunakan `evaluate.py` resmi dari clone GitHub dengan exit code 0. Kelas `kertas_kardus` menjadi prioritas optimasi berikutnya.

*Hasil di atas berasal dari validasi lokal, bukan hidden validation set milik tim pengajar.*

## Cara Reproduksi

### 1. Clone Repository

```bash
git clone https://github.com/raapstronaut/studio2-klasifikasi-sampah_Kelompok10.git
cd studio2-klasifikasi-sampah_Kelompok10
```

### 2. Instalasi Dependencies

```bash
pip install -r requirements.txt
```

### 3. Persiapan Dataset

Download [dataset resmi Studio 2](https://drive.google.com/drive/folders/1_XcZoDFRznHB7fI603VHzdFxreg3ma60?usp=sharing) dan ekstrak sehingga folder `train/` berada di root proyek.

Dataset tidak diunggah ke GitHub sesuai ketentuan dosen.

### 4. Training Model

```bash
python train_baseline.py --data ./train --epochs 8 --batch-size 32
```

Proses training menghasilkan folder `my_train/`, `my_val/`, dan checkpoint `submission/model.pth`.

### 5. Evaluasi Model

```bash
python evaluate.py --submission ./submission --data ./my_val
```

Evaluasi menghasilkan Macro-F1, akurasi, F1-score setiap kelas, dan confusion matrix.

## Struktur Repositori

```text
studio2-klasifikasi-sampah_Kelompok10/
├── README.md
├── .gitignore
├── requirements.txt
├── evaluate.py
├── train_baseline.py
└── submission/
    ├── model.py
    └── model.pth
```

- `model.py`: Implementasi `get_model()` dan `get_transform()`.
- `model.pth`: State dictionary model hasil fine-tuning.
- `evaluate.py`: Skrip evaluasi resmi dosen.
- `train_baseline.py`: Kode pelatihan ResNet-18.

Dataset dan split validasi lokal tidak disertakan dalam repository.