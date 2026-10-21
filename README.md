# Studio 2 — Klasifikasi Sampah · Kelompok 10

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi |
|----|--------------|-----|------------|
| 1 | Cikal Galih Nur Arifin | 123140109 | Pengelolaan GitHub, integrasi submission dan dokumentasi Preview 2 |
| 2 | Muhamad Rafi Ilham | 123140173 | Eksperimen training, evaluasi, dan audit submission Preview 2 |

## Ringkasan Pendekatan

Proyek mengklasifikasikan lima jenis sampah: `botol_plastik`, `gelas_plastik`, `kertas_kardus`, `saset_kemasan`, dan `styrofoam`. Dataset resmi berisi 2.753 gambar. Validasi lokal menggunakan split per kelas sebesar 15% dengan seed 42: **2.343 gambar training** dan **410 gambar validasi**. Tidak digunakan dataset tambahan.

- **Preview 1:** full fine-tuning **ResNet-18 pretrained ImageNet**, 8 epoch, batch size 32, AdamW (learning rate 3e-4, weight decay 1e-4), CosineAnnealingLR, label smoothing 0.1, dengan checkpoint terbaik menurut akurasi validasi.
- **Preview 2 (submission aktif):** full fine-tuning **EfficientNet-B0 pretrained ImageNet (`IMAGENET1K_V1`)**, 8 epoch, batch size 24, AdamW (learning rate 1e-4, weight decay 1e-4), CosineAnnealingLR, label smoothing 0.05, dengan checkpoint terbaik menurut **Macro-F1** validasi lokal.
- **Augmentasi Preview 2:** RandomResizedCrop 224 × 224 (scale 0.7–1.0, bicubic), RandomHorizontalFlip, ColorJitter (0.15, 0.15, 0.15), dan normalisasi ImageNet.
- **Validasi Preview 2:** transform standar `EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()`.
- **Output:** logits lima kelas dengan urutan alfabetis sesuai `evaluate.py`.

## Hasil

| Putaran / model | Commit submission | Macro-F1 lokal | Akurasi lokal | Macro-F1 resmi asdos |
|----------------|-------------------|----------------|---------------|----------------------|
| Preview 1 — ResNet-18 | `b17b55c` | 0.9295 | 0.9341 | 0.9172 |
| Preview 2 — EfficientNet-B0 | `4cf2c6a` | **0.9433** | **0.9463** | Menunggu evaluasi |
| Final | — | — | — | — |

**Catatan:** Skor lokal dihitung pada 410 gambar `my_val`, bukan pada *hidden validation set* asdos. Karena datasetnya berbeda, skor lokal dan resmi tidak dapat dibandingkan langsung. Hasil training ulang juga bisa sedikit berbeda akibat proses stokastik dan lingkungan komputasi.

### Eksperimen Preview 2

| Eksperimen | Metode | Macro-F1 lokal tertinggi | Keputusan |
|------------|--------|-------------------------|-----------|
| Baseline | ResNet-18 Preview 1 | 0.9295 | Pembanding |
| 01 | Continued fine-tuning ResNet-18, LR 2e-5 | 0.9245 | Tidak dipakai |
| 02 | Training ResNet-18 dengan class-balanced sampling | 0.9239 | Tidak dipakai |
| 03 | EfficientNet-B0 pretrained ImageNet | **0.9433** (epoch 5 dari 8) | **Dipilih** |

F1 setiap kelas pada checkpoint EfficientNet-B0 terpilih:

| Kelas | F1 |
|-------|----|
| `botol_plastik` | 0.9806 |
| `gelas_plastik` | 0.9351 |
| `kertas_kardus` | 0.9038 |
| `saset_kemasan` | 0.9453 |
| `styrofoam` | 0.9515 |

Model Preview 2 telah diaudit menggunakan `evaluate.py` resmi dari clone GitHub bersih, commit `4cf2c6a`, dengan **exit code 0**, skor lokal **Macro-F1 0.9433** dan akurasi **0.9463**. SHA-256 bobot GitHub juga identik dengan checkpoint terbaik yang dihasilkan di Colab.

## Cara Reproduksi

### 1. Clone repository dan instal dependensi

```bash
git clone https://github.com/raapstronaut/studio2-klasifikasi-sampah_Kelompok10.git
cd studio2-klasifikasi-sampah_Kelompok10
pip install -r requirements.txt
```

### 2. Siapkan dataset

Unduh [dataset resmi Studio 2](https://drive.google.com/drive/folders/1_XcZoDFRznHB7fI603VHzdFxreg3ma60?usp=sharing) dan ekstrak sehingga tersedia `train/<nama_kelas>/*.jpg` di root proyek. Dataset tidak diunggah ke GitHub.

### 3. Jalankan training Preview 2

```bash
python train_preview2.py --data ./train --epochs 8 --batch-size 24
```

Skrip menyiapkan `my_train/` dan `my_val/` dengan seed 42 bila belum ada. Checkpoint terbaik berdasarkan Macro-F1 disimpan di `runs/preview2/efficientnet_b0_best.pth`, **bukan** langsung ke `submission/model.pth`, untuk mencegah perubahan checkpoint yang sudah diaudit.

### 4. Evaluasi submission aktif

```bash
python evaluate.py --submission ./submission --data ./my_val
```

Skrip evaluasi menampilkan Macro-F1, akurasi, F1 per kelas, serta confusion matrix. `submission/model.py` hanya membangun EfficientNet-B0 (`weights=None`) dan transform; `evaluate.py` yang memuat `submission/model.pth`.

> `train_baseline.py` adalah skrip starter-kit ResNet-18 dari Preview 1 dan tidak digunakan untuk mereproduksi submission Preview 2. Jalankan `train_preview2.py` untuk eksperimen aktif. Skrip starter-kit tersebut masih mengimpor arsitektur dari `submission/model.py`, sehingga tidak boleh dijalankan apa adanya setelah arsitektur submission diganti.

## Struktur Repositori

```text
studio2-klasifikasi-sampah_Kelompok10/
├── README.md
├── .gitignore
├── requirements.txt
├── evaluate.py
├── train_baseline.py          # kode referensi Preview 1
├── train_preview2.py          # reproduksi eksperimen EfficientNet-B0
└── submission/
    ├── model.py               # arsitektur + transform EfficientNet-B0
    └── model.pth              # checkpoint terbaik Preview 2
```

Dataset, split lokal, dan checkpoint eksperimen di luar `submission/` tidak disertakan dalam repository.
