
# Studio 2 — Klasifikasi Sampah · Kelompok 10

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1 | Cikal Galih Nur Arifin | 123140109 | Model Training & Evaluation |
| 2 | Muhamad Rafi Ilham | 123140173 | Development & Integration |

## Ringkasan Pendekatan

Proyek ini bertujuan membangun model CNN untuk mengklasifikasikan gambar sampah ke dalam lima kelas:

1. `botol_plastik`
2. `gelas_plastik`
3. `kertas_kardus`
4. `saset_kemasan`
5. `styrofoam`

Pendekatan awal yang direncanakan menggunakan transfer learning dengan backbone ResNet-18 pretrained ImageNet.

- **Arsitektur / backbone:** ResNet-18 pretrained ImageNet dengan output lima kelas.
- **Strategi pelatihan:** Fine-tuning menggunakan AdamW, learning rate 0.0003, weight decay 0.0001, CosineAnnealingLR, dan label smoothing 0.1.
- **Epoch awal:** 8.
- **Batch size:** 32.
- **Augmentasi:** RandomResizedCrop, RandomHorizontalFlip, dan ColorJitter.
- **Data tambahan:** Tidak ada, menggunakan dataset resmi dosen.
- **Ukuran input:** 224 × 224 piksel.
- **Preprocessing validasi:** Resize(256), CenterCrop(224), ToTensor, dan normalisasi ImageNet.
- **Pembagian dataset:** Sekitar 85% training dan 15% validasi lokal.
- **Metrik utama:** Macro-F1.

Konfigurasi ini merupakan rencana baseline awal dan akan diperbarui sesuai hasil eksperimen.

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | - | Belum dievaluasi | - |
| Preview 2 (21 Okt) | - | Belum dievaluasi | - |
| Final (25 Okt) | - | Belum dievaluasi | - |

### Catatan Eksperimen

Eksperimen pertama direncanakan menggunakan ResNet-18 pretrained ImageNet sebagai baseline.

Hasil training, evaluasi Macro-F1, analisis confusion matrix, dan eksperimen optimasi akan didokumentasikan setelah model dijalankan.

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

Download dataset resmi melalui:

https://drive.google.com/drive/folders/1_XcZoDFRznHB7fI603VHzdFxreg3ma60?usp=sharing

Ekstrak dataset sehingga folder `train/` berada di root proyek dan berisi lima subfolder kelas.

Dataset tidak disertakan di repository GitHub sesuai aturan lisensi dan etika penggunaan dataset.

### 4. Training Model

Jalankan baseline menggunakan:

```bash
python train_baseline.py --data ./train --epochs 8
```

Training akan membuat split `my_train/` dan `my_val/`, kemudian menghasilkan `submission/model.pth`.

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

File `model.pth` akan tersedia setelah training berhasil dilakukan.

Folder dataset dan split validasi disimpan di lingkungan training dan tidak diunggah ke GitHub.

## Referensi

- [Portal Deep Learning ITERA](https://informatika-itera.github.io/deep-learning-IF25-40401/)
- [Studio Proyek 2](https://informatika-itera.github.io/deep-learning-IF25-40401/studio-2/)
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
