
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
| Preview 1 (14 Okt) | `b17b55c` | 0.9295 | Menunggu hasil |
| Preview 2 (21 Okt) | - | - | - |
| Final (25 Okt) | - | - | - |


### Catatan Eksperimen

#### Eksperimen 01 — Baseline ResNet-18

Model baseline dilatih menggunakan transfer learning dengan backbone ResNet-18 pretrained ImageNet.

**Konfigurasi training:**

- Backbone: ResNet-18 pretrained ImageNet
- Epoch: 8
- Batch size: 32
- Learning rate: 0.0003
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- Label smoothing: 0.1
- Dataset: 2.753 gambar
- Validasi lokal: 410 gambar

**Hasil evaluasi lokal:**

| Metrik | Hasil |
|--------|-------|
| Macro-F1 | 0.9295 |
| Akurasi | 0.9341 |
| Waktu evaluasi | 4.6 detik |
| Ukuran model | 42.71 MiB |

**F1-score per kelas:**

| Kelas | F1-score |
|-------|----------|
| botol_plastik | 0.9474 |
| gelas_plastik | 0.9494 |
| kertas_kardus | 0.8738 |
| saset_kemasan | 0.9353 |
| styrofoam | 0.9417 |

**Analisis:**

Model mencapai Macro-F1 lokal sebesar 0.9295.

Kelas kertas_kardus memiliki F1-score terendah, yaitu 0.8738. Dari 54 gambar validasi kelas tersebut, 45 berhasil diklasifikasikan dengan benar.

Kelas ini akan menjadi salah satu fokus optimasi pada eksperimen berikutnya.

**Audit submission:**

- Commit model: b17b55c
- Bobot GitHub identik dengan bobot lokal
- Evaluasi berhasil menggunakan evaluate.py resmi
- Exit code: 0

Hasil di atas berasal dari split validasi lokal, bukan hidden validation set milik tim pengajar.

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
