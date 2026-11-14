# Project Context & Engineering Standards: Steam Aspect Sentiment Analysis (ACSA)

## 1. Project Mission & Identity
Penelitian Skripsi Teknik Informatika S1 (Universitas Widyatama):
Membangun sistem Aspect-Category Sentiment Analysis (ACSA) berbasis Transformer (BERT) untuk mengevaluasi ulasan gim indie platform Steam ke dalam 3 aspek independen (Gameplay, Story & UX, Technical & Performance) serta menghasilkan agregasi level produk berupa Game Scorecard (Aspect Satisfaction Index / ASI) pada 150 gim indie.

## 2. Core Architecture & Constraints
- **Paradigma:** Aspect-Category Sentiment Analysis (ACSA) multi-model.
- **Model:** 3 Model BERT (`bert-base-uncased`) terpisah secara fisik:
  1. `BERT_Gameplay` -> memprediksi sentimen gameplay `[0: none, 1: pos, 2: neg]`
  2. `BERT_Story` -> memprediksi sentimen story/UX `[0: none, 1: pos, 2: neg]`
  3. `BERT_Technical` -> memprediksi sentimen teknis/bugs `[0: none, 1: pos, 2: neg]`
- **Anti-Bloat & Raw Engine Rule:**
  - WAJIB menggunakan manual training loop murni PyTorch (`torch`, `torch.nn`, `torch.optim.AdamW`).
  - DILARANG menggunakan framework wrapper tingkat tinggi (NO HuggingFace Trainer, NO PyTorch Lightning, NO fastai, NO LangChain).
  - Loop training harus transparan: forward pass, loss computation, backward pass, gradient clipping (1.0), optimizer step, scheduler step.
- **Hardware & Memory Management:**
  - Target GPU laptop 4GB VRAM (GTX 1650).
  - Pelatihan dijalankan sekuensial (satu model selesai, baru lanjut model berikutnya).
  - Panggil `torch.cuda.empty_cache()` secara eksplisit di akhir setiap epoch evaluasi.
- **Loss Function:** Dynamic Class-Weighted CrossEntropyLoss untuk menangani ketimpangan kelas `none` yang dominan.
- **Checkpointing Metric:** Disimpan berdasarkan Macro-F1 pada KELAS AKTIF (`pos` dan `neg`), bukan akurasi global.
- **Baseline Pembanding:** 3 model Linear Support Vector Machine (`TfidfVectorizer` + `LinearSVC`) pada test set identik.

## 3. Data Contracts & Directory Layout
```text
NLP/
├── configs/
│   └── config.yaml              # Parameter, seeds, hyperparameters, target 150 app_ids
├── data/
│   ├── raw/                     # JSONL ulasan per game dari Steamworks API
│   ├── clean/                   # CSV ulasan bersih (bebas BBCode, mojibake, noise)
│   ├── annotation/              # 1.500 ulasan tersampel & dataset ground truth 3 kolom
│   └── splits/                  # split_indices.json (train 70%, val 15%, test 15%)
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── scraper.py           # Steamworks API cursor-based scraper + backoff
│   │   └── cleaner.py           # Pembersihan BBCode, HTML entities, filtering
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── sampler.py           # Stratified sampling 1.500 data
│   │   └── loader.py            # PyTorch Dataset & DataLoader per aspek
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bert_classifier.py   # BERT + Custom Head (GELU + Dropout 0.2)
│   │   └── baseline_svm.py      # TF-IDF + LinearSVC
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Manual PyTorch training loop + dynamic class weights
│   │   └── metrics.py           # Macro-F1, Active-F1, confusion matrix
│   └── inference/
│       ├── __init__.py
│       ├── predict.py           # Batch inference ke 50.000 ulasan unseen
│       └── scorecard.py         # Aspect Satisfaction Index (ASI) generator
├── tests/                       # Unit tests & smoke tests
├── scripts/                     # Helper standalone run scripts
├── requirements.txt
└── run.py                       # CLI entry point
```

## 4. Engineering & Execution Principles
1. **Verifikasi Mandiri (Smoke Test):** Setiap selesai mengimplementasikan modul, buat dan jalankan smoke test dengan dummy data untuk memverifikasi shape tensor, import, dan fungsi dasar.
2. **Deterministic:** Kunci random seed = 42 di `torch`, `numpy`, dan `random`.
3. **Defensive Programming:** Validasi skema DataFrame, format JSON, dan tipe data sebelum komputasi.
4. **No Placeholders / No TODO Stubs:** Semua fungsi harus terimplementasi penuh dengan penanganan error yang jelas.
