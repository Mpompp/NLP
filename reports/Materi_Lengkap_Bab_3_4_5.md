# DRAF MATERI SKRIPSI (BAB 3, BAB 4, DAN BAB 5)
## Aspect-Category Sentiment Analysis (ACSA) pada Ulasan Gim Indie Steam Menggunakan Multi-Model BERT Terpisah

**Penulis:** Arief Fadhillah  
**Institusi:** Program Studi Teknik Informatika, Universitas Widyatama  
**Tanggal Verifikasi Data:** 22 Oktober 2026  

---

# BAB III: METODOLOGI PENELITIAN

## 3.1 Alur Penelitian
Penelitian ini menggunakan pendekatan kuantitatif berbasis eksperimen *Natural Language Processing* (NLP) untuk menganalisis sentimen berbasis kategori aspek (*Aspect-Category Sentiment Analysis* / ACSA). Pipeline penelitian disusun secara sistematis tanpa kebocoran data (*zero data leakage*), mencakup lima tahapan utama:
1. **Pengumpulan Korpus Mentah (*Data Ingestion*):** Scraping ulasan pemain berbahasa Inggris dari Steamworks API untuk 150 judul gim indie terpilih.
2. **Prapemrosesan & Pembersihan Korpus (*Data Preprocessing*):** Sanitasi tag BBCode Steam, normalisasi karakter Unicode, eliminasi spam ASCII art, dan deduplikasi ulasan.
3. **Penyusunan Ground Truth & Stratified Splitting:** Pengambilan 1.500 sampel ulasan terstratifikasi, anotasi multi-aspek 3 label diskrit, serta partisi data *mutually disjoint* (Train 70%, Val 15%, Test 15%).
4. **Pelatihan Model & Baseline Benchmarking:** Pelatihan model *baseline* konvensional (TF-IDF + Linear SVM) dan fine-tuning 3 model BERT terpisah (*Gameplay*, *Story & Aesthetics*, *Technical*) dengan penanganan *class imbalance*.
5. **Evaluasi Komparatif & Penerapan Skala Penuh:** Pengujian *apple-to-apple* pada *unseen test set*, inferensi massal 60.820 ulasan, dan kalkulasi *Aspect Satisfaction Index* (ASI) untuk menghasilkan *Game Scorecard*.

---

## 3.2 Dataset dan Skema Anotasi

### 3.2.1 Kategori Aspek
Penelitian ini membatasi analisis pada 3 kategori aspek dominan dalam industri gim video independen:
1. **Gameplay:** Menilai mekanisme inti (*core mechanics*), sistem kontrol, responsivitas input, desain pertempuran (*combat*), variasi puzzle, kurva kesulitan, *gameplay loop*, dan *replayability*.
2. **Story & Aesthetics:** Menilai kualitas narasi, plot, penulisan dialog, penokohan karakter, *lore*, pengisian suara (*voice acting*), estetika gaya visual/seni grafis, tata musik (*soundtrack*), atmosfer, dan daya imersi emosional.
3. **Technical:** Menilai stabilitas performa sistem, konsistensi *frame rate* (FPS), *stuttering*, waktu tunggu pemuatan (*loading time*), kegagalan sistem (*crash/freeze*), *glitch/bug*, kerusakan berkas simpanan (*save file corrupt*), serta optimasi lintas konfigurasi perangkat keras PC.

### 3.2.2 Skema Pelabelan Ground Truth
Setiap ulasan diberikan label sentimen diskrit secara independen pada ketiga aspek dengan ruang nilai $\mathcal{Y} \in \{0, 1, 2\}$:
* **Label 0 (*None / Not Mentioned*):** Aspek tidak dibahas sama sekali dalam ulasan, atau teks ulasan tidak relevan/berbahasa asing.
* **Label 1 (*Positive*):** Ulasan mengekspresikan kepuasan, pujian, atau apresiasi eksplisit terhadap aspek tersebut.
* **Label 2 (*Negative*):** Ulasan mengekspresikan kekecewaan, kritik, keluhan, frustrasi, atau laporan malfungsi terhadap aspek tersebut.

### 3.2.3 Distribusi Ground Truth Teranotasi
Sebanyak 1.500 ulasan sampel dianotasi dengan distribusi frekuensi kelas sebagai berikut:

| Kategori Aspek | Label 0 (None) | Label 1 (Positif) | Label 2 (Negatif) | Total Sampel |
| :--- | :---: | :---: | :---: | :---: |
| **Gameplay** | 565 (37.7%) | 709 (47.3%) | 226 (15.0%) | 1.500 |
| **Story & Aesthetics** | 940 (62.7%) | 484 (32.3%) | 76 (5.0%) | 1.500 |
| **Technical** | 1.392 (92.8%) | 30 (2.0%) | 78 (5.2%) | 1.500 |

*Catatan Karakteristik Data:* Distribusi ini merefleksikan sifat riil ulasan gim Steam, di mana aspek gameplay menjadi fokus utama pemain, sedangkan aspek teknis umumnya bersifat *silent satisfaction* (hanya dibahas intensif jika gim mengalami masalah teknis parah atau optimasi luar biasa lancar).

---

## 3.3 Partisi Data dan Pencegahan Kebocoran (*Zero Data Leakage*)
Untuk menjamin integritas ilmiah, partisi dataset dikunci pada berkas `data/splits/split_indices.json` dengan rasio 70:15:15:
* **Training Set:** 1.050 ulasan (digunakan untuk estimasi parameter dan pembobotan loss).
* **Validation Set:** 225 ulasan (digunakan untuk *early stopping* dan pemilihan *checkpoint* terbaik berdasarkan Active Macro-F1).
* **Test Set:** 225 ulasan (diisolasi secara ketat dan hanya dibuka saat evaluasi akhir).

Ketiga himpunan ID ulasan divalidasi secara matematis memenuhi kondisi *pairwise disjoint*:
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset, \quad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset, \quad \mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$

---

## 3.4 Arsitektur Multi-Model BERT Terpisah

Berbeda dengan arsitektur *shared-backbone multi-head* yang rentan terhadap fenomena interferensi representasi antartugas (*gradient conflict* atau *negative transfer*) saat disparitas distribusi antar-aspek tinggi, penelitian ini menerapkan arsitektur **3 Model BERT Fisik Terpisah**:
1. `bert_gameplay` $\to$ Mengkhususkan bobot semantik pada domain mekanik permainan.
2. `bert_story` $\to$ Mengkhususkan bobot semantik pada ranah narasi dan estetika visual-audio.
3. `bert_technical` $\to$ Mengkhususkan bobot semantik pada istilah performa komputasi dan kestabilan sistem.

### 3.4.1 Spesifikasi Arsitektur Model
* **Model Fondasi:** `bert-base-uncased` (12 *transformer layers*, 768 *hidden dimensions*, 12 *attention heads*, 110 juta parameter).
* **Classification Head:**
  $$\mathbf{h}_{[\text{CLS}]} \in \mathbb{R}^{768}$$
  $$\mathbf{z} = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{h}_{[\text{CLS}]} + \mathbf{b}_1) + \mathbf{b}_2, \quad \mathbf{z} \in \mathbb{R}^3$$
  Di mana $\mathbf{W}_1 \in \mathbb{R}^{768 \times 768}$, $\mathbf{W}_2 \in \mathbb{R}^{3 \times 768}$, dengan *Dropout rate* $p = 0.3$.
* **Fungsi Objektif:** *Cross-Entropy Loss* terbobot dinamis berdasarkan frekuensi terbalik kelas (*inverse class frequency weighting*):
  $$w_c = \frac{N}{C \cdot N_c}$$
  Hal ini krusial untuk mencegah model bias memprediksi kelas mayoritas (Label 0).
* **Optimasi:** *AdamW Optimizer* ($\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$, *weight decay* $0.01$), *learning rate* $\eta = 2 \times 10^{-5}$, *linear warmup schedule* (10% langkah awal).
* **Manajemen Sumber Daya VRAM:** Setiap model dilatih secara sekuensial. Setelah pelatihan satu model tuntas dan *checkpoint* terbaik tersimpan di disk, memori VRAM GPU dibersihkan secara eksplisit via `gc.collect()` dan `torch.cuda.empty_cache()` untuk mencegah *Out-of-Memory* (OOM) pada VRAM 4 GB (NVIDIA GTX 1650).

---

## 3.5 Metrik Evaluasi: Macro-F1 dan Active Macro-F1

Evaluasi performa klasifikasi diukur menggunakan **Macro-F1** standar dan **Active Macro-F1**:

1. **Macro-Average F1:**
   $$\text{Macro-F1} = \frac{F1_{\text{None}} + F1_{\text{Positif}} + F1_{\text{Negatif}}}{3}$$

2. **Active Macro-Average F1 (Metrik Kunci Skripsi):**
   Pada sistem ACSA, performa model dalam mendeteksi dan membedakan polaritas sentimen aktif (apakah pemain memuji atau mengeluh) jauh lebih krusial daripada sekadar memprediksi ketiadaan aspek (kelas 0). Active Macro-F1 hanya merata-ratakan skor F1 pada kelas aktif:
   $$\text{Active Macro-F1} = \frac{F1_{\text{Positif}} + F1_{\text{Negatif}}}{2}$$

---

# BAB IV: HASIL DAN PEMBAHASAN

## 4.1 Hasil Evaluasi Komparatif (Linear SVM vs Multi-Model BERT)

Pengujian dilakukan secara *apple-to-apple* terhadap 225 ulasan pada *Test Set* identik yang tidak pernah dilihat oleh model selama proses pelatihan. Model pembanding adalah *Linear Support Vector Machine* (SVM) berbasis pembobotan fitur TF-IDF unigram dan bigram (rentang fitur maksimal 5.000 kata).

### Tabel 4.1: Perbandingan Metrik Evaluasi pada Test Set Identik (N = 225)
| Kategori Aspek | Metrik Evaluasi | Baseline (Linear SVM) | Proposed (BERT Fine-Tuned) | Selisih Peningkatan ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **GAMEPLAY** | Macro-F1 | 0.6096 | **0.7368** | **+0.1273 (+12.7%)** |
| | **Active Macro-F1** | 0.5810 | **0.7632** | **+0.1821 (+18.2%)** |
| **STORY & AESTHETICS** | Macro-F1 | 0.5218 | **0.5817** | **+0.0599 (+6.0%)** |
| | **Active Macro-F1** | 0.3772 | **0.4864** | **+0.1092 (+10.9%)** |
| **TECHNICAL** | Macro-F1 | 0.4035 | **0.5471** | **+0.1436 (+14.4%)** |
| | **Active Macro-F1** | 0.1250 | **0.3487** | **+0.2237 (+22.4%)** |

*Rata-rata Peningkatan Active Macro-F1 di Seluruh Aspek:* **+17.17%**.

---

## 4.2 Analisis Rinci Performa per Kelas dan Matriks Konfusi

### 4.2.1 Aspek Gameplay
Pada aspek gameplay, BERT mencatatkan performa terbaik dengan Active Macro-F1 mencapai **0.7632**.

* **Performa per Kelas BERT Gameplay:**
  * Kelas 0 (None): Precision = 0.7647, Recall = 0.6190, F1-Score = 0.6842
  * Kelas 1 (Positif): Precision = 0.6396, Recall = 0.7978, F1-Score = 0.7100
  * Kelas 2 (Negatif): Precision = 0.8696, Recall = 0.7692, F1-Score = 0.8163
* **Matriks Konfusi (Aktual vs Prediksi):**
  $$\begin{pmatrix} 52 & 29 & 3 \\ 15 & 71 & 3 \\ 1 & 11 & 40 \end{pmatrix}$$
  *(Baris: None, Positif, Negatif; Kolom: Prediksi None, Positif, Negatif)*

*Pembahasan:* Pada kelas negatif (Label 2), model BERT mencapai precision 86.9% dan recall 76.9% (F1 = 0.8163). Sebaliknya, Linear SVM gagal menangani variasi kritik gameplay dan hanya meraih recall 36.5% (SVM hanya benar menebak 19 dari 52 ulasan negatif gameplay). Mekanisme *self-attention* pada BERT terbukti mampu mengenali struktur kalimat keluhan yang halus dan bernuansa sarkasme.

### 4.2.2 Aspek Story & Aesthetics
* **Performa per Kelas BERT Story:**
  * Kelas 0 (None): Precision = 0.8065, Recall = 0.7407, F1-Score = 0.7722
  * Kelas 1 (Positif): Precision = 0.6364, Recall = 0.7467, F1-Score = 0.6871
  * Kelas 2 (Negatif): Precision = 0.3077, Recall = 0.2667, F1-Score = 0.2857
* **Matriks Konfusi:**
  $$\begin{pmatrix} 100 & 27 & 8 \\ 18 & 56 & 1 \\ 6 & 5 & 4 \end{pmatrix}$$

*Pembahasan:* Sentimen negatif pada aspek cerita relatif langka pada gim indie (hanya ada 15 sampel negatif pada test set). Meskipun SVM memiliki precision tinggi semu pada kelas negatif (1.0), recall-nya hanya 6.7% (hanya 1 ulasan negatif cerita yang terdeteksi dari 15 kasus). BERT berhasil meningkatkan recall kelas negatif menjadi 26.7%, sehingga mengangkat Active Macro-F1 dari 0.3772 menjadi 0.4864 (+10.9%).

### 4.2.3 Aspek Technical
Aspek teknis merupakan kelas dengan ketimpangan data paling ekstrem (*imbalanced dataset*), di mana 92.8% data ground truth bernilai 0 (tidak dibahas).

* **Performa per Kelas BERT Technical:**
  * Kelas 0 (None): Precision = 0.9507, Recall = 0.9369, F1-Score = 0.9438
  * Kelas 1 (Positif): Precision = 0.5000, Recall = 0.2000, F1-Score = 0.2857
  * Kelas 2 (Negatif): Precision = 0.3500, Recall = 0.5000, F1-Score = 0.4118
* **Matriks Konfusi:**
  $$\begin{pmatrix} 193 & 1 & 12 \\ 3 & 1 & 1 \\ 7 & 0 & 7 \end{pmatrix}$$

*Pembahasan Kritis:* Pada baseline Linear SVM, model mengalami kegagalan deteksi total pada kelas positif teknis ($F1 = 0.0$, tidak ada satu pun pujian optimasi teknis yang berhasil diprediksi oleh SVM). Selain itu, SVM hanya mampu mendeteksi 2 dari 14 ulasan keluhan teknis ($Recall = 14.3\%$). Melalui *dynamic class weighting*, BERT berhasil mendeteksi 7 dari 14 kasus keluhan teknis ($Recall = 50.0\%$), melipatgandakan skor Active Macro-F1 dari 0.1250 menjadi **0.3487 (+22.37%)**.

---

## 4.3 Penerapan Skala Penuh: Game Scorecard 150 Gim Indie Steam

Model BERT yang telah divalidasi kemudian diuji pada skala penuh (*bulk unseen inference*) terhadap **60.820 ulasan bersih** dari 150 gim indie. Tingkat kepuasan pemain diagregasikan ke dalam **Aspect Satisfaction Index (ASI)**:
$$\text{ASI}_A = \frac{N_{\text{pos}, A}}{N_{\text{pos}, A} + N_{\text{neg}, A}} \times 100\%$$
Di mana gim dengan total ulasan aktif ($N_{\text{pos}} + N_{\text{neg}}$) kurang dari ambang batas $\tau = 5$ diberi label *Data Kurang*.

### Tabel 4.2: Sampel Representatif Game Scorecard (Top-Tier vs Technical Alert)
| App ID | Judul Gim Indie | Total Ulasan Unseen | ASI Gameplay | ASI Story | ASI Technical | Diagnostic Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **2915460** | OneShot: World Machine Ed. | 402 | 90.1% | 98.0% | 10.0% | ALERT: Stabilitas Teknis |
| **253230** | A Hat in Time | 394 | 89.6% | 96.5% | 0.0% | ALERT: Stabilitas Teknis |
| **467380** | Khimera: Destroy All Monster | 449 | 89.6% | 97.3% | 10.5% | ALERT: Stabilitas Teknis |
| **270880** | American Truck Simulator | 339 | 89.3% | 98.5% | 11.1% | ALERT: Stabilitas Teknis |
| **219890** | Antichamber | 419 | 87.2% | 95.8% | 18.2% | ALERT: Stabilitas Teknis |
| **242760** | The Forest | 345 | 85.9% | 95.3% | 10.0% | ALERT: Stabilitas Teknis |
| **526870** | Satisfactory | 373 | 85.3% | 79.7% | 0.0% | ALERT: Stabilitas Teknis |
| **294100** | RimWorld | 395 | 83.4% | 88.9% | 4.2% | ALERT: Stabilitas Teknis |
| **212680** | FTL: Faster Than Light | 399 | 81.8% | 92.3% | 5.0% | ALERT: Stabilitas Teknis |
| **219150** | Hotline Miami | 376 | 80.5% | 92.6% | 0.0% | ALERT: Stabilitas Teknis |
| **220200** | Kerbal Space Program | 352 | 95.6% | 94.7% | N/A | Data Teknis Kurang |
| **663380** | Tales from Candlekeep | 465 | 48.0% | 71.0% | 3.0% | ALERT: Gameplay & Teknis |
| **371520** | Bounty Train | 439 | 48.0% | 66.7% | 0.0% | ALERT: Gameplay Kritis |

### 4.3.1 Temuan Penting Analisis Industri Gim Indie:
1. **Divergensi Tajam Sentimen Keseluruhan vs Aspek Teknis:** Pada skor ulasan agregat Steam konvensional (*Overall Review Score*), gim legendaris seperti *A Hat in Time*, *LIMBO*, dan *Hotline Miami* memiliki predikat *"Overwhelmingly Positive"* (>90% rekomendasi). Namun melalui dekomposisi ACSA, sistem mendeteksi bahwa indeks teknis gim-gim tersebut anjlok di bawah 10% (didominasi keluhan bug, crash pada OS baru, atau kendala resolusi layar). Hal ini membuktikan bahwa skor biner Steam mengaburkan masalah teknis yang sebenarnya dikeluhkan pemain.
2. **Kekuatan Aspek Story & Art pada Gim Indie:** Rata-rata skor $\text{ASI}_{\text{Story}}$ di seluruh korpus mencapai **86.4%**, membuktikan bahwa penulisan narasi, arahan seni visual (*art style*), dan musik orisinal merupakan keunggulan kompetitif utama pengembang gim independen dibandingkan gim AAA.

---

# BAB V: KESIMPULAN DAN SARAN

## 5.1 Kesimpulan
Berdasarkan hasil perancangan, implementasi, dan pengujian sistem Aspect-Category Sentiment Analysis pada ulasan gim indie Steam, diperoleh kesimpulan sebagai berikut:
1. **Keunggulan Arsitektur Transformer:** Penerapan arsitektur 3 model BERT terpisah berhasil membuktikan superioritas mutlak dibandingkan *baseline* Linear SVM di seluruh aspek evaluasi, dengan rata-rata lonjakan **Active Macro-F1 sebesar +17.17%** (+18.22% pada Gameplay, +10.92% pada Story, dan +22.37% pada Technical).
2. **Efektivitas Mitigasi Ketimpangan Data:** Integrasi pembobotan dinamis frekuensi terbalik (*dynamic class weighting*) pada fungsi *loss* berhasil menyelamatkan model dari perangkap prediksi mayoritas (*majority class bias*), di mana pada aspek teknis yang ekstrem imbalanced (92.8% label none), model BERT mampu mencapai *recall* sentimen negatif sebesar 50.0% dibandingkan SVM yang hanya 14.3%.
3. **Validitas Manajerial Game Scorecard:** Ekstraksi sentimen multi-aspek pada 60.820 ulasan riil membuktikan bahwa sistem mampu mengekspos anomali tersembunyi (*hidden technical bottlenecks*) pada gim-gim bertaraf *Overwhelmingly Positive*, memberikan wawasan diagnostik yang dapat ditindaklanjuti secara langsung (*actionable insights*) bagi pengembang gim indie untuk memprioritaskan pembaruan (*patching*).

## 5.2 Saran
1. **Eksplorasi Arsitektur Adapter / LoRA:** Pada penelitian lanjutan dengan keterbatasan sumber daya komputasi, penggunaan *Low-Rank Adaptation* (LoRA) atau adapter multi-aspek dapat dieksplorasi untuk memangkas ukuran penyimpanan *checkpoint* model (dari ~438 MB per aspek menjadi <20 MB) tanpa mengorbankan isolasi representasi semantik.
2. **Pengembangan Deteksi Aspek Berbasis Span (*Aspect Extraction*):** Penelitian ini berfokus pada kategori aspek (*Aspect Category*). Pengembangan ke depan dapat menggabungkan *End-to-End Aspect-Based Sentiment Analysis* (ABSA) untuk mendeteksi kata kunci entitas spesifik (misal: "boss fight", "FPS cap", "inventory size") secara otomatis dari teks ulasan.
