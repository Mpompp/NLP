# BAB I
# PENDAHULUAN

## 1.1 Latar Belakang Masalah

Industri permainan video (*video games*) kini tumbuh menjadi pilar utama ekonomi digital global. Nilai perputaran modal sektor ini melampaui valuasi industri perfilman dan musik rekaman. Skalanya bertambah sangat masif. Laporan pasar global *Newzoo* memproyeksikan pendapatan pasar gim dunia pada tahun 2024 mencapai US$187,7 miliar [1]. Segmen *Personal Computer* (PC) memegang peranan strategis dalam monetisasi perangkat lunak dan keterlibatan komunitas pemain [1]. Di dalam ekosistem distribusi PC, platform Steam milik Valve Corporation mempertahankan hegemoni sebagai lokapasar (*marketplace*) digital terbesar di dunia yang melayani puluhan juta pengguna aktif harian [2]. Komunikasi pascapembelian pada ekosistem Steam difasilitasi melalui fitur ulasan pengguna (*user reviews*). Ulasan tersebut berfungsi sebagai instrumen reputasi komersial berbasis *electronic Word-of-Mouth* (eWOM) [2], [3]. Ulasan publik mengendalikan algoritma kurasi etalase toko secara langsung (*Steam Store Discovery Queue*) [2]. Dampaknya sangat nyata. Data umpan balik ini mencerminkan kepuasan konsumen sekaligus menentukan kelangsungan sirkulasi komersial produk perangkat lunak di pasar internasional.

Ulasan pemain memiliki urgensi strategis yang jauh lebih krusial bagi studio pengembang independen (*indie game developers*) dibandingkan korporasi raksasa (*triple-A* / AAA) [3]. Korporasi besar umumnya didukung penerbit multinasional, anggaran promosi bernilai jutaan dolar, serta divisi penjamin mutu (*Quality Assurance* / QA) internal yang mapan. Sebaliknya, studio gim indie beroperasi dengan tim ramping beranggotakan dua hingga lima personel dengan modal kerja yang sangat terbatas. Ketiadaan divisi QA internal memaksa pengembang indie mengandalkan ulasan pengguna Steam sebagai instrumen audit fungsional utama [2], [3]. Melalui ulasan publik, pengembang mendeteksi galat kode (*bugs*), kegagalan eksekusi fatal (*crashes*), penurunan stabilitas performa komputasi (*stuttering*), disonansi mekanisme kontrol (*gameplay mechanics*), hingga respon narasi visual [2], [3]. Ketergantungan tersebut menghadapi tantangan berat akibat hiper-saturasi pasar (*hyper-saturated market*). Tekanan kompetisinya berlangsung sangat ketat. Basis data *SteamDB* mencatat lebih dari 19.000 judul gim baru dirilis di platform tersebut dalam rentang satu tahun saja [4]. Laporan industri *PC Gamer* membuktikan bahwa hampir separuh produk rilis hanya menghimpun kurang dari 10 ulasan pemain sehingga gagal memperoleh visibilitas etalase toko [5]. Situasi kompetitif ini memusatkan pertaruhan hidup produk pada jendela kritis 72 jam pertama pasca-peluncuran (*launch window*) [5]. Keterlambatan studio indie dalam merespons keluhan teknis selama fase awal ini dapat merusak reputasi toko secara permanen [2]. Dampak finansialnya sangat fatal.

### 1.1.2 Kondisi Normatif dan Harapan Sistemik (*Das Sollen*)
Standar rekayasa perangkat lunak ISO/IEC 25010 menetapkan bahwa mekanisme umpan balik pengguna (*user feedback*) harus berfungsi sebagai instrumen audit kualitas fungsional [18], [20]. Informasi tersebut semestinya bersifat transparan, terstruktur, dan dapat ditindaklanjuti secara teknis (*actionable insights*). Kondisi ini sangat ideal. Dalam kondisi normatif (*das sollen*), sistem evaluasi pada lokapasar digital membedah persepsi konsumen ke dalam dimensi mutu yang terisolasi secara komprehensif:
1. Memisahkan evaluasi terhadap ergonomi kontrol, mekanik interaktif, dan daya tarik permainan (*gameplay mechanics*).
2. Mengisolasi penilaian terhadap kedalaman narasi, dialog, penokohan, estetika visual, serta komposisi musik latar (*story & aesthetics*).
3. Mengidentifikasi laporan anomali teknis, galat kode (*bugs*), penurunan laju bingkai (*FPS drop*), kegagalan memori (*stuttering*), hingga kerusakan fatal sistem (*crash to desktop*).

Melalui pemetaan multidimensi tersebut, pengembang perangkat lunak dapat menapis tiket insiden (*issue triage*) secara instan [46]. Alokasi sumber daya perbaikan sistem dapat ditentukan secara proporsional dan terarah. Peta jalan pembaruan produk (*product roadmap*) juga dapat disusun berdasarkan bukti empiris yang objektif. Proses evaluasi menjadi terstruktur. Di samping itu, calon konsumen memperoleh transparansi mengenai kondisi riil setiap aspek produk sebelum melakukan transaksi pembelian [19].

### 1.1.3 Realitas Faktual dan Fenomena Lapangan (*Das Sein*)
Realitas konkret di lapangan (*das sein*) menunjukkan ketimpangan tajam terhadap kondisi ideal tersebut. Sistem evaluasi resmi platform Steam hanya menyediakan skema penilaian biner, yaitu ulasan direkomendasikan (*Recommended*) atau tidak direkomendasikan (*Not Recommended*) [2]. Skema biner global ini mereduksi dan mengaburkan rincian diagnostik mutu perangkat lunak yang sesungguhnya [12], [14]. Informasi teknis menjadi bias. Bukti empiris menunjukkan bahwa gim dengan predikat positif global (*Overwhelmingly Positive*) sering kali masih menyimpan cacat teknis tersembunyi yang hanya muncul pada variasi perangkat keras tertentu. Sebaliknya, akumulasi ulasan negatif (*Mostly Negative*) kerap kali timbul bukan karena kegagalan fungsional program. Penilaian buruk sering kali dipicu oleh ketidaksukaan subjektif pemain terhadap kurva kesulitan mekanik (*difficulty curve*). Agregasi biner menyatukan seluruh keluhan dan apresiasi ke dalam satu kutub nilai tunggal.

Kondisi faktual tersebut diperberat oleh karakteristik teks ulasan pemain sebagai *User-Generated Content* (UGC) informal [18]. Dokumen ulasan komunitas gim menggunakan struktur kalimat nirbaku serta sarat bahasa gaul (*gamer slang*) [3], [30]. Teks ulasan memuat ungkapan sarkasme, metafora, singkatan kata, dan istilah teknis khusus (*glitch*, *stuttering*, *hitbox*, *softlock*). Pada saat ribuan ulasan membanjiri etalase pasca-rilis, pemilahan manual (*manual inspection*) mustahil dilakukan oleh tim pengembang yang ramping. Laporan kerusakan teknis kritis berisiko terabaikan. Akibatnya, reputasi toko jatuh. Angka penjualan produk merosot secara permanen.

### 1.1.4 Identifikasi Kesenjangan (*Research Gap*) dan Tinjauan Debat Ilmiah
Kesenjangan antara harapan tata kelola umpan balik (*das sollen*) dan data ulasan biner yang berderau (*das sein*) memicu perdebatan metodologis yang luas. Berbagai paradigma pemodelan telah diuji oleh peneliti terdahulu guna menjembatani celah tersebut:

1. **Perdebatan Pendekatan Biner Pembelajaran Mesin Konvensional:**  
   Penelitian generasi awal membatasi cakupan pada klasifikasi sentimen biner positif versus negatif [12], [14], [30]. Model yang digunakan bertumpu pada algoritma *Support Vector Machine* (SVM), *Random Forest*, atau *Naive Bayes* dengan pembobotan leksikal TF-IDF. Pendekatan ini memiliki kelemahan mendasar. Representasi n-gram mengabaikan konteks semantik dinamis dan gagal menangkap pembalikan makna akibat negasi terselubung [8], [25]. Selain itu, label sentimen biner tidak menyediakan informasi fungsional yang dapat ditindaklanjuti dalam rekayasa perangkat lunak [14].

2. **Perdebatan Pendekatan Multikelas Tunggal (*Single-Label Multiclass*):**  
   Peneliti lain mengelompokkan ulasan ke dalam kategori fungsional menggunakan fungsi aktivasi *Softmax* [17], [31]. Kelemahan teoretis pendekatan ini terletak pada sifat matematis *Softmax* yang membatasi total distribusi probabilitas bernilai satu (*zero-sum constraint*) [6], [48]. Sifat tersebut memaksa setiap dokumen diasosiasikan secara eksklusif ke dalam satu label kelas saja (*mutually-exclusive*). Padahal, ulasan pemain kerap kali memuat multi-topik dan multi-sentimen secara simultan. Sebagai contoh, kalimat ulasan dapat memuji keunggulan mekanik pertempuran sekaligus mengeluhkan galat fatal sistem (*crash to desktop*). Prediksi kelas tunggal memicu benturan klasifikasi (*classification conflict*). Pemilihan satu aspek secara sepihak mengeliminasi sinyal laporan cacat sistem kritis dari proses penapisan tiket insiden.

3. **Perdebatan Arsitektur Tulang Punggung Bersama (*Shared-Backbone*):**  
   Penerapan model *Transformer* seperti BERT melahirkan pendekatan *Aspect-Category Sentiment Analysis* (ACSA) untuk mengevaluasi sentimen pada beberapa kategori sekaligus [8], [41]. Meskipun demikian, arsitektur dengan tulang punggung representasi bersama (*single shared-backbone*) rentan menimbulkan interferensi antartugas (*negative transfer* atau *gradient conflict*) [48]. Ruang representasi estetika narasi (*story*) memiliki pola semantik yang sangat berbeda dari kestabilan komputasi perangkat keras (*technical*). Karakteristik data ulasan juga menunjukkan ketimpangan frekuensi kemunculan kelas yang ekstrem (*extreme class imbalance*). Laporan keluhan teknis pada gim indie frekuensinya jauh lebih sedikit daripada ulasan mekanik permainan. Kondisi tersebut berpotensi menyebabkan gradien aspek mayoritas (*gameplay*) mendominasi pembaruan bobot internal model [37], [38]. Model kesulitan mendeteksi kelas minoritas teknis.

### 1.1.5 Celah Spesifik dan Pandangan Pakar (*The Deepest Void*)
Berdasarkan tinjauan perdebatan ilmiah tersebut, celah penelitian utama (*the deepest void*) yang perlu diuji secara empiris mencakup:
1. Pengujian hipotesis efektivitas arsitektur pemodelan modular terpisah dalam memitigasi konflik gradien (*gradient conflict*) dan transfer negatif antartugas pada korpus ulasan gim indie berkategori aspek majemuk.
2. Penerapan rekayasa siklus pelatihan manual (*manual training loop*) PyTorch yang terukur dan deterministik. Implementasi ini memadukan pemotongan gradien (*gradient clipping*), penjadwal laju belajar adaptif AdamW, serta pembobotan fungsi kerugian dinamis (*dynamic class-weighted loss*) untuk mengatasi ketimpangan data tanpa lapisan pembungkus tertutup (*black-box wrapper*).
3. Evaluasi komparatif yang ketat (*apple-to-apple benchmark*) terhadap model dasar konvensional (Linear SVM dengan TF-IDF n-gram) pada himpunan data uji identik bebas kebocoran data (*zero data leakage*). Celah ini mencakup integrasi hilir prediksi model ke dalam metrik **Aspect Satisfaction Index (ASI)** dan instrumen **Game Scorecard** pada korpus berskala besar.

Literatur komputasi bahasa alami [8], [39], [41] mendiskusikan bahwa modularitas representasi parameter membuka peluang mitigasi konflik optimasi antar-tugas yang divergen. Pendekatan ini diposisikan sebagai hipotesis kerja komparatif. Penelitian ini menguji apakah pemisahan fisik ruang representasi bobot per aspek mampu mengungguli arsitektur dasar konvensional pada domain teks ulasan yang sarat ketimpangan kelas.

### 1.1.6 Urgensi Penelitian
Urgensi penelitian ini bertumpu pada kontribusi teoretis komputasi dan kebutuhan praktis industri:
1. **Urgensi Teoretis:** Penelitian ini menguji secara empiris hipotesis keunggulan arsitektur multi-model spesialis terhadap model konvensional pada domain teks ulasan informal. Pengujian difokuskan pada evaluasi metrik *Macro-F1* dan *Active Macro-F1* di tengah kondisi ketimpangan kelas yang tajam.
2. **Urgensi Praktis:** Ketiadaan sistem penapisan otomatis berisiko mengisolasi pengembang indie dari informasi mutu kritis selama jendela 72 jam pasca-peluncuran. Laporan galat fatal yang terkubur di bawah ribuan teks ulasan informal dapat merusak reputasi toko serta menimbulkan kerugian komersial permanen.

Menjawab tantangan tersebut, penelitian ini menginvestigasi arsitektur **Multi-Model BERT Terpisah** menggunakan *manual training loop* PyTorch. Tiga model spesialis dilatih mandiri untuk mengklasifikasikan aspek *Gameplay*, *Story & Aesthetics*, dan *Technical*. Kinerja model diuji secara komparatif terhadap *baseline* Linear SVM. Selanjutnya, model terpilih diimplementasikan pada inferensi massal korpus 150 judul gim indie Steam guna menghasilkan instrumen diagnostik *Game Scorecard* berbasis formulasi *Aspect Satisfaction Index* (ASI).

Rangkaian penyelidikan ilmiah ini dilaksanakan di bawah judul resmi: **"KLASIFIKASI MULTI-KATEGORI ULASAN PEMAIN GIM INDIE DI STEAM MENGGUNAKAN MODEL BERT BERBASIS NATURAL LANGUAGE PROCESSING"**.

---

## 1.2 Rumusan Masalah
Rumusan masalah yang ditetapkan dalam penelitian ini adalah sebagai berikut:
1. Bagaimana merancang taksonomi tiga kategori aspek ulasan gim indie (*Gameplay*, *Story & Aesthetics*, *Technical*) dengan skema sentimen tri-kondisi {None, Positif, Negatif}? Bagaimana menerapkan siklus pelatihan manual (*manual training loop*) PyTorch berbobot kelas dinamis (*class-weighted loss*) untuk menangani ketimpangan distribusi data pada arsitektur Multi-Model BERT terpisah?
2. Bagaimana efektivitas komparatif arsitektur Multi-Model BERT terhadap model pembanding konvensional (Linear SVM berbasis TF-IDF) pada himpunan data uji identik (*zero data leakage*) berdasarkan evaluasi metrik *Macro-F1* dan *Active Macro-F1*?
3. Bagaimana menerapkan luaran model klasifikasi pada skala inferensi korpus 150 judul gim indie Steam guna menghasilkan instrumen *Game Scorecard* berbasis formulasi *Aspect Satisfaction Index* (ASI) yang memetakan disparitas rating global toko terhadap kepuasan aspek riil?

---

## 1.3 Tujuan Penelitian
Tujuan yang hendak dicapai dalam penelitian ini meliputi:
1. Mengembangkan taksonomi serta dataset *ground truth* tiga kategori aspek ulasan berlabel tri-kondisi. Tujuan ini mencakup implementasi siklus pelatihan manual PyTorch untuk melatih tiga model spesialis (`bert_gameplay`, `bert_story`, `bert_technical`) dengan stabilisasi gradien dan pembobotan *loss* adaptif.
2. Memvalidasi performa klasifikasi sentimen arsitektur Multi-Model BERT terhadap model *baseline* Linear SVM secara setara (*apple-to-apple*) pada data uji independen (*zero data leakage*) menggunakan metrik *Macro-F1*, *Active Macro-F1*, dan matriks konfusi.
3. Melakukan inferensi berskala besar terhadap korpus ulasan belum berlabel dari 150 judul gim indie Steam. Luaran inferensi digunakan untuk menghitung nilai *Aspect Satisfaction Index* (ASI) serta menyajikan instrumen diagnostik *Game Scorecard*.

---

## 1.4 Batasan Masalah
Batasan masalah ditetapkan agar ruang lingkup penelitian tetap terarah dan terukur:
1. **Sumber Data:** Data ulasan diakuisisi secara terprogram melalui antarmuka *Valve Steamworks API* (*ISteamReviews*). Teks dibatasi pada ulasan berbahasa Inggris.
2. **Kriteria Korpus Gim:** Objek penelitian mencakup 150 judul gim komersial berlabel resmi "Indie" pada platform Steam. Setiap judul memiliki minimal 500 ulasan. Gim daring masif (*Massively Multiplayer Online* / MMO) dieksklusikan guna mencegah bias akibat latensi peladen maupun kecurangan pemain.
3. **Kategori Aspek:** Analisis difokuskan pada tiga kategori fungsional: *Gameplay*, *Story & Aesthetics*, dan *Technical*. Setiap aspek memiliki tiga kelas polaritas diskrit: Label 0 (*None*), Label 1 (*Positif*), dan Label 2 (*Negatif*).
4. **Dataset Ground Truth:** Korpus berlabel mencakup 1.500 sampel ulasan unik yang ditarik melalui teknik *stratified random sampling*. Data dipartisi secara deterministik menjadi 70% data latih (1.050 ulasan), 15% data validasi (225 ulasan), dan 15% data uji (225 ulasan). Pembagian data menjamin isolasi mutlak (*zero data leakage*).
5. **Arsitektur Model:** Model utama bertumpu pada fondasi `bert-base-uncased` (12 *layers*, 768 *hidden units*, 110 juta parameter). Pelatihan model dieksekusi menggunakan siklus manual PyTorch tanpa pustaka automasi *wrapper* pihak ketiga.
6. **Model Pembanding:** Pembanding performa menggunakan algoritma *Linear Support Vector Machine* (Linear SVM). Model dipadukan dengan pembobotan TF-IDF unigram dan bigram berkapasitas 5.000 fitur leksikal teratas.
7. **Lingkungan Komputasi:** Seluruh eksperimen komputasi dijalankan pada akselerator grafis tunggal NVIDIA GeForce GTX 1650 (4GB GDDR6 VRAM). Sistem didukung oleh memori utama 16GB DDR4 RAM, CUDA versi 12.1, serta kerangka kerja PyTorch versi 2.x.
8. **Eksklusi Ruang Lingkup:** Studi ini tidak mencakup peramalan volume penjualan komersial, audit sistem monetisasi, maupun perancangan mesin sistem rekomendasi.

---

## 1.5 Manfaat Penelitian
Penelitian ini memberikan kontribusi teoretis bagi keilmuan informatika dan nilai guna praktis bagi industri perangkat lunak gim:

### 1.5.1 Manfaat Teoretis (Akademik)
1. **Pengembangan Metodologi NLP Korpus Informal:** Menyajikan bukti empiris pengujian modularitas arsitektur model bahasa dalam mereduksi transfer negatif (*negative transfer*) dan konflik optimasi antartugas. Temuan ini memperkaya kajian pemrosesan teks komunitas daring yang sarat sarkasme dan leksikon slang.
2. **Mitigasi Ketimpangan Data Spesifik:** Memberikan kontribusi metodologis mengenai stabilisasi pelatihan pada distribusi data timpang ekstrem (*extreme class imbalance*). Strategi ini mengintegrasikan fungsi kerugian berbobot dinamis serta evaluasi metrik *Active Macro-F1*.
3. **Validitas Komparasi Empiris Model:** Menyajikan tolok ukur performa yang setara (*apple-to-apple benchmark*) antara metode pembelajaran mesin konvensional dan arsitektur *deep learning* kontekstual. Pengujian dilakukan pada partisi data uji identik bergaransi bebas kebocoran data (*zero data leakage*).

### 1.5.2 Manfaat Praktis (Industri Perangkat Lunak Gim)
1. **Automasi Penapisan Tiket Insiden:** Membantu studio indie memilah laporan galat fatal (*crashes*, *stuttering*) secara cepat pada periode kritis peluncuran 72 jam. Otomatisasi ini memangkas durasi penanganan insiden (*Mean Time to Resolve* / MTTR).
2. **Penyusunan Backlog Berbasis Bukti:** Menyediakan data empiris mengenai kritik mekanik kontrol dan estetika naratif bagi tim pengembang. Informasi terstruktur ini mempermudah penetapan prioritas pembaruan konten produk (*product backlog*).
3. **Transparansi Mutu bagi Komunitas:** Menyediakan metrik *Aspect Satisfaction Index* (ASI) dan *Game Scorecard* sebagai panduan evaluasi komprehensif. Calon konsumen memperoleh gambaran mutu produk di luar penilaian biner platform.

---

## 1.6 Sistematika Penulisan
Sistematika penulisan naskah skripsi ini dirancang ke dalam lima bab utama:

1. **BAB I PENDAHULUAN**  
   Bab ini menguraikan latar belakang masalah, fenomena keterbatasan rating biner platform, dan disparitas operasional studio indie. Bagian ini merumuskan masalah penelitian, menetapkan tujuan dan batasan studi, memaparkan manfaat teoretis serta praktis, dan menyajikan sistematika penulisan naskah.

2. **BAB II KAJIAN PUSTAKA**  
   Bab ini memaparkan landasan teori dan tinjauan literatur yang mendasari penelitian. Pembahasan mencakup dinamika komunikasi ulasan pengguna (*eWOM*), karakteristik korpus teks informal, paradigma ACSA, mekanisme *Self-Attention* pada arsitektur *Transformer*, model pembanding Linear SVM, formulasi matematis metrik evaluasi *Active Macro-F1*, formulasi *Aspect Satisfaction Index* (ASI), serta sintesis kerangka berpikir.

3. **BAB III METODOLOGI PENELITIAN**  
   Bab ini merinci alur rekayasa data dan prosedur eksperimen komputasi. Tahapan mencakup akuisisi data via *Steamworks API*, protokol prapemrosesan teks, penyusunan *ground truth* terstratifikasi, jaminan partisi data tanpa kebocoran (*zero data leakage*), perancangan siklus pelatihan manual PyTorch dengan stabilisasi gradien, konfigurasi perangkat keras, serta protokol evaluasi komparatif.

4. **BAB IV HASIL DAN PEMBAHASAN**  
   Bab ini memaparkan temuan empiris hasil eksperimen. Pembahasan meliputi perbandingan metrik klasifikasi Linear SVM dan arsitektur Multi-Model BERT pada *test set* independen, evaluasi matriks konfusi per aspek, analisis galat kualitatif, serta penerapan inferensi skala besar pada korpus 150 judul gim untuk menghasilkan metrik kepuasan aspek riil (*Game Scorecard*).

5. **BAB V KESIMPULAN DAN SARAN**  
   Bab ini merangkum kesimpulan penyelidikan yang menjawab seluruh rumusan masalah berdasarkan bukti empiris. Bab ini juga menyajikan rekomendasi metodologis bagi penelitian lanjutan serta arahan aplikatif bagi industri pengembang gim indie.
