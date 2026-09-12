# Audit dan model selection v0.2

## Audit v0.1

Audit dilakukan pada commit `a61d747ff5cfa9f3da9d9ea4f999e1cb1162788c`: generator,
feature allowlist, preprocessing, split, metrics, tests, README, dan JSON report.
Git blob hash lokal cocok dengan remote. Tidak ditemukan material leakage atau
metric error, sehingga generator dan baseline dipertahankan.

Input mendahului outcome sesuai kontrak; latent intent, generating probability,
ID, dan target bukan feature. Pipeline fit preprocessing hanya pada training.
Top-K memakai score descending dengan tie stabil; lift menggunakan prevalence
cohort evaluasi. PR-AUC adalah average precision. Noise dan Bernoulli draw membuat
label tidak deterministik. Simulator bias berbeda dengan leakage atau bukti pasar.

Seluruh metric baseline berhasil direproduksi: ROC-AUC 0.750236, AP 0.482240,
dan 59 converters pada top 100. Report v0.1 tidak diubah. Test set sudah pernah
dilihat saat v0.1; istilah yang jujur adalah **holdout yang dipertahankan**, bukan
never-seen data.

## Protokol sebelum eksekusi

Dataset fingerprint dicek terhadap report v0.1. Outer split tetap 4.000 train /
1.000 test. Train dibagi lagi menjadi 3.000 fit / 1.000 validation, stratified,
seed 42. Semua row IDs dicatat supaya pemisahan bisa diaudit.

Dua model family, masing-masing satu konfigurasi:

- Logistic Regression: C=1, lbfgs, max_iter=1000.
- HistGradientBoosting: max_iter=100, learning_rate=0.05, max_leaf_nodes=7,
  min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, seed 42.

Tree kecil membatasi variance sambil menangkap interaction/threshold effects.
Dense one-hot masih murah pada kategori yang sedikit. Scaling tidak dibutuhkan
tree, tetapi dipertahankan untuk konsistensi preprocessing. Tidak ada giant grid search.

Setiap family punya raw dan sigmoid-calibrated candidate. CalibratedClassifierCV
memakai tiga inner stratified folds hanya pada fit rows, termasuk refitting seluruh
preprocessing. `ensemble=False` memasang calibrator pada inner OOF predictions,
lalu refit estimator pada semua fit rows. Keempat candidate dibandingkan pada
validation yang sama. Sigmoid dipilih sebelum hasil dilihat; isotonic tidak dicoba
karena positive sample hanya ratusan dan kita membatasi fleksibilitas calibration.

Adopsi calibration mensyaratkan penurunan Brier ≥0.002 DAN log loss ≥0.005,
dengan kehilangan AP ≤0.01 dan lift ≤0.10. Adopsi challenger mensyaratkan kenaikan
AP ≥0.02 DAN lift ≥0.15, dengan kenaikan Brier maksimal 0.005. Ini kriteria praktis
project, bukan significance test atau hasil perhitungan profit. Pada prevalence
validation 21.4%, lift +0.15 setara kira-kira tiga converters tambahan di queue 100.

AP mengukur ranking lintas recall; Lift@10% menguji asumsi kapasitas sales.
Kenaikan ROC-AUC kecil saja tidak cukup. `selection_frozen.json` ditulis sebelum
prediction test. Setelah pilihan beku, model refit pada semua train rows dan
dievaluasi sekali pada test. Tests otomatis memakai fixture 500 row terpisah.
Reproduksi mengulang eksperimen beku, bukan membuka tuning berbasis test.

## Keputusan yang dijalankan

**KEEP BASELINE: Logistic Regression tanpa calibration.**

| Candidate | AP validation | Lift@10% | Brier | Log loss |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.467682 | 2.570093 | 0.144997 | 0.454130 |
| Logistic Regression + sigmoid | 0.467682 | 2.570093 | 0.145697 | 0.456108 |
| HistGradientBoosting | 0.464771 | 2.570093 | 0.145036 | 0.454597 |
| HistGradientBoosting + sigmoid | 0.464771 | 2.570093 | 0.145426 | 0.456020 |

Challenger AP turun 0.002910 dan lift tidak berubah. Sigmoid memperburuk Brier dan
log loss keduanya. Tambahan complexity tidak memenuhi kriteria.

Pada test, challenger justru mendapat AP 0.496096 vs 0.482240 untuk baseline.
Pilihan **tidak dibalik**: memilih setelah melihat test berarti memakai test
sebagai selection set. v0.3 mengonsumsi keputusan ini tanpa menjalankan comparison ulang.

## Ranking, calibration, dan model reliance

Ranking menentukan urutan queue. Calibration menilai apakah kelompok dengan
probability tertentu punya observed rate yang sejalan. Sigmoid monotonic dapat
mengubah probability tanpa mengubah ranking, seperti hasil validation di atas.
Brier/log loss dan sepuluh reliability bins berisi count ada dalam report JSON.
Empty bins eksplisit; sparse bins dan ECE bergantung pada binning. Proper scores
juga dipengaruhi discrimination, jadi bukan diagnosis calibration murni.

Threshold priority kelak butuh evidence lebih dari ranking. Score synthetic,
baik raw maupun calibrated, bukan peluang beli nyata customer.

Coefficients Logistic Regression dipasangkan dengan transformed feature names.
Numeric sudah distandardisasi dan diimputasi. Semua kategori one-hot dipertahankan,
jadi satu coefficient bukan reference-category contrast. Missing indicators
memiliki nama. Coefficients mendeskripsikan log-odds model, bukan sebab-akibat.

Permutation importance pada validation memakai AP decrease dan tiga repeats.
Signal baseline terbesar: test-drive activity (0.0997), appointment activity
(0.0602), purchase timeline (0.0586). Analisis memakai fit-only model, bukan test
cohort. Feature berkorelasi bisa membagi importance; shuffle bisa menghasilkan
kombinasi tidak lazim. Standard deviation repeats bukan confidence interval populasi.

## Batasan

Satu validation split tetap noisy. Data synthetic tidak menangkap CRM drift,
customer berulang, timestamp problems, atau censoring nyata. Tidak ada klaim
statistical significance atau real-world generalization. Report historis disimpan
apa adanya agar angka, keputusan, dan dependency versions tetap bisa diaudit.
