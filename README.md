# Automotive Lead Intelligence Engine

Tim sales bisa menerima lebih banyak lead daripada yang sempat mereka follow up.
Project ini membantu menjawab dua pertanyaan: **lead mana yang perlu diprioritaskan,
dan apa tindakan berikutnya?** Semua berjalan lokal, tanpa layanan berbayar.

Model → score/ranking → Decision Engine → priority + action + reason codes

Model belajar pola dari data. **Decision policy adalah aturan demo yang ditulis
engineer**, bukan rekomendasi yang dipelajari model. Pemisahan ini membuat model
bisa diganti tanpa menulis ulang seluruh policy, dan policy bisa direvisi tanpa
retraining. Alasan tiap keputusan juga bisa diaudit.

## Progress

- **v0.1:** synthetic leads, preprocessing, Logistic Regression, ranking, dan business metrics.
- **v0.2:** satu challenger HistGradientBoosting, evaluasi calibration, model selection
  berbasis validation, coefficients, dan permutation importance.
- **v0.3:** decision policy deterministik, tujuh action, reason codes, formatter
  Indonesia, batch output, dan sanity checks. Model tetap Logistic Regression
  tanpa calibration, sesuai keputusan v0.2.

FastAPI, Docker, frontend, LLM, SHAP, database, cloud deployment, dan CI/CD belum
ada. Tidak ada kontak pelanggan yang dikirim otomatis.

## Cara menjalankan

Gunakan Python 3.11+ dari root repo, idealnya di virtual environment:

```bash
python -m pip install -r requirements.txt
# Fresh clone: bangun kembali model yang SUDAH dipilih, tanpa membuka model selection.
python scripts/score_leads.py --rebuild-model
# Berikutnya, pakai artifact yang sama:
python scripts/score_leads.py
python -m pytest -q
```

`--rebuild-model` hanya refit Logistic Regression dengan parameter beku pada
4.000 train rows yang tercatat di report v0.2. Tidak ada challenger, tuning,
calibration, atau evaluasi test di jalur ini. Tanpa flag, artifact harus sudah ada.
Binary model dan CSV batch tidak di-commit. Load joblib hanya dari sumber lokal
tepercaya; file tersebut bisa menjalankan kode saat dibuka.

Output lokal:

- `artifacts/v02/selected_model.joblib`: pipeline model pilihan.
- `artifacts/v03/decisions.csv`: semua lead, score, percentile, priority, action, reason codes.
- `artifacts/v03/decision_summary.json`: distribusi, checks, fingerprint, dan contoh.
- [reports/decision_summary.json](reports/decision_summary.json): salinan ringkasan run yang direview.

CLI menampilkan sepuluh lead dengan score tertinggi dan ringkasan distribusi.
Jalur v0.1/v0.2 tetap tersedia; untuk reproduksi, arahkan ke folder terpisah agar
report historis tidak tertimpa:

```bash
python scripts/train_baseline.py --output artifacts/reproduce-v01 --artifacts artifacts/reproduce-v01
python scripts/compare_models.py --output artifacts/reproduce-v02/model_comparison.json --artifacts artifacts/reproduce-v02
```

## Policy v0.3

Percentile adalah persentase lead dalam batch yang punya score **strictly lebih
rendah**. Score sama mendapat percentile sama; batch dengan semua score sama
mendapat percentile nol. Hasil tergantung komposisi batch dan tidak setara dengan
probability yang terkalibrasi. Batch kecil kurang cocok untuk menentukan kapasitas.

Aturan dievaluasi berurutan; kondisi pertama yang cocok menentukan satu action:

| Kondisi | Priority | Action |
|---|---|---|
| Status kontak unknown; atau status appointment/test drive unknown saat reachable | MEDIUM | REVIEW_LEAD_DATA |
| Kontak intermittent/unreachable | MEDIUM jika percentile ≥60, selain itu LOW | RETRY_CONTACT |
| Follow-up ≥5 dan latency ≥48 jam, tanpa appointment scheduled | MEDIUM jika percentile ≥90, selain itu LOW | NURTURE |
| Percentile ≥90, timeline ≤14 hari, reachable, angka operasional lengkap | URGENT | CONTACT_NOW |
| Appointment scheduled | HIGH | CONFIRM_APPOINTMENT |
| Test drive completed | HIGH jika percentile ≥90, selain itu MEDIUM | FOLLOW_UP_TEST_DRIVE |
| Percentile ≥90 | HIGH | STANDARD_FOLLOW_UP |
| Percentile ≥60, interaksi ≥3, atau appointment attended | MEDIUM | STANDARD_FOLLOW_UP |
| Sisanya | LOW | NURTURE |

Semua threshold ada di `decision.py`. **Ini asumsi policy portfolio, bukan policy
dealer yang tervalidasi.** URGENT dibatasi maksimal 10% batch oleh ranking dan
sanity check. HIGH bisa melebihi 10% karena komitmen appointment perlu ditangani.
Priority bukan SLA otomatis atau quota total kontak.

Contoh aktual: **lead 4320**, score **0.829229**, percentile **99.9**, menghasilkan
**URGENT / CONTACT_NOW** dengan reason codes:
`HIGH_MODEL_SCORE`, `SHORT_PURCHASE_TIMELINE`, `CONTACTABLE`.
Formatter menjelaskan bahwa lead masuk kelompok score tertinggi, rencana pembelian
maksimal 14 hari, dan masih reachable. Score tersebut **bukan klaim peluang beli 82,9%**.

Run 1.000 lead menggunakan membership test v0.2, tanpa memberikan target ke policy:

| Priority | Jumlah | Persentase |
|---|---:|---:|
| URGENT | 9 | 0.9% |
| HIGH | 140 | 14.0% |
| MEDIUM | 375 | 37.5% |
| LOW | 476 | 47.6% |

| Action | Jumlah | Persentase |
|---|---:|---:|
| CONTACT_NOW | 9 | 0.9% |
| CONFIRM_APPOINTMENT | 76 | 7.6% |
| FOLLOW_UP_TEST_DRIVE | 113 | 11.3% |
| RETRY_CONTACT | 400 | 40.0% |
| NURTURE | 167 | 16.7% |
| STANDARD_FOLLOW_UP | 235 | 23.5% |
| REVIEW_LEAD_DATA | 0 | 0.0% |

Kelompok score tinggi mendapat lima jenis action. REVIEW_LEAD_DATA nol pada run
ini karena status kategorikal generator lengkap; fallback-nya diuji lewat tests.
[Detail policy dan audit](docs/decision-policy.md) mencakup precedence, missing
values, ties, asumsi kontak, dan batas penggunaan.

## Hasil model yang dipertahankan

Model selection v0.2 memakai 3.000 fit / 1.000 validation dari train portion.
Challenger tidak lolos syarat improvement validation: AP **0.4648 vs 0.4677**
untuk baseline, dengan lift sama **2.5701×**. Sigmoid memperburuk Brier/log loss
keduanya. Karena itu **KEEP BASELINE**, walaupun challenger kebetulan lebih baik
pada test. Memilih ulang berdasarkan test akan melanggar disiplin model selection.

| Metric test historis | Logistic Regression | HistGradientBoosting |
|---|---:|---:|
| ROC-AUC | 0.7502 | 0.7543 |
| PR-AUC (average precision) | 0.4822 | 0.4961 |
| Precision@10% | 59.00% | 62.00% |
| Recall@10% | 27.44% | 28.84% |
| Lift@10% | 2.7442× | 2.8837× |
| Brier score | 0.143269 | 0.141717 |
| Log loss | 0.451095 | 0.446661 |

Conversion rate test: 21.50%; seluruh synthetic dataset: 21.46%. Hasil ini berasal
dari report historis, **tidak dihitung ulang untuk memilih policy v0.3**.
Ranking menentukan urutan lead; calibration menilai kecocokan probability dengan
frekuensi outcome. Keduanya berbeda. Lift juga bukan tambahan penjualan akibat kontak.

## Data dan batasan

5.000 lead adalah data **synthetic**. Snapshot dibuat tujuh hari setelah lead masuk,
untuk lead yang masih open, dengan target pembelian dalam 30 hari berikutnya.
Model ini belum ditujukan untuk lead baru di hari nol. Generator mengandung noise,
latent intent, missing values, weak predictors, dan kelas yang saling overlap.
Tidak ada data perusahaan, identitas customer, atau business logic rahasia.

Decision policy juga synthetic/demo. Action belum terbukti meningkatkan conversion;
intervention effectiveness belum diukur. Score bersifat predictive, bukan causal.
Threshold production harus mempertimbangkan kapasitas nyata, biaya intervensi,
expected value, dan production data—bukan optimasi test outcome. Consent,
suppression list, jam kontak, dan batas kontak lintas hari belum tersedia: semua
action adalah saran yang harus direview manusia, bukan izin menghubungi customer.

## Struktur dan dokumentasi

Satu package `src/lead_intelligence/` memisahkan data, model, evaluation, comparison,
dan decision. `scripts/score_leads.py` menjalankan batch melalui `decision_demo.py`.
Tidak ada layer kosong atau framework konfigurasi tambahan.

- [Kontrak product dan data](docs/product-and-data.md)
- [Temuan eksplorasi v0.1](docs/week1-findings.md)
- [Audit dan model selection v0.2](docs/model-selection.md)
- [Policy dan audit v0.3](docs/decision-policy.md)

Tests memeriksa pipeline, metric, split, serialization, policy precedence, input
invalid, missing values, score ties, label independence, dan alignment lead_id.
Report JSON historis mempertahankan isi serta field asli untuk reproducibility.
LICENSE MIT tetap memakai teks legal aslinya.

## Roadmap

| Versi | Scope |
|---|---|
| v0.1 | Baseline ranking — selesai |
| v0.2 | Model comparison + calibration — selesai |
| v0.3 | Decision layer — selesai |
| v0.4 | FastAPI prediction/decision service |
| v0.5 | Docker + portfolio polish |
