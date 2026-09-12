# Kontrak product dan data

## Siapa pengguna dan keputusan yang dibantu?

Sales representative atau team lead perlu memilih lead yang diprioritaskan saat
waktu follow-up terbatas. Model mengestimasi score conversion; decision policy
v0.3 menggabungkan ranking dengan kondisi operasional untuk memberi satu tindakan.
Saran tetap direview manusia. Pipeline tidak mengirim pesan atau mengakses CRM.

Snapshot adalah **hari ketujuh setelah lead dibuat**, hanya untuk lead yang masih
open dan belum membeli. Target `converted` berarti pembelian dalam **30 hari
setelah snapshot**. Semua aktivitas hanya berasal dari window tujuh hari sebelum
scoring. Timeline berarti sisa hari menuju rencana beli, bukan umur lead.
Response latency adalah rata-rata waktu customer merespons, bukan SLA salesperson.

Satu row mewakili satu lead fiktif yang independen. Day-zero scoring dan snapshot
berulang membutuhkan kontrak serta evaluasi baru. Pada data nyata, outcome yang
belum punya window lengkap tidak boleh diam-diam diberi label negatif; customer
berulang perlu grouping dan timestamps harus diaudit.

## Schema

| Field | Nilai / satuan | Makna |
|---|---|---|
| lead_id | Integer unik | Identitas teknis, bukan feature model |
| purchase_timeline_days | 1–365; bisa missing | Sisa hari menuju rencana beli |
| contactability | reachable / intermittent / unreachable | Status kontak observasi |
| financing_interest | yes / no / undecided; bisa missing | Minat financing, bukan kelayakan kredit |
| test_drive_activity | none / requested / completed | Status test drive |
| vehicle_price_segment | entry / mid / premium | Segmen relatif fiktif |
| previous_interactions | 0–20 | Interaksi customer, tidak termasuk upaya follow-up outbound |
| follow_up_count | 0–12 | Jumlah upaya follow-up outbound |
| lead_source | website / marketplace / walk_in / referral / event | Kanal akuisisi |
| trade_in_interest | yes / no / undecided | Minat trade-in |
| response_latency_hours | 0.1–168; bisa missing | Rata-rata waktu respons customer |
| appointment_activity | none / scheduled / attended / missed | Status appointment |
| converted | 0 / 1 | Target setelah snapshot; tidak diteruskan ke decision |

Test drive dan appointment dianggap aktivitas berbeda. Test drive completed tidak
mengharuskan appointment attended. Tidak ada PII, protected attributes, data
perusahaan, atau aturan internal. Tidak adanya protected attributes bukan bukti fairness.

## Asumsi generator

Latent intent normal memengaruhi timeline, interaksi, respons, appointment, dan
test drive. Score logistik menggabungkan feature dengan latent intent dan noise
normal tambahan (standard deviation 0.85). Timeline pendek, test drive completed,
dan appointment attended diberi signal lebih kuat. Financing, trade-in, source,
dan segment sengaja lemah. Ada interaction effects, saturasi interaksi, dan
penalti follow-up berlebih agar hubungan tidak sepenuhnya linear.

Probability simulator: `0.025 + 0.95 * sigmoid(score)`. Label diambil lewat
Bernoulli draw, jadi lead kuat bisa gagal dan lead lemah bisa convert. Sekitar 5%
recording gaps ditambahkan pada timeline, latency, dan financing. Unreachable
selalu tidak punya latency terukur. Missingness ditambahkan setelah outcome
dibentuk dari observasi lengkap; model melihat informasi lebih sedikit. Latent
intent, score simulator, dan probability pembentuk label tidak diekspor.

Hubungan ini adalah asumsi demonstrasi, **bukan temuan tentang customer automotive**.
Mengubah feature bukan bukti bahwa conversion akan ikut berubah secara causal.

## Leakage dan evaluasi

- Seed 42 dan stratified 80/20 split sebelum fitting; ID dan target dikeluarkan lewat allowlist.
- Median imputation, missing indicators, scaling, dan one-hot encoding fit hanya pada train.
- Validasi otomatis mengecek schema, ID, target, dan exact numeric target copies;
  validitas waktu tetap asumsi karena tidak ada real timestamps.
- Baseline Logistic Regression memakai C=1, lbfgs, max_iter=1000 tanpa class weighting.
  Threshold 0.5 hanya untuk metric klasifikasi historis, bukan policy sales.
- PR-AUC berarti average precision. Confusion matrix: baris actual 0/1, kolom predicted 0/1.
- K = ceil(fraction × jumlah lead test). Precision@K = selected conversions / K;
  Recall@K = selected conversions / seluruh conversions; Lift@K = Precision@K / prevalence.
  Random ranking punya ekspektasi prevalence, K/N, dan lift 1. Tie ranking metric
  memakai urutan input stabil. Tanpa positives, recall/lift adalah null.
- v0.2 melakukan selection di train/validation; v0.3 hanya memakai pilihan beku.

Model mengukur propensity, bukan causal uplift. Tidak ada klaim revenue, fairness,
creditworthiness, atau kenaikan produktivitas. Perbandingan berulang pada holdout
tidak boleh menjadi jalur tuning; data nyata membutuhkan temporal holdout dan
audit censoring. Detail operasional ada di [decision policy](decision-policy.md).
