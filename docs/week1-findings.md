# Temuan eksplorasi v0.1

Eksplorasi memakai 4.000 train rows dari generator 5.000 lead, seed 42. Report
lengkap: `reports/data_validation.json`. Test metrics dilaporkan terpisah.

| Pertanyaan | Observasi | Implikasi |
|---|---|---|
| Apakah target imbalanced? | 858/4.000 convert, atau 21.45% | Accuracy saja tidak cukup; gunakan AP dan ranking metrics |
| Feature mana perlu imputasi? | Timeline missing 191; latency 649; financing 206; target dan ID lengkap | Imputation harus fit pada train |
| Apakah timeline membawa signal? | Quartile terpendek: 34.94% convert; terpanjang: 11.18% | Ada signal sekaligus overlap; bukan aturan deterministik |
| Apakah aktivitas bermakna? | Test drive completed: 37.40%; none: 17.01%. Appointment attended: 31.22%; none: 17.79% | Pertahankan aktivitas sebelum snapshot; jangan tafsirkan secara causal |
| Apakah source cukup untuk ranking? | Rate per source hanya 20.66%–22.52% | Weak predictor; efek referral simulator tidak menjamin observed advantage |
| Ada leakage yang terlihat? | ID unik, schema sesuai, tidak ada target copy, ID/target bukan input | Checks dasar lolos; kebenaran waktu tetap asumsi kontrak synthetic |

Distribusi numeric dan kategori disimpan sebagai angka tanpa plot dekoratif.
Grouping numeric memakai quartile; ties bisa mengurangi jumlah bin. Missing rows
dihitung terpisah. Ini validasi simulator, bukan kesimpulan pasar.

Run v0.1 mencatat lift 2.7442 pada K=100. Ada 156 dari 215 converters di luar queue.
Trade-off kapasitas harus terlihat, tetapi tidak membuktikan manfaat intervensi.
v0.2 kemudian mengevaluasi calibration; v0.3 menambahkan policy, tanpa mengubah
angka historis atau mengoptimalkan aturan berdasarkan label test.
