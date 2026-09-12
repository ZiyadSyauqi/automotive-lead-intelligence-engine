# Decision policy v0.3

## Audit model yang dikonsumsi

Main sebelum perubahan: `6cfae8ad022729a1aa93965eeaf994b4bc90f7c2`. Source lokal
cocok dengan Git blob hash remote. Report v0.2 memilih `logistic_regression` raw.
Artifact yang tersedia adalah sklearn Pipeline dengan classifier LogisticRegression,
feature names sesuai schema, kelas [0,1], dan parameter sesuai report pilihan.
Serialization sudah diuji pada v0.1/v0.2. Tidak ditemukan bug material yang
mengharuskan perubahan generator, preprocessing, atau model selection.

`decision_demo.py` mengecek keputusan beku, fingerprint dataset/report, keutuhan
train/test IDs, tipe classifier, feature names, kelas, dan parameter. Default-nya
memakai artifact yang ada. Fresh clone memakai `--rebuild-model` untuk refit
pilihan yang sama pada train IDs dari report; tidak compare model atau mengukur
metric test lagi. Hash artifact dicatat untuk audit run. Checks struktur bukan
bukti kriptografis bahwa artifact dilatih oleh proses yang diharapkan: pakai hanya
artifact lokal tepercaya. Tidak ada model registry pada versi ini.

## Kontrak output dan pemisahan tanggung jawab

`decide()` menerima operational signals, model_score, dan score_percentile.
`DecisionResult` berisi satu priority, satu recommended_action, dan reason_codes.
`decide_batch()` menerima DataFrame lead dan Series score dengan index lead_id.
Output mempertahankan urutan input. CSV menyimpan reason_codes sebagai JSON list.
Tidak ada label conversion yang diteruskan ke policy atau ringkasan.

Model belajar pola untuk menghasilkan score. Policy menerjemahkan score dan
konteks ke saran tindakan. Reason codes menjelaskan **policy yang terpenuhi**,
bukan feature attribution model atau alasan perilaku customer. Formatter
Indonesia hanya presentation; structured fields tetap sumber kebenaran.

Tidak ada klaim score 0.71 berarti peluang beli nyata 71%. Ranking digunakan
karena calibration tambahan tidak menang pada validation v0.2. Policy bisa
berubah tanpa retraining, model bisa diganti selama score contract cocok, dan
business assumptions tetap terlihat. Jalur demo saat ini sengaja mengunci model
pilihan v0.2; perubahan model harus disengaja dan diaudit.

## Threshold dan percentile

Semua konstanta ada di `decision.py::POLICY`:

| Konstanta | Nilai | Alasan demo |
|---|---:|---|
| high_percentile | 90 | Fokus kelompok top 10% |
| medium_percentile | 60 | Top 40% masih mendapat follow-up standar |
| near_purchase_days | 14 | Rencana dekat dapat memperkuat urgency |
| meaningful_interactions | 3 | Proxy engagement pada window tujuh hari |
| followup_limit | 5 | Batasi follow-up berulang saat respons lambat |
| slow_response_hours | 48 | Dipakai bersama follow-up count untuk nurture |
| urgent_max_fraction | 0.10 | Guardrail batch; bukan target yang harus dipenuhi |

Percentile = jumlah score yang strictly lebih rendah / jumlah batch ×100.
Tie tidak dipecah memakai lead_id; semua score sama berarti percentile nol.
Batch singleton juga nol. Memasukkan atau mengeluarkan lead dapat mengubah
percentile lead lain. Batch kecil, homogen, atau semua score lemah perlu review:
top 10% tetap relatif, bukan bukti bahwa lead tersebut bagus secara absolut.

Semua threshold ditetapkan sebelum distribusi run dilihat dan tidak dioptimalkan
dengan conversion label. Threshold production harus mempertimbangkan kapasitas,
biaya intervensi, expected value, dan data production. HIGH bukan jaminan masuk
quota 10%: appointment bisa mengangkat lead dari kelompok score rendah.

## Precedence: aturan pertama yang cocok

1. Kontak unknown → MEDIUM / REVIEW_LEAD_DATA.
2. Intermittent atau unreachable → RETRY_CONTACT, MEDIUM bila percentile ≥60,
   selain itu LOW. Ini mendahului appointment dan urgency.
3. Pada lead reachable, status appointment/test drive unknown → MEDIUM / REVIEW_LEAD_DATA.
4. Follow-up ≥5 DAN latency ≥48 jam TANPA appointment scheduled → NURTURE;
   MEDIUM untuk percentile ≥90, selain itu LOW. Appointment adalah komitmen yang
   tidak dibatalkan oleh guardrail ini.
5. Percentile ≥90 DAN timeline ≤14 DAN seluruh angka operasional lengkap →
   URGENT / CONTACT_NOW. Lead sudah dipastikan reachable. Ini mendahului
   konfirmasi appointment; petugas dapat membahas appointment dalam kontak utama.
6. Appointment scheduled → HIGH / CONFIRM_APPOINTMENT.
7. Test drive completed → FOLLOW_UP_TEST_DRIVE; HIGH bila percentile ≥90, selain itu MEDIUM.
8. Percentile ≥90 → HIGH / STANDARD_FOLLOW_UP.
9. Interaksi ≥3 atau appointment attended atau percentile ≥60 → MEDIUM / STANDARD_FOLLOW_UP.
10. Sisanya → LOW / NURTURE.

Ada tujuh action total, termasuk REVIEW_LEAD_DATA. REVIEW bukan aksi menghubungi
customer. RETRY_CONTACT berarti review kanal/waktu dan izin sebelum percobaan,
bukan instruksi untuk menelpon berkali-kali. NURTURE belum punya jadwal otomatis.
Data tidak punya event timestamps detail, jadi `LOW_RECENT_ENGAGEMENT` hanya
berarti tidak ada proxy engagement pada window observasi; bukan recency event
atau ukuran minat customer yang pasti.

## Missing dan input invalid

- Score NaN/infinite/di luar [0,1], percentile invalid, batch kosong, ID duplicate,
  dan index score yang tidak cocok → ValueError. Tidak menerbitkan decision palsu.
- Status kontak missing/unseen → REVIEW_LEAD_DATA. Status appointment/test drive
  missing/unseen juga direview bila reachable; status kontak sulit dijangkau
  tetap menghasilkan RETRY_CONTACT terlebih dahulu.
- Numeric missing, nonfinite, negatif, melebihi domain, atau count fractional
  dianggap unknown dan diberi OPERATIONAL_DATA_INCOMPLETE. Tidak diisi nol dan
  tidak bisa menghasilkan URGENT. Aturan appointment/test drive/ranking lain masih
  bisa berlaku berdasarkan data yang diketahui.
- Unreachable memang tidak punya latency terukur di generator; jalur RETRY_CONTACT
  tidak membutuhkan angka tersebut. Missing operational field pada `decide()`
  punya fallback, tetapi scoring model batch tetap membutuhkan kolom FEATURES.
- Policy tidak memakai financing, vehicle segment, atau source secara langsung.
  Nilai tersebut tetap input ML. Unseen kategori model ditangani OneHotEncoder.

## Sanity checks dan hasil

Run 1.000 lead: URGENT 9, HIGH 140, MEDIUM 375, LOW 476. Action: CONTACT_NOW 9,
CONFIRM_APPOINTMENT 76, FOLLOW_UP_TEST_DRIVE 113, RETRY_CONTACT 400, NURTURE 167,
STANDARD_FOLLOW_UP 235, REVIEW_LEAD_DATA 0. Lima action muncul pada kelompok score
tinggi. Tidak ada policy revision berdasarkan conversion outcome.

Hard checks menolak enum invalid, reason kosong, ID duplicate, URGENT >10%,
CONTACT_NOW tanpa URGENT, dan reason kontak yang bertentangan dengan action.
Action homogen di kelompok atas menghasilkan warning diagnostik; batch yang
homogen bisa memang butuh action sama, jadi kita tidak memaksakan keragaman.
Tests memakai input buatan untuk tiap rule, boundaries, precedence, missing data,
ties, determinisme, label independence, dan alignment saat urutan batch diubah.

Contoh run: lead 4320, score 0.829229, percentile 99.9 → URGENT / CONTACT_NOW.
Reason codes: HIGH_MODEL_SCORE, SHORT_PURCHASE_TIMELINE, CONTACTABLE. Ini menjelaskan
kenapa policy memberi treatment lebih kuat, bukan bukti causal conversion.

## Batasan operasional

Data synthetic dan policy demo. Efektivitas action, intervention lift, dan
peningkatan conversion belum diukur. Tidak ada consent field, suppression list,
business-hour calendar, contact channel validity, atau histori lintas hari.
Manusia harus memeriksa semua itu sebelum bertindak. Tidak ada SLA dua jam yang
dijanjikan, pesan otomatis, keputusan kredit, atau penolakan layanan.

Asumsi terbesar: ranking relatif ditambah status operasional cukup berguna untuk
mengatur perhatian tim sales. Bahkan model bagus tidak membuktikan lead dengan
score tertinggi paling diuntungkan oleh kontak. Itu membutuhkan validasi nyata.
