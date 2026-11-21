# Prediksi Keterlambatan Paket (Hub Bandung) · Tim OTOKOWOK

**Kasus:** 08 · Keterlambatan paket · **Aplikasi:** https://prediksi-delay-paket-a9hwlyfxdj9gg7expzv8cp.streamlit.app/
**Video:** https://drive.google.com/file/d/1KxLZppon2mMZPtcqVkzfETvZXXt-2byF/view?usp=drive_link

| Anggota | NIM | Bagian yang dikerjakan |
|---|---|---|
| Haidar Yahya | 2402909 | Audit data dan pembersihan, EDA, baseline dan perbandingan, Laporan |
| Achmad Choeron | 2403160 | Threshold biaya, kalibrasi, dan evaluasi final, aplikasi Streamlit, deploy, Video |


## Masalah dan pengguna

Pengguna adalah tim operasional hub Bandung. Saat paket dijemput, model menghitung probabilitas paket akan terlambat, lalu menyarankan salah satu tindakan: `AMAN`, `KABARI PELANGGAN`, atau `PRIORITASKAN & KABARI`. Keputusan ini didasarkan pada biaya: paket terlambat tanpa kabar Rp50.000, sedangkan notifikasi yang ternyata tidak perlu hanya Rp5.000.

## Data dan pembersihan

Data mentah berisi 40.500 baris dan 13 kolom. Setelah duplikat dibuang tersisa 40.000 paket (tingkat terlambat 17,8%).

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---|---|
| Baris duplikat | 500 | Dihapus dengan `drop_duplicates` sebelum split agar skor tidak menggelembung |
| Penulisan `layanan` tidak seragam (7 varian, mis. `Express`, `EXP`, `reg`) | Seluruh kolom | Dinormalisasi menjadi 3 kategori: `REG`, `EXPRESS`, `HEMAT` |
| Penulisan `kota_tujuan` tidak seragam (28 varian, mis. `Jkt`, `DKI Jakarta`) | Seluruh kolom | Dinormalisasi menjadi 15 kota |
| `berat_kg` bersatuan gram (nilai > 50) | 1.197 | Dibagi 1000 menjadi kg (distribusinya identik dengan kg setelah konversi) |
| `berat_kg` nol atau negatif (mustahil) | 80 | Diubah menjadi kosong (NaN), lalu diimputasi median berat dari data latih; ditandai fitur `berat_hilang` |
| `jarak_km` kosong | 800 | Diimputasi median jarak per kota tujuan (dari data latih saja) |
| `cuaca_saat_pickup` kosong | 400 | Diisi kategori `tidak_diketahui` |
| `volume_hub_hari` tinggi (tampak seperti outlier) | Hari puncak (Maret, Nov-Des) | Dipertahankan karena merupakan sinyal hari puncak (±75% paket terlambat); dimodelkan dengan fitur ambang (*hinge*) 7.000 paket |
| `janji_hari` tidak cocok dengan layanan hasil normalisasi | ±800 | Belum diketahui penyebabnya; dicatat sebagai pertanyaan untuk klien |

Data dibagi berdasarkan waktu: *train* Jan-Des 2025, *validation* Jan-Mar 2026, *test* Apr-Jun 2026. Model final dilatih ulang pada *train* + *validation* (33.256 baris), lalu *test* (6.744 baris) dibuka satu kali.

## Hasil

| Model | PR-AUC *validation* | PR-AUC *test* |
|---|---|---|
| *Baseline*: regresi logistik, fitur asli | 0,646 | 0,355 |
| **Final: regresi logistik + fitur rekayasa** | **0,660** | **0,379** |

Biaya per paket pada data *test*: model final Rp3.340, *baseline* Rp3.456, tidak pernah mengabari Rp6.962, mengabari semua Rp4.304. Model final menghemat ±52% dibanding tidak pernah mengabari dan ±22% dibanding mengabari semua.

LightGBM dengan beragam kapasitas dicoba, tetapi tidak lebih baik (PR-AUC *validation* ≤ 0,661 dan *test* ≤ 0,378). Struktur datanya aditif pada log-odds, sehingga model yang lebih sederhana dan mudah dijelaskan dipilih.

*Threshold* yang dipakai: **0,0909** (`5.000 / (5.000 + 50.000)`), karena kabari pelanggan hanya menguntungkan bila `p × Rp50.000 > (1 − p) × Rp5.000`. Dikonfirmasi lewat kurva biaya pada *validation*. Paket dengan `p ≥ 0,40` masuk tingkat "prioritaskan".

Catatan: prevalensi *validation* (21,9%) lebih tinggi daripada *test* (13,9%) karena *validation* memuat lonjakan Maret, sehingga kedua angka PR-AUC tidak langsung sebanding.

## Cara menjalankan

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt     # lingkungan pengembangan lengkap
python -m src.train                     # melatih ulang model -> models/
pytest -q                               # menjalankan tes
streamlit run app/app.py                # aplikasi lokal
```

Untuk deploy di Streamlit Community Cloud, `requirements.txt` di root repositori hanya perlu berisi paket yang dipakai aplikasi (`streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`), dengan `scikit-learn` sama persis dengan versi pelatihan (1.9.1).

## Struktur repositori

```
data/raw.csv                  data mentah (tidak diubah)
notebooks/01_audit_cleaning   audit data dan bukti tiap aturan pembersihan
notebooks/02_modeling         split waktu, baseline, fitur, LightGBM, ablasi, kalibrasi, threshold
src/features.py               clean() + FeatureBuilder + build_pipeline() (dipakai notebook DAN aplikasi)
src/train.py                  melatih dan menyimpan model
src/predict.py                memuat model, prediksi, rekomendasi, penjelasan faktor
models/model.joblib           pipeline lengkap (fitur + scaler + model)
models/metadata.json          versi model, fitur masukan, threshold, metrik, keterbatasan
app/app.py                    aplikasi Streamlit (satu paket + unggah CSV)
tests/                        pytest
```

## Keterbatasan

* `volume_hub_hari` harus sudah diketahui **saat pickup**. Bila hanya tersedia di akhir hari, nilainya tidak boleh dipakai apa adanya. Tanpa fitur ini, PR-AUC *validation* turun dari ≈0,66 menjadi ≈0,43. Ketersediaannya perlu dikonfirmasi ke klien.
* Data *test* (Apr-Jun 2026) hampir tanpa hari puncak, sehingga performa di musim puncak hanya terbukti pada *validation*.
* Di hari biasa kemampuan membedakan terbatas (ROC-AUC *test* ≈ 0,76): banyak keterlambatan tidak dapat diprediksi dari data pickup saja. Pada *threshold* 0,0909, presisi hanya ±21% (recall ±82%), sehingga banyak notifikasi akan "berlebih" (ini sengaja, karena biayanya jauh lebih murah).
* Model tidak dapat dipercaya untuk kota tujuan, layanan, atau kondisi di luar data latih, dan ±800 baris punya `janji_hari` yang belum dapat dijelaskan.
* Perlu pemantauan *drift* (tingkat terlambat, distribusi volume) dan *retraining* berkala karena pola musiman (Maret dan Nov-Des).
