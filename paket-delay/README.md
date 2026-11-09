# 📦 Kasus 08 · Prediksi Keterlambatan Paket (hub Bandung)

Model *binary classification* yang menerima data satu paket **saat dijemput** dan mengeluarkan
**probabilitas terlambat** beserta rekomendasi: `AMAN` · `KABARI PELANGGAN` · `PRIORITASKAN & KABARI`.

## Hasil singkat

| | PR-AUC val | PR-AUC **test** | Biaya/paket (test) |
|---|---|---|---|
| Baseline: regresi logistik, fitur asli | 0,646 | 0,355 | Rp3.456 |
| **Final: regresi logistik + fitur rekayasa** | 0,660 | **0,379** | **Rp3.340** |
| LightGBM (beragam kapasitas) | ≤ 0,661 | ≤ 0,378 | – |
| Tak pernah mengabari / mengabari semua | – | – | Rp6.962 / Rp4.304 |

* Prevalensi test 13,9% (acuan acak PR-AUC = 0,139). Val 21,9% karena memuat lonjakan Maret, **jadi val dan test tidak sebanding**.
* **Temuan:** model yang lebih rumit (LightGBM) tidak lebih baik dari regresi logistik berfitur benar. Struktur datanya aditif pada log-odds; satu-satunya non-linearitas besar (ambang volume hub) sudah ditangkap fitur *hinge*. Detail dan alasan di `notebooks/02_modeling.ipynb`.
* **Threshold = 0,0909** (`5.000 / (5.000 + 50.000)`): kabari pelanggan bila p lebih besar dari ini. Dikonfirmasi lewat kurva biaya validation; probabilitas model terkalibrasi baik. `p ≥ 0,40` → prioritaskan.
* Hemat ±52% biaya dibanding tak pernah mengabari, ±22% dibanding mengabari semua (test).

## Struktur repo
```
data/raw.csv                  data mentah (tidak diubah)
notebooks/01_audit_cleaning   audit data, bukti tiap aturan cleaning
notebooks/02_modeling         split waktu, baseline, rekayasa fitur, LightGBM, ablasi, kalibrasi, threshold
src/features.py               clean() + FeatureBuilder + build_pipeline()  (dipakai notebook DAN app)
src/train.py                  melatih & menyimpan model:   python -m src.train
src/predict.py                load model, prediksi, rekomendasi, penjelasan faktor
models/model.joblib           pipeline lengkap (fitur + scaler + model)
models/metadata.json          metrik, threshold, data latih, asumsi, keterbatasan
app/app.py                    aplikasi Streamlit (satu paket + unggah CSV)
tests/                        pytest
```

## Menjalankan ulang
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.train                    # melatih ulang -> models/
pytest -q                              # tes
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
streamlit run app/app.py               # aplikasi lokal
```

## Keputusan penting
1. **Duplikat dibuang sebelum split** (500 baris identik) agar skor tak menggelembung.
2. **Cleaning berbasis bukti:** kategori `layanan` (7→3) dan `kota` (28→15) dinormalisasi; `berat_kg > 50` ternyata **gram** (distribusinya identik dengan kg setelah ÷1000) dan `≤ 0` → NaN. Cleaning terbukti **tidak** mengubah skor, tetapi menjaga kebenaran data.
3. **Imputasi tanpa leakage:** median jarak per kota dan median berat dihitung dari train saja (`FeatureBuilder.fit`).
4. **`volume_hub_hari` bukan outlier** melainkan sinyal hari puncak (Maret, Nov–Des; ±75% paket terlambat). Dipertahankan dan dimodelkan dengan fitur ambang.
5. **Refit pada train+validation** setelah pemilihan model, lalu test dievaluasi **sekali**.

## Asumsi & keterbatasan (baca sebelum dipakai produksi)
* **Ketersediaan `volume_hub_hari` saat pickup harus dikonfirmasi ke klien.** Bila hanya diketahui di akhir hari, ia
  tidak boleh dipakai apa adanya. Ablasi: tanpa volume, PR-AUC validation turun ≈0,66 → ≈0,43; di hari biasa (test) hampir tidak berubah.
* Test (Apr–Jun 2026) hampir tanpa hari puncak; performa di musim puncak hanya terbukti di validation.
* Di hari biasa kemampuan membedakan terbatas (ROC-AUC test ≈ 0,76): banyak keterlambatan tidak terprediksi dari data pickup.
* ±800 baris punya `janji_hari` yang tidak cocok dengan layanan hasil normalisasi; penyebabnya belum diketahui (tanyakan ke klien).
* Perlu pemantauan drift (tingkat terlambat, distribusi volume) dan retraining berkala.

## Deploy (Streamlit Community Cloud)
1. Push seluruh folder ini ke repo GitHub (publik, atau privat bila akun terhubung). Pastikan `models/model.joblib` ikut ter-commit.
2. Buka <https://share.streamlit.io>, login dengan GitHub → **New app**.
3. Pilih repo + branch, **Main file path:** `app/app.py`. Di *Advanced settings* pilih Python 3.11 atau lebih baru.
4. **Deploy.** Tautan publik muncul dalam beberapa menit.
5. Bila muncul galat versi scikit-learn saat memuat model, jalankan `python -m src.train` di lingkungan yang sama, commit `models/` baru, push ulang.
