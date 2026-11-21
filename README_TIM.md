# [Nama proyek] · Tim [nama tim]

> Salin berkas ini menjadi `README.md` di repositori tim kalian, lalu isi semua bagian di dalam kurung siku.

**Kasus:** [nomor dan nama kasus] · **Aplikasi:** [tautan aplikasi] · **Video:** [tautan video]

| Anggota | NIM | Bagian yang dikerjakan |
|---|---|---|
| [nama] | [NIM] | [misalnya EDA dan pembersihan data] |
| [nama] | [NIM] | [misalnya model dan evaluasi] |
| [nama] | [NIM] | [misalnya API dan aplikasi] |

## Masalah dan pengguna

[Dua atau tiga kalimat: siapa penggunanya dan keputusan apa yang dibantu oleh model ini.]

## Data dan pembersihan

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---|---|
| [misalnya baris duplikat] | [angka] | [misalnya dihapus dengan drop_duplicates] |

## Hasil

| Model | [metrik utama] *validation* | [metrik utama] *test* |
|---|---|---|
| *Baseline*: [nama *baseline*] | | |
| [model final] | | |

*Threshold* yang dipakai: [angka], karena [alasan bisnis].

## Cara menjalankan

```bash
pip install -r requirements.txt
uvicorn app.api.main:app --reload       # API
streamlit run app/ui/streamlit_app.py   # aplikasi
```

## Struktur repositori

```
notebooks/      notebook dari EDA sampai penyimpanan model
data/           data mentah (dan data bersih jika perlu)
src/features.py fungsi pembuat fitur, dipakai saat pelatihan dan saat prediksi
artifacts/      model.joblib + metadata.json
app/            API (FastAPI) dan antarmuka (Streamlit)
```

## Keterbatasan

[Dalam kondisi apa model ini tidak dapat dipercaya? Data apa yang belum tersedia?]

## Daftar periksa sebelum dikumpulkan

- [ ] Repositori bersifat publik dan setiap anggota memiliki *commit*
- [ ] *Notebook* dapat dijalankan ulang dari awal sampai akhir tanpa galat
- [ ] *Baseline* wajib dilaporkan dan dibandingkan
- [ ] Data *test* hanya dibuka satu kali, di akhir
- [ ] `artifacts/metadata.json` berisi versi model, fitur masukan, *threshold*, dan metrik
- [ ] Aplikasi dapat dibuka orang lain melalui tautan, termasuk dari ponsel
- [ ] Aplikasi tidak berhenti mendadak ketika menerima masukan yang tidak wajar
- [ ] Laporan paling banyak 6 halaman dan video paling lama 5 menit
