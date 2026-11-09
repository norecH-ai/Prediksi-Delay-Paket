import pandas as pd
from src.features import clean, time_split
from src.predict import load_model, predict_proba

RAW = pd.DataFrame([
    dict(id_resi="A", tanggal_pickup="2025-01-02", jam_pickup=10, kota_tujuan="Jkt", jarak_km=None, jumlah_transit=1,
         layanan="EXP", janji_hari=2, berat_kg=1500.0, metode_bayar="COD", cuaca_saat_pickup=None, volume_hub_hari=5000, terlambat=0),
    dict(id_resi="B", tanggal_pickup="2025-01-02", jam_pickup=18, kota_tujuan="Kota Bekasi", jarak_km=130.0, jumlah_transit=2,
         layanan="reg", janji_hari=3, berat_kg=-1.0, metode_bayar="non_COD", cuaca_saat_pickup="hujan", volume_hub_hari=9000, terlambat=1),
])


def test_clean_normalizes():
    d = clean(pd.concat([RAW, RAW.iloc[[0]]]))
    assert len(d) == 2                                   # duplikat dibuang
    assert list(d.layanan) == ["EXPRESS", "REG"]
    assert list(d.kota_tujuan) == ["Jakarta", "Bekasi"]
    assert d.berat_kg.iloc[0] == 1.5 and pd.isna(d.berat_kg.iloc[1])   # gram->kg, negatif->NaN


def test_model_predicts_probabilities():
    p = predict_proba(load_model(), RAW.drop(columns="terlambat"))
    assert ((0 <= p) & (p <= 1)).all() and p[1] > p[0]   # hari puncak + transit lebih banyak = lebih berisiko
