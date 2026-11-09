"""Inferensi: muat model, prediksi, tier rekomendasi, dan penjelasan faktor risiko."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import clean

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["tanggal_pickup", "jam_pickup", "kota_tujuan", "jumlah_transit", "layanan", "janji_hari",
            "berat_kg", "metode_bayar", "volume_hub_hari"]  # jarak_km & cuaca boleh kosong (diimputasi)

LABEL = {
    "jumlah_transit": "Jumlah transit", "janji_hari": "Janji (SLA) hari", "jarak_km": "Jarak",
    "volume_hub_hari": "Volume hub", "volume_log": "Volume hub (log)", "volume_peak": "Hub sangat padat",
    "setelah_17": "Pickup setelah jam 17", "jam_pickup": "Jam pickup", "slack_hari": "Selisih SLA - transit",
    "km_per_hari": "Jarak per hari SLA", "hujan_lebat": "Hujan lebat", "hujan_apapun": "Hujan",
    "cod": "Pembayaran COD", "senin": "Hari Senin", "berat_kg": "Berat", "berat_log": "Berat",
    "bulan": "Bulan", "hari_ke_minggu": "Hari dalam minggu", "berat_hilang": "Berat tidak tercatat",
}


def load_model(path: Path | str = ROOT / "models" / "model.joblib"):
    return joblib.load(path)


def load_metadata(path: Path | str = ROOT / "models" / "metadata.json") -> dict:
    return json.loads(Path(path).read_text())


def predict_proba(model, df_raw: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(clean(df_raw, drop_duplicates=False))[:, 1]


def recommend(p: float, notify_thr: float, priority_thr: float) -> str:
    if p >= priority_thr:
        return "PRIORITASKAN & KABARI"
    if p >= notify_thr:
        return "KABARI PELANGGAN"
    return "AMAN"


def explain(model, df_raw: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """Kontribusi tiap fitur pada log-odds (koefisien x nilai terstandardisasi), untuk 1 paket.

    Hanya valid untuk model linear. Nilai positif = menaikkan risiko terlambat.
    """
    d = clean(df_raw, drop_duplicates=False)
    prep = model.named_steps["prep"]
    lr = model.named_steps["model"]
    x = prep.transform(model.named_steps["fe"].transform(d))[0]
    contrib = lr.coef_[0] * x
    names = prep.get_feature_names_out()
    out = pd.DataFrame({"fitur": names, "kontribusi": contrib})
    out["fitur"] = out["fitur"].map(lambda n: LABEL.get(n, n.replace("_", " ")))
    out = out.groupby("fitur", as_index=False)["kontribusi"].sum()  # gabung fitur kembar (mis. berat vs berat_log)
    out = out.reindex(out["kontribusi"].abs().sort_values(ascending=False).index)
    return out.head(top).reset_index(drop=True)
