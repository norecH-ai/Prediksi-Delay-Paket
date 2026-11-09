"""Latih model final dan simpan artefak.   Jalankan:  python -m src.train

Urutan (sama dengan notebook 02):
 1. bersihkan data -> split waktu (train / validation / test)
 2. baseline wajib: regresi logistik FITUR ASLI
 3. kandidat terpilih: regresi logistik + fitur rekayasa
 4. threshold dari biaya bisnis (Rp50.000 vs Rp5.000), dikonfirmasi pada validation
 5. refit pada train+validation, evaluasi SEKALI pada test
 6. simpan models/model.joblib + models/metadata.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.features import (COST_FN, COST_FP, build_pipeline, clean, time_split, total_cost,
                          NUM_RAW, CAT_COLS)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
META_PATH = ROOT / "models" / "metadata.json"

# Keputusan biaya: kabari bila p * 50.000 > (1 - p) * 5.000  ->  p > 5.000 / 55.000
THRESHOLD = COST_FP / (COST_FP + COST_FN)   # = 0.0909
PRIORITY_THRESHOLD = 0.40                    # tier "prioritaskan": paket yang hampir pasti terlambat
LR_PARAMS = dict(C=1.0, max_iter=3000)


def make_model(feature_set="eng"):
    return build_pipeline(LogisticRegression(**LR_PARAMS), feature_set=feature_set, scale=True)


def metrics(y, p, thr=THRESHOLD) -> dict:
    y = np.asarray(y)
    flag = p >= thr
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    return {
        "n": int(len(y)), "prevalence": round(float(y.mean()), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "precision@thr": round(tp / max(tp + fp, 1), 4),
        "recall@thr": round(tp / max(tp + fn, 1), 4),
        "cost_per_package_rp": round(total_cost(y, p, thr) / len(y)),
        "cost_never_notify_rp": round(COST_FN * float(y.mean())),
        "cost_always_notify_rp": round(COST_FP * (1 - float(y.mean()))),
    }


def main():
    df = clean(pd.read_csv(DATA))
    tr, va, te = time_split(df)
    trva = pd.concat([tr, va])
    y = lambda d: d["terlambat"]

    # --- baseline wajib (fitur asli) dan kandidat (fitur rekayasa) pada train -> validation
    base = make_model("raw").fit(tr, y(tr))
    cand = make_model("eng").fit(tr, y(tr))
    val_base = metrics(y(va), base.predict_proba(va)[:, 1])
    val_cand = metrics(y(va), cand.predict_proba(va)[:, 1])

    # --- refit pada train+validation (agar model melihat lonjakan Maret 2026), lalu test sekali
    final = make_model("eng").fit(trva, y(trva))
    base_final = make_model("raw").fit(trva, y(trva))
    test_final = metrics(y(te), final.predict_proba(te)[:, 1])
    test_base = metrics(y(te), base_final.predict_proba(te)[:, 1])

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)

    fe = final.named_steps["fe"]
    meta = {
        "model": "LogisticRegression (C=1.0) + fitur rekayasa, terstandardisasi",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "train_data": f"{trva['tanggal_pickup'].min().date()} .. {trva['tanggal_pickup'].max().date()} (n={len(trva)})",
        "test_data": f"{te['tanggal_pickup'].min().date()} .. {te['tanggal_pickup'].max().date()} (n={len(te)})",
        "threshold": {
            "notify": round(THRESHOLD, 4), "priority": PRIORITY_THRESHOLD,
            "rule": "kabari bila p*50000 > (1-p)*5000  =>  p > 5000/55000",
            "cost_fn_rp": COST_FN, "cost_fp_rp": COST_FP,
        },
        "metrics": {
            "validation_baseline_LR_fitur_asli": val_base,
            "validation_kandidat_LR_fitur_rekayasa": val_cand,
            "test_baseline_LR_fitur_asli": test_base,
            "test_final_model": test_final,
        },
        "input_features": NUM_RAW + CAT_COLS + ["tanggal_pickup"],
        "imputation": {"jarak_km": "median per kota (dari train)", "berat_kg": f"median train = {fe.berat_median_:.3f}",
                       "cuaca_saat_pickup": "kategori 'tidak_diketahui'"},
        "known_limitations": [
            "volume_hub_hari harus tersedia saat prediksi (estimasi beban hub); tanpanya PR-AUC turun tajam.",
            "Test (Apr-Jun 2026) tidak memuat hari puncak; performa di musim puncak hanya terbukti di validation.",
            "Pemantauan drift dan retraining berkala diperlukan (pola musiman Maret & Nov-Des).",
        ],
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta["metrics"], indent=2))


if __name__ == "__main__":
    main()
