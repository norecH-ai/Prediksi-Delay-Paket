"""Pembersihan data dan rekayasa fitur untuk prediksi keterlambatan paket.

Modul ini dipakai oleh NOTEBOOK dan APLIKASI sehingga logika yang sama berjalan
saat training dan saat inferensi (mencegah training/serving skew).

Alur:
    raw df --clean()--> df bersih --FeatureBuilder--> tabel fitur --ColumnTransformer--> model
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# 1. Pembersihan (deterministik, tanpa statistik dari data -> aman dari leakage)
# --------------------------------------------------------------------------
LAYANAN_MAP = {
    "express": "EXPRESS", "exp": "EXPRESS",
    "reg": "REG",
    "hemat": "HEMAT", "hem": "HEMAT",
}
KOTA_MAP = {
    "jakarta": "Jakarta", "jkt": "Jakarta", "dki jakarta": "Jakarta",
    "bekasi": "Bekasi", "kota bekasi": "Bekasi",
    "surabaya": "Surabaya", "sby": "Surabaya",
    "yogyakarta": "Yogyakarta", "di yogyakarta": "Yogyakarta",
    "jogja": "Yogyakarta", "yogya": "Yogyakarta",
    "denpasar": "Denpasar", "bali": "Denpasar",
}
GRAM_THRESHOLD = 50.0  # berat > 50 "kg" tidak masuk akal -> sebenarnya gram (lihat notebook 01)


def _norm_kota(s: pd.Series) -> pd.Series:
    key = s.astype(str).str.strip().str.lower()
    return key.map(lambda k: KOTA_MAP.get(k, k.title()))


def clean(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Bersihkan data mentah. Tidak menyentuh kolom target."""
    out = df.copy()
    if drop_duplicates:
        out = out.drop_duplicates().reset_index(drop=True)
    out["tanggal_pickup"] = pd.to_datetime(out["tanggal_pickup"])
    out["layanan"] = out["layanan"].astype(str).str.strip().str.lower().map(LAYANAN_MAP)
    out["kota_tujuan"] = _norm_kota(out["kota_tujuan"])
    out["metode_bayar"] = out["metode_bayar"].astype(str).str.strip()
    # Satuan berat campur: nilai > 50 adalah gram -> ubah ke kg. Nilai <= 0 mustahil -> NaN.
    berat = out["berat_kg"].astype(float)
    berat = berat.where(berat <= GRAM_THRESHOLD, berat / 1000.0)
    out["berat_kg"] = berat.where(berat > 0)
    return out


# --------------------------------------------------------------------------
# 2. Rekayasa fitur (hanya memakai informasi yang tersedia SAAT PICKUP)
# --------------------------------------------------------------------------
NUM_RAW = ["jam_pickup", "jarak_km", "jumlah_transit", "janji_hari", "berat_kg", "volume_hub_hari"]
CAT_COLS = ["kota_tujuan", "layanan", "metode_bayar", "cuaca_saat_pickup"]
NUM_ENG = ["berat_log", "volume_log", "volume_peak", "km_per_hari", "slack_hari",
           "hari_ke_minggu", "bulan", "setelah_17", "hujan_lebat", "hujan_apapun",
           "cod", "senin", "berat_hilang"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Imputasi + fitur turunan. Statistik imputasi dipelajari HANYA dari train (fit).

    feature_set: "raw" (fitur asli, untuk baseline) atau "eng" (asli + turunan)
    use_volume : False untuk ablasi kebocoran (tanpa volume_hub_hari)
    use_month  : False untuk menguji ketergantungan pada musim kalender
    """

    def __init__(self, feature_set: str = "eng", use_volume: bool = True, use_month: bool = True):
        self.feature_set = feature_set
        self.use_volume = use_volume
        self.use_month = use_month

    def fit(self, X: pd.DataFrame, y=None):
        self.jarak_kota_ = X.groupby("kota_tujuan")["jarak_km"].median().to_dict()
        self.jarak_global_ = float(X["jarak_km"].median())
        self.berat_median_ = float(X["berat_kg"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        d = pd.DataFrame(index=X.index)
        # Jarak hampir fungsi dari kota -> isi dengan median per kota (fallback global)
        jarak = X["jarak_km"].fillna(X["kota_tujuan"].map(self.jarak_kota_)).fillna(self.jarak_global_)
        berat = X["berat_kg"].fillna(self.berat_median_)
        d["jam_pickup"] = X["jam_pickup"].astype(float)
        d["jarak_km"] = jarak
        d["jumlah_transit"] = X["jumlah_transit"].astype(float)
        d["janji_hari"] = X["janji_hari"].astype(float)
        d["berat_kg"] = berat
        if self.use_volume:
            d["volume_hub_hari"] = X["volume_hub_hari"].astype(float)
        d["kota_tujuan"] = X["kota_tujuan"].values
        d["layanan"] = X["layanan"].values
        d["metode_bayar"] = X["metode_bayar"].values
        cuaca = X["cuaca_saat_pickup"].astype(object)
        d["cuaca_saat_pickup"] = cuaca.where(cuaca.notna(), "tidak_diketahui").values

        if self.feature_set == "eng":
            tgl = pd.to_datetime(X["tanggal_pickup"])
            d["berat_log"] = np.log1p(berat)
            d["km_per_hari"] = jarak / X["janji_hari"]
            d["slack_hari"] = X["janji_hari"] - X["jumlah_transit"]
            d["hari_ke_minggu"] = tgl.dt.dayofweek.astype(float)
            d["senin"] = (tgl.dt.dayofweek == 0).astype(float)
            if self.use_month:
                d["bulan"] = tgl.dt.month.astype(float)
            d["setelah_17"] = (X["jam_pickup"] >= 17).astype(float)
            d["hujan_lebat"] = (cuaca == "hujan_lebat").astype(float)
            d["hujan_apapun"] = cuaca.isin(["hujan", "hujan_lebat"]).astype(float)
            d["cod"] = (X["metode_bayar"] == "COD").astype(float)
            d["berat_hilang"] = X["berat_kg"].isna().astype(float)
            if self.use_volume:
                v = X["volume_hub_hari"].astype(float)
                d["volume_log"] = np.log(v)
                d["volume_peak"] = np.clip(v - 7000.0, 0, None) / 1000.0  # hinge: aktif saat hub padat
        return d


def build_pipeline(estimator, feature_set: str = "eng", scale: bool = False,
                   use_volume: bool = True, use_month: bool = True) -> Pipeline:
    """FeatureBuilder -> ColumnTransformer (one-hot kategori [+ scaling]) -> estimator."""
    fb = FeatureBuilder(feature_set=feature_set, use_volume=use_volume, use_month=use_month)
    num_cols = list(NUM_RAW) + (list(NUM_ENG) if feature_set == "eng" else [])
    if not use_volume:
        num_cols = [c for c in num_cols if c not in ("volume_hub_hari", "volume_log", "volume_peak")]
    if not use_month:
        num_cols = [c for c in num_cols if c != "bulan"]
    num_tf = StandardScaler() if scale else "passthrough"
    ct = ColumnTransformer(
        [("num", num_tf, num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS)],
        verbose_feature_names_out=False,
    )
    return Pipeline([("fe", fb), ("prep", ct), ("model", estimator)])


# --------------------------------------------------------------------------
# 3. Utilitas split waktu & biaya bisnis
# --------------------------------------------------------------------------
TRAIN_END = "2025-12-31"
VAL_START, VAL_END = "2026-01-01", "2026-03-31"
TEST_START = "2026-04-01"
COST_FN = 50_000  # paket terlambat tanpa kabar
COST_FP = 5_000   # notifikasi keliru


def time_split(df: pd.DataFrame):
    t = df["tanggal_pickup"]
    return (df[t <= TRAIN_END].copy(),
            df[(t >= VAL_START) & (t <= VAL_END)].copy(),
            df[t >= TEST_START].copy())


def total_cost(y_true, p, threshold: float) -> float:
    """Biaya total (Rp) bila kita 'kabari' semua paket dengan p >= threshold."""
    y_true = np.asarray(y_true)
    flag = np.asarray(p) >= threshold
    fn = ((~flag) & (y_true == 1)).sum()
    fp = (flag & (y_true == 0)).sum()
    return float(fn * COST_FN + fp * COST_FP)
