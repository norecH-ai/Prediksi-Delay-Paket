"""Aplikasi Streamlit: prediksi keterlambatan paket saat pickup (hub Bandung)."""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agar `src` dapat diimpor
from src.predict import REQUIRED, explain, load_metadata, load_model, predict_proba, recommend  # noqa: E402

st.set_page_config(page_title="Prediksi Keterlambatan Paket", page_icon="📦", layout="wide")


@st.cache_resource
def get_model():
    return load_model(), load_metadata()


model, meta = get_model()
NOTIFY = meta["threshold"]["notify"]
PRIORITY = meta["threshold"]["priority"]
jarak_kota = model.named_steps["fe"].jarak_kota_
KOTA = sorted(jarak_kota)

COLOR = {"AMAN": "#2e7d32", "KABARI PELANGGAN": "#ef6c00", "PRIORITASKAN & KABARI": "#c62828"}
HINT = {
    "AMAN": "Risiko rendah. Tidak perlu tindakan khusus.",
    "KABARI PELANGGAN": "Kirim pemberitahuan 'kemungkinan terlambat' ke pelanggan lebih awal.",
    "PRIORITASKAN & KABARI": "Risiko tinggi. Prioritaskan di sortir/pengiriman dan kabari pelanggan.",
}

st.title("📦 Prediksi Keterlambatan Paket")
st.caption("Hub Bandung · masukkan data paket saat dijemput → probabilitas terlambat + rekomendasi tindakan")

tab1, tab2, tab3 = st.tabs(["Satu paket", "Unggah CSV (banyak paket)", "Tentang model"])

# ------------------------------------------------------------------ satu paket
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        tanggal = st.date_input("Tanggal pickup", value=date.today())
        jam = st.slider("Jam pickup", 7, 22, 14)
        kota = st.selectbox("Kota tujuan", KOTA, index=KOTA.index("Jakarta"))
        jarak = st.number_input("Jarak dari Bandung (km)", 50.0, 3000.0, float(jarak_kota[kota]), step=1.0,
                                help="Otomatis terisi median jarak kota tujuan; ubah jika berbeda.")
    with c2:
        layanan = st.selectbox("Layanan", ["REG", "EXPRESS", "HEMAT"])
        janji = st.number_input("Janji tiba (SLA, hari)", 1, 6, {"EXPRESS": 2, "REG": 3, "HEMAT": 5}[layanan])
        transit = st.number_input("Jumlah hub transit", 1, 5, 2)
        berat = st.number_input("Berat (kg)", 0.05, 50.0, 1.5, step=0.1)
    with c3:
        bayar = st.selectbox("Metode bayar", ["non_COD", "COD"])
        cuaca = st.selectbox("Cuaca saat pickup", ["cerah", "berawan", "hujan", "hujan_lebat"])
        volume = st.number_input("Volume paket masuk hub hari ini", 2000, 20000, 5400, step=100,
                                 help="Beban hub saat ini. Di atas ±8.000 = hari puncak (risiko melonjak).")

    row = pd.DataFrame([{
        "tanggal_pickup": tanggal, "jam_pickup": jam, "kota_tujuan": kota, "jarak_km": jarak,
        "jumlah_transit": transit, "layanan": layanan, "janji_hari": janji, "berat_kg": berat,
        "metode_bayar": bayar, "cuaca_saat_pickup": cuaca, "volume_hub_hari": volume}])

    if st.button("Prediksi", type="primary"):
        p = float(predict_proba(model, row)[0])
        rec = recommend(p, NOTIFY, PRIORITY)
        m1, m2 = st.columns([1, 2])
        m1.metric("Probabilitas terlambat", f"{p:.1%}")
        m2.markdown(f"<div style='padding:14px;border-radius:10px;background:{COLOR[rec]};color:white'>"
                    f"<b style='font-size:1.3rem'>{rec}</b><br>{HINT[rec]}</div>", unsafe_allow_html=True)
        st.progress(min(p, 1.0))
        st.subheader("Faktor yang paling berpengaruh")
        st.caption("Dibandingkan paket rata-rata. Plus = menaikkan risiko terlambat.")
        ex = explain(model, row, top=5)
        for _, r in ex.iterrows():
            arah = "⬆ menaikkan risiko" if r["kontribusi"] > 0 else "⬇ menurunkan risiko"
            st.write(f"- **{r['fitur']}** : {arah} ({r['kontribusi']:+.2f} log-odds)")
        st.caption(f"Ambang kabari = {NOTIFY:.1%} (dari biaya Rp50.000 vs Rp5.000) · "
                   f"ambang prioritas = {PRIORITY:.0%}.")

# ------------------------------------------------------------------ batch
with tab2:
    st.write("Kolom wajib: " + ", ".join(f"`{c}`" for c in REQUIRED) +
             ". Opsional: `jarak_km`, `cuaca_saat_pickup`, `id_resi`.")
    up = st.file_uploader("Unggah CSV", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        missing = [c for c in REQUIRED if c not in raw.columns]
        if missing:
            st.error(f"Kolom belum ada: {missing}")
        else:
            for c in ["jarak_km", "cuaca_saat_pickup"]:
                if c not in raw.columns:
                    raw[c] = pd.NA
            p = predict_proba(model, raw)
            res = raw.copy()
            res["prob_terlambat"] = p.round(4)
            res["rekomendasi"] = [recommend(x, NOTIFY, PRIORITY) for x in p]
            st.write(res["rekomendasi"].value_counts().rename("jumlah paket").to_frame())
            st.dataframe(res.sort_values("prob_terlambat", ascending=False), use_container_width=True)
            st.download_button("Unduh hasil (CSV)", res.to_csv(index=False).encode(), "prediksi.csv", "text/csv")

# ------------------------------------------------------------------ info
with tab3:
    t = meta["metrics"]["test_final_model"]; b = meta["metrics"]["test_baseline_LR_fitur_asli"]
    st.write(f"**Model:** {meta['model']}")
    st.write(f"**Data latih:** {meta['train_data']}  \n**Data uji:** {meta['test_data']}")
    cA, cB, cC = st.columns(3)
    cA.metric("PR-AUC test (model)", t["pr_auc"], f"{t['pr_auc'] - b['pr_auc']:+.3f} vs baseline")
    cB.metric("Biaya per paket (model)", f"Rp{t['cost_per_package_rp']:,}".replace(",", "."))
    cC.metric("Biaya jika tidak pernah mengabari", f"Rp{t['cost_never_notify_rp']:,}".replace(",", "."))
    st.write("**Keterbatasan:**")
    for s in meta["known_limitations"]:
        st.write(f"- {s}")
