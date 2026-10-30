import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
st.set_page_config(page_title="Defect Predictor", page_icon="🏭", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE / "defect_model.joblib")
    meta = json.load(open(BASE / "model_meta.json"))
    return model, meta


model, meta = load_artifacts()
FEATURES = meta["selected_features"]      # already ordered: most important first
THR = float(meta["threshold"])
STATS = meta["feature_stats"]
EX = meta["examples"]
N_KEY = 5
KEY, OTHER = FEATURES[:N_KEY], FEATURES[N_KEY:]


def nice(f):
    return "Sensor " + f.split("_")[-1]


def slider_range(f):
    s = STATS[f]
    vals = [s["p1"], s["p99"], s["median"], EX["pass"][f], EX["fail"][f]]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def band(risk):
    if risk >= THR:
        return "HIGH RISK", "#dc2626", "❌", "Defect likely"
    if risk >= THR * 0.5:
        return "MEDIUM RISK", "#f59e0b", "⚠️", "Borderline – keep an eye on it"
    return "LOW RISK", "#16a34a", "✅", "Looks good"


ADVICE = {
    "HIGH RISK": "**Recommended action:** inspect this process before continuing the batch. "
                 "Start with the readings listed below – they differ most from normal production.",
    "MEDIUM RISK": "**Recommended action:** the readings are in a grey zone. Consider a spot check "
                   "and keep monitoring the next units.",
    "LOW RISK": "**Recommended action:** none needed. These readings look like normal, good production.",
}

PRESETS = {
    "Typical unit": {f: STATS[f]["median"] for f in FEATURES},
    "Good unit example": EX["pass"],
    "Risky unit example": EX["fail"],
}


def apply_preset():
    vals = PRESETS[st.session_state["preset"]]
    for f in FEATURES:
        st.session_state[f"in_{f}"] = float(vals[f])


if "preset" not in st.session_state:
    st.session_state["preset"] = "Typical unit"
    apply_preset()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("How to use")
    st.markdown("**1.** Pick a starting point  \n**2.** Move the sliders  \n**3.** The result updates instantly")
    st.divider()
    m = meta["test_metrics"]
    st.caption("How well the model did on units it had never seen")
    st.metric("Defects caught", f"{m['recall']:.0%}", help="Recall: share of real defects the model flagged")
    st.metric("Alerts that were real", f"{m['precision']:.0%}", help="Precision: share of flagged units that really were defective")
    st.caption(f"Model: {meta['best_model']} · SECOM semiconductor data")

# ------------------------------------------------------------------- header
st.title("🏭 Manufacturing Defect Predictor")
st.markdown("Predict whether a production unit is likely to come out **defective** – *before* final inspection – "
            "from its process and sensor readings.")

tab1, tab2, tab3 = st.tabs(["🔍 Check one unit", "📄 Check many units", "ℹ️ About"])

# --------------------------------------------------------------- single unit
with tab1:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("1 · Choose a starting point")
        st.radio("Starting point", list(PRESETS), key="preset", horizontal=True,
                 on_change=apply_preset, label_visibility="collapsed")

        st.subheader("2 · Adjust the key readings")
        st.caption(f"These {N_KEY} sensors influence defects the most. Sensor names are anonymised in the dataset. "
                   "Hover over the ? for the normal range.")
        for f in KEY:
            lo, hi = slider_range(f)
            st.slider(nice(f), min_value=lo, max_value=hi, step=(hi - lo) / 200, key=f"in_{f}",
                      format="%g", help=f"Normal range: {STATS[f]['p1']:.4g} to {STATS[f]['p99']:.4g} "
                                        f"(typical {STATS[f]['median']:.4g})")

        with st.expander(f"Fine-tune the other {len(OTHER)} readings (optional)"):
            cols = st.columns(3)
            for i, f in enumerate(OTHER):
                cols[i % 3].number_input(
                    nice(f), key=f"in_{f}", format="%g", step=max(float(STATS[f]["std"]) / 20, 1e-6),
                    help=f"Normal range: {STATS[f]['p1']:.4g} to {STATS[f]['p99']:.4g}")

    # live prediction
    x = pd.DataFrame([{f: st.session_state[f"in_{f}"] for f in FEATURES}])[FEATURES]
    risk = float(model.predict_proba(x)[0, 1])
    level, color, icon, headline = band(risk)

    with right:
        st.subheader("3 · Result")
        st.markdown(
            f'<div style="border-left:8px solid {color};background:{color}22;padding:18px 22px;border-radius:10px;">'
            f'<div style="font-size:15px;opacity:.75;">Predicted outcome</div>'
            f'<div style="font-size:28px;font-weight:700;">{icon} {headline}</div>'
            f'<div style="font-size:56px;font-weight:800;color:{color};line-height:1.15;">{risk:.0%}</div>'
            f'<div style="font-size:15px;opacity:.75;">defect risk · {level}</div></div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div style="margin:16px 0 4px;position:relative;height:14px;background:#9ca3af55;border-radius:7px;">'
            f'<div style="width:{min(risk, 1) * 100:.1f}%;height:100%;background:{color};border-radius:7px;"></div>'
            f'<div style="position:absolute;left:{THR * 100:.1f}%;top:-5px;height:24px;border-left:3px solid #6366f1;"></div>'
            f'</div>'
            f'<div style="font-size:12px;opacity:.75;">The purple line is the alert level ({THR:.0%}). '
            f'Above it the unit is flagged as likely defective.</div>',
            unsafe_allow_html=True)
        st.write("")
        st.markdown(ADVICE[level])

        z = {f: (x.iloc[0][f] - STATS[f]["mean"]) / (STATS[f]["std"] or 1) for f in FEATURES}
        odd = sorted([(f, v) for f, v in z.items() if abs(v) > 2], key=lambda t: -abs(t[1]))
        if odd:
            lines = [f"- **{nice(f)}** is unusually {'high' if v > 0 else 'low'} ({abs(v):.1f}× the normal spread)"
                     for f, v in odd[:5]]
            st.info("**Readings to check first**\n\n" + "\n".join(lines))
        st.caption("The risk % is a model score for flagging and ranking units, not an exact probability.")

# ----------------------------------------------------------------- many units
with tab2:
    st.subheader("Check a whole batch at once")
    st.markdown("**1.** Download the sample file to see the format  \n"
                "**2.** Upload your own CSV (one row per unit)  \n"
                "**3.** Review the ranked results and download them")
    st.download_button("⬇️ Download sample CSV", data=open(BASE / "sample_batch.csv", "rb").read(),
                       file_name="sample_batch.csv", mime="text/csv")
    with st.expander("Required columns"):
        st.code(", ".join(FEATURES))
        st.caption("Empty cells are allowed – they are filled with typical values.")

    up = st.file_uploader("Upload your CSV file", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception:
            df = None
            st.error("That file could not be read. Please upload a plain CSV file.")
        if df is not None:
            missing = [f for f in FEATURES if f not in df.columns]
            if missing:
                st.error("Your file is missing these columns: " + ", ".join(missing))
            else:
                data = df[FEATURES].apply(pd.to_numeric, errors="coerce")
                r = model.predict_proba(data)[:, 1]
                view = pd.DataFrame({
                    "Unit": range(1, len(r) + 1),
                    "Defect risk (%)": (r * 100).round(1),
                    "Verdict": ["Defective" if v >= THR else "Good" for v in r],
                    "Risk level": [band(v)[0] for v in r],
                })
                n_bad = int((r >= THR).sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Units checked", len(r))
                c2.metric("Flagged as defective", n_bad)
                c3.metric("Share flagged", f"{n_bad / len(r):.0%}")
                st.caption("Sorted with the riskiest units first.")
                st.dataframe(
                    view.sort_values("Defect risk (%)", ascending=False), hide_index=True,
                    column_config={"Defect risk (%)": st.column_config.ProgressColumn(
                        "Defect risk (%)", min_value=0, max_value=100, format="%.0f")})
                full = pd.concat([view, df.reset_index(drop=True)], axis=1)
                st.download_button("⬇️ Download full results", full.to_csv(index=False),
                                   "defect_predictions.csv", "text/csv")

# --------------------------------------------------------------------- about
with tab3:
    st.subheader("What this app does")
    st.markdown(
        "Factories usually find defects only at final inspection, after material and time are already spent. "
        "This app uses machine learning on real semiconductor production data (the **SECOM** dataset) to "
        "flag risky units **early**, so corrective action can be taken before a large batch is wasted.")

    st.subheader("How good is it?")
    m = meta["test_metrics"]
    k = st.columns(4)
    k[0].metric("Defects caught", f"{m['recall']:.0%}", help="Recall")
    k[1].metric("Alerts that were real", f"{m['precision']:.0%}", help="Precision")
    k[2].metric("F1-score", f"{m['f1']:.2f}", help="Balance of the two above (1.0 is perfect)")
    k[3].metric("ROC-AUC", f"{m['roc_auc']:.2f}", help="How well the model ranks defective above good units")
    st.info("Only about 1 in 15 units in this dataset is defective, so overall accuracy looks high for any model. "
            "That is why we judge it on how many real defects it catches and how trustworthy its alerts are.")

    st.subheader("The four models we compared")
    comp = pd.DataFrame(meta["all_models_test"]).set_index("Model")
    st.dataframe(comp)
    st.caption(f"Best model (chosen with cross-validation): **{meta['best_model']}**")

    with st.expander("Technical details"):
        st.write("Top 20 sensors were selected with a Random Forest. Missing values are filled with the median, "
                 "data is scaled, and SMOTE balanced the defect class during training.")
        st.dataframe(pd.DataFrame(STATS).T[["min", "p1", "median", "p99", "max"]])
