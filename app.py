import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
st.set_page_config(page_title="Manufacturing Defect Prediction", page_icon="🏭", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE / "defect_model.joblib")
    meta = json.load(open(BASE / "model_meta.json"))
    return model, meta


model, meta = load_artifacts()
FEATURES = meta["selected_features"]
THR = meta["threshold"]
STATS = meta["feature_stats"]


def risk_band(p):
    if p >= THR:
        return "HIGH RISK"
    if p >= THR * 0.5:
        return "MEDIUM RISK"
    return "LOW RISK"


def set_values(values):
    for f in FEATURES:
        st.session_state[f"in_{f}"] = float(values[f])


# first load: start every input at the typical (median) value
for f in FEATURES:
    st.session_state.setdefault(f"in_{f}", float(STATS[f]["median"]))

st.title("🏭 Manufacturing Defect Prediction")
st.caption("Machine Learning-Based Predictive Quality · SECOM semiconductor dataset · "
           f"Model: **{meta['best_model']}**")

tab1, tab2, tab3 = st.tabs(["🔍 Single prediction", "📄 Batch prediction (CSV)", "ℹ️ About the model"])

# ---------------------------------------------------------------- single
with tab1:
    st.write("Enter the process / sensor readings below, or load an example, then click **Predict**.")
    b1, b2, b3, _ = st.columns([1, 1, 1, 2])
    b1.button("✅ Load good-unit example", on_click=set_values, args=(meta["examples"]["pass"],))
    b2.button("⚠️ Load risky-unit example", on_click=set_values, args=(meta["examples"]["fail"],))
    b3.button("↺ Reset to typical", on_click=set_values,
              args=({f: STATS[f]["median"] for f in FEATURES},))

    cols = st.columns(4)
    for i, f in enumerate(FEATURES):
        s = STATS[f]
        cols[i % 4].number_input(
            f.replace("_", " ").title(), key=f"in_{f}", format="%g",
            step=max(float(s["std"]) / 20, 1e-6),
            help=f"Typical range: {s['p1']:.4g} to {s['p99']:.4g} (median {s['median']:.4g})")

    if st.button("🔍 Predict defect risk", type="primary"):
        x = pd.DataFrame([{f: st.session_state[f"in_{f}"] for f in FEATURES}])[FEATURES]
        risk = float(model.predict_proba(x)[0, 1])
        band = risk_band(risk)

        st.divider()
        c1, c2 = st.columns([1, 2])
        c1.metric("Defect Risk", f"{risk:.0%}")
        c1.progress(min(max(risk, 0.0), 1.0))
        c1.write(f"**{band}**")
        if risk >= THR:
            c2.error("### ❌ Predicted: DEFECTIVE / FAIL\nInspect the process before producing a large batch.")
        else:
            c2.success("### ✅ Predicted: GOOD / PASS\nProcess parameters look consistent with acceptable quality.")

        z = {f: (x.iloc[0][f] - STATS[f]["mean"]) / (STATS[f]["std"] or 1) for f in FEATURES}
        odd = sorted([(f, v) for f, v in z.items() if abs(v) > 2], key=lambda t: -abs(t[1]))
        if odd:
            st.warning("**Parameters far from their typical value (possible causes to check):**\n\n" +
                       "\n".join(f"- `{f}`: {v:+.1f} standard deviations" for f, v in odd[:6]))
        st.caption("The risk score is a model score for ranking/alerting, not a guaranteed probability.")

# ----------------------------------------------------------------- batch
with tab2:
    st.write("Upload a CSV with one row per unit and these columns (blank cells are allowed):")
    st.code(", ".join(FEATURES))
    st.download_button("⬇️ Download sample CSV", data=open(BASE / "sample_batch.csv", "rb").read(),
                       file_name="sample_batch.csv", mime="text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            risk = model.predict_proba(df[FEATURES])[:, 1]
            out = df.copy()
            out["defect_risk_%"] = (risk * 100).round(1)
            out["prediction"] = ["DEFECTIVE" if r >= THR else "GOOD" for r in risk]
            out["risk_level"] = [risk_band(r) for r in risk]
            m1, m2, m3 = st.columns(3)
            m1.metric("Units checked", len(out))
            m2.metric("Predicted defective", int((out["prediction"] == "DEFECTIVE").sum()))
            m3.metric("Average risk", f"{risk.mean():.0%}")
            st.dataframe(out.sort_values("defect_risk_%", ascending=False), use_container_width=True)
            st.download_button("⬇️ Download results", out.to_csv(index=False), "predictions.csv", "text/csv")

# ----------------------------------------------------------------- about
with tab3:
    m = meta["test_metrics"]
    st.subheader("Performance on unseen test data")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Accuracy", f"{m['accuracy']:.1%}")
    k2.metric("Precision", f"{m['precision']:.1%}")
    k3.metric("Recall", f"{m['recall']:.1%}")
    k4.metric("F1-Score", f"{m['f1']:.2f}")
    k5.metric("ROC-AUC", f"{m['roc_auc']:.2f}")
    st.subheader("All four models compared (test set, threshold 0.5)")
    st.dataframe(pd.DataFrame(meta["all_models_test"]).set_index("Model"), use_container_width=True)
    st.subheader("Key parameters used by the model")
    st.dataframe(pd.DataFrame(STATS).T[["min", "p1", "median", "p99", "max"]], use_container_width=True)
    st.info("Trained on the SECOM dataset (1,567 runs, ~6.6% defects). Defects are rare, so Recall and F1 "
            "on the defect class are more meaningful than overall accuracy.")
