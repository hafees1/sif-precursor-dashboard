"""
SIF Precursor Detection Dashboard — OIL India Ltd. (SIH 2026, PS 26165)

Run with:  streamlit run dashboard_app.py

Reads the artifacts exported by sif_precursor_engine.ipynb (in ./artifacts/).
Run the notebook at least once before launching this app.
"""
import ast
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ARTIFACT_DIR = "artifacts"

st.set_page_config(page_title="OIL SIF Precursor Dashboard", layout="wide",
                    page_icon="⚠️")


# ── Data & model loading ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(ARTIFACT_DIR, "processed_reports.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["hazard_categories"] = df["hazard_categories"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else [])
    return df


@st.cache_data
def load_rankings():
    tables = {}
    for name in ["site_ranking", "activity_ranking", "location_ranking",
                 "barrier_ranking", "precursor_signatures"]:
        path = os.path.join(ARTIFACT_DIR, f"{name}.csv")
        tables[name] = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    return tables


@st.cache_resource
def load_models():
    models = {}
    for name in ["sif_classifier", "tfidf_vectorizer", "lsr_classifier", "lsr_tfidf_vectorizer"]:
        path = os.path.join(ARTIFACT_DIR, f"{name}.joblib")
        models[name] = joblib.load(path) if os.path.exists(path) else None
    return models


import re
from scipy.sparse import hstack, csr_matrix

SIF_HAZARD_LEXICON = {
    "energy_isolation": ["lock out", "lockout", "tag out", "tagout", "loto",
                          "stored energy", "residual pressure", "incomplete isolation",
                          "isolation verification", "unexpected energization", "de-energiz",
                          "isolat", "did not isolate", "failed to isolate"],
    "line_of_fire": ["line of fire", "struck by", "caught in", "caught between",
                      "load path", "suspended load", "dropped object", "unsafe positioning",
                      "pinch point", "swinging load", "over the top of", "below a suspended",
                      "workers standing below", "overhead load"],
    "gravity_height": ["fall from height", "working at height", "unprotected edge",
                        "improper ladder", "fall protection", "roof maintenance",
                        "scaffold", "elevated maintenance", "guardrail", "loose railing",
                        "platform edge"],
    "mechanical_lifting": ["crane", "lifting", "sling", "rigging", "hoist",
                            "overloaded crane", "lifting plan", "outrigger", "worn out sling",
                            "worn sling"],
    "confined_space": ["confined space", "manhole", "vessel entry", "entry permit",
                        "atmospheric testing", "engulfment", "inadequate ventilation",
                        "tank cleaning", "pit cleaning", "gas test", "without a permit",
                        "permit was not"],
    "hot_work_fire": ["hot work", "welding", "grinding", "spark", "flammable material",
                       "fire watch", "gas testing", "explosion", "ignition source"],
    "electrical": ["electrical", "live wire", "exposed conductor", "grounding",
                    "arc flash", "high voltage", "energized circuit"],
    "vehicle_driving": ["seat belt", "driver distraction", "vehicle condition",
                         "speeding", "fatigue", "journey management", "rollover"],
    "excavation": ["excavation", "trench", "underground utility", "cave-in", "shoring"],
    "pressure_process": ["pressure testing", "over-pressure", "valve", "relief valve",
                          "process upset", "leak", "gasket", "blew out"],
}
SIF_SEVERITY_PHRASES_HIGH = [
    "potential for serious injury or fatality", "high-consequence event",
    "immediate intervention was required to prevent a serious incident",
    "potentially serious hazard", "potentially exposed to a high-consequence",
    "serious injury or fatality potential", "before an injury occurred",
    "could have", "easily could have", "nearly", "narrowly avoided", "lucky",
    "close call", "was not hurt", "nobody hurt but", "no one was injured but",
]
SIF_SEVERITY_PHRASES_LOW = [
    "corrected without significant safety exposure", "no high-consequence hazard",
    "addressed as part of normal housekeeping", "low-severity observation",
    "no significant exposure to serious injury", "cleaned immediately",
    "cleared up", "fixed same day", "resolved immediately", "routine top-up",
    "minor spill",
]
HAZARD_TO_LSR = {
    "energy_isolation": "Energy Isolation", "electrical": "Energy Isolation",
    "line_of_fire": "Line of Fire", "gravity_height": "Working at Height",
    "mechanical_lifting": "Safe Mechanical Lifting", "confined_space": "Confined Space",
    "hot_work_fire": "Hot Work", "vehicle_driving": "Driving",
    "excavation": "Work Authorisation", "pressure_process": "Energy Isolation",
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rule_based_score(text: str):
    t = text.lower()
    matched = [cat for cat, kws in SIF_HAZARD_LEXICON.items() if any(kw in t for kw in kws)]
    hazard_score = len(matched)
    severity_score = (sum(1 for p in SIF_SEVERITY_PHRASES_HIGH if p in t)
                       - sum(1 for p in SIF_SEVERITY_PHRASES_LOW if p in t))
    return hazard_score, severity_score, matched


def score_report(text, models):
    clean = clean_text(text)
    hz, sev, cats = rule_based_score(text)
    tfidf, sif_clf = models["tfidf_vectorizer"], models["sif_classifier"]
    lsr_tfidf, lsr_clf = models["lsr_tfidf_vectorizer"], models["lsr_classifier"]

    vec = tfidf.transform([clean])
    hybrid = hstack([vec, csr_matrix(np.array([[hz, sev]]))])
    proba = sif_clf.predict_proba(hybrid)[0, 1]

    lsr_pred = lsr_clf.predict(lsr_tfidf.transform([clean]))[0]
    keyword_lsr = HAZARD_TO_LSR.get(cats[0], "Unclassified") if cats else "Unclassified"

    return {
        "sif_probability": float(proba),
        "sif_potential": bool(proba >= 0.5),
        "hazard_categories": cats,
        "lsr_ml": lsr_pred,
        "lsr_keyword": keyword_lsr,
    }


# ── App ──────────────────────────────────────────────────────────────────
st.title("⚠️ SIF Precursor Detection Dashboard")
st.caption("OIL India Ltd. — HSSE Unsafe-Act / Unsafe-Condition & Near-Miss Reports "
           "| SIH 2026 · Problem Statement 26165")

df = load_data()
if df is None:
    st.error("No processed data found. Run `sif_precursor_engine.ipynb` first to "
             "generate `artifacts/processed_reports.csv`.")
    st.stop()

rankings = load_rankings()
models = load_models()

# Sidebar filters
st.sidebar.header("Filters")
sites = st.sidebar.multiselect("Site", sorted(df["site"].unique()))
report_types = st.sidebar.multiselect("Report Type", sorted(df["report_type"].unique()))
lsrs = st.sidebar.multiselect("Life-Saving Rule", sorted(df["final_lsr_tag"].dropna().unique()))

filtered = df.copy()
if sites:
    filtered = filtered[filtered["site"].isin(sites)]
if report_types:
    filtered = filtered[filtered["report_type"].isin(report_types)]
if lsrs:
    filtered = filtered[filtered["final_lsr_tag"].isin(lsrs)]

# ── KPI row ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Reports", f"{len(filtered):,}")
sif_count = int(filtered["predicted_sif"].sum())
c2.metric("SIF-Potential Flagged", f"{sif_count:,}",
          f"{sif_count / max(len(filtered), 1):.1%} of reports")
c3.metric("Sites Covered", filtered["site"].nunique())
c4.metric("Life-Saving Rules Triggered", filtered["final_lsr_tag"].nunique())

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Precursor Density Rankings", "🛡️ Life-Saving Rule Mapping",
     "🔁 Recurring Precursor Patterns", "🔎 Score a New Report"])

# ── Tab 1: Rankings ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Rank sites & activities by SIF-precursor density")
    st.caption("Density = share of reports at that site/activity/location flagged as "
               "SIF-potential. High density = focus HSE intervention here first, "
               "regardless of total report volume.")

    colA, colB = st.columns(2)
    with colA:
        site_df = filtered.groupby("site").agg(
            total=("predicted_sif", "count"), sif=("predicted_sif", "sum")).reset_index()
        site_df["density"] = site_df["sif"] / site_df["total"]
        site_df = site_df.sort_values("density", ascending=False)
        fig = px.bar(site_df, x="density", y="site", orientation="h",
                      color="density", color_continuous_scale="Reds",
                      title="SIF-Precursor Density by Site", labels={"density": "SIF density"})
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        act_df = filtered.groupby("activity").agg(
            total=("predicted_sif", "count"), sif=("predicted_sif", "sum")).reset_index()
        act_df["density"] = act_df["sif"] / act_df["total"]
        act_df = act_df[act_df["total"] >= 3].sort_values("density", ascending=False).head(12)
        fig = px.bar(act_df, x="density", y="activity", orientation="h",
                      color="density", color_continuous_scale="Oranges",
                      title="SIF-Precursor Density by Activity (top 12)",
                      labels={"density": "SIF density"})
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    loc_df = filtered.groupby("location").agg(
        total=("predicted_sif", "count"), sif=("predicted_sif", "sum")).reset_index()
    loc_df["density"] = loc_df["sif"] / loc_df["total"]
    loc_df = loc_df[loc_df["total"] >= 3].sort_values("density", ascending=False)
    st.markdown("**Location ranking (full table)**")
    st.dataframe(loc_df.style.format({"density": "{:.1%}"}), use_container_width=True)

# ── Tab 2: LSR mapping ───────────────────────────────────────────────────
with tab2:
    st.subheader("IOGP Life-Saving Rule Mapping")
    lsr_counts = filtered["final_lsr_tag"].value_counts().reset_index()
    lsr_counts.columns = ["life_saving_rule", "count"]
    fig = px.pie(lsr_counts, names="life_saving_rule", values="count", hole=0.4,
                  title="Report Volume by Life-Saving Rule")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**SIF density by Site × Life-Saving Rule** — darker = more fatal-potential "
                "reports of that rule type at that site.")
    heat = pd.crosstab(filtered["site"], filtered["final_lsr_tag"],
                        values=filtered["predicted_sif"], aggfunc="mean").fillna(0)
    fig2 = px.imshow(heat, color_continuous_scale="Reds", aspect="auto",
                       labels=dict(color="SIF density"))
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Recurring patterns ────────────────────────────────────────────
with tab3:
    st.subheader("Recurring Precursor Patterns")
    st.caption("Activity × barrier-failure combinations with elevated SIF density — "
               "these are the specific patterns to brief supervisors on.")
    st.dataframe(rankings["precursor_signatures"], use_container_width=True)

    st.markdown("**Barrier failures most associated with SIF-potential reports**")
    barrier_df = filtered.groupby("barrier_failure").agg(
        total=("predicted_sif", "count"), sif=("predicted_sif", "sum")).reset_index()
    barrier_df["density"] = barrier_df["sif"] / barrier_df["total"]
    barrier_df = barrier_df[barrier_df["total"] >= 3].sort_values("density", ascending=False).head(12)
    fig3 = px.bar(barrier_df, x="density", y="barrier_failure", orientation="h",
                   color="density", color_continuous_scale="Purples",
                   labels={"density": "SIF density"})
    fig3.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Drill down into flagged reports**")
    st.dataframe(
        filtered[filtered["predicted_sif"] == 1][
            ["report_id", "site", "activity", "barrier_failure", "final_lsr_tag",
             "predicted_sif_proba", "report_text"]
        ].sort_values("predicted_sif_proba", ascending=False),
        use_container_width=True, height=350)

# ── Tab 4: Live scoring ──────────────────────────────────────────────────
with tab4:
    st.subheader("Score a new free-text report")
    st.caption("Paste in a new UA/UC observation, near-miss, or incident report exactly "
               "as a field supervisor might type it.")
    text_input = st.text_area("Report text", height=120,
        placeholder="e.g. Contractor entered the tank without gas testing, permit was not signed...")
    if st.button("Score Report", type="primary") and text_input.strip():
        if any(m is None for m in models.values()):
            st.error("Model artifacts not found. Run the notebook first to train and export models.")
        else:
            result = score_report(text_input, models)
            colX, colY, colZ = st.columns(3)
            colX.metric("SIF-Potential?", "YES ⚠️" if result["sif_potential"] else "No")
            colY.metric("SIF Probability", f"{result['sif_probability']:.1%}")
            colZ.metric("Life-Saving Rule (ML)", result["lsr_ml"])
            st.write("**Matched hazard categories (rule engine):**",
                      ", ".join(result["hazard_categories"]) or "none")
            st.write("**Keyword-based Life-Saving Rule tag:**", result["lsr_keyword"])

st.markdown("---")
st.caption("Prototype built for SIH 2026 · PS 26165 · Oil India Limited. "
           "Trained on synthetic data — retrain on real OIL HSSE exports before production use.")
