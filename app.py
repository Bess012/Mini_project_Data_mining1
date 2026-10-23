"""Streamlit dashboard — football player loan prediction with Explainable AI (SHAP + LIME)."""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_val_predict,
    StratifiedKFold, RepeatedStratifiedKFold,
)
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, f1_score, precision_score,
    recall_score, accuracy_score, confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---- Explainable AI (Chapter 6) ----------------------------------
import shap
from lime.lime_tabular import LimeTabularExplainer

import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")

st.set_page_config(
    page_title="Loan Prediction Dashboard · XAI",
    page_icon="⚽", layout="wide",
)

# ==================================================================
# Helpers
# ==================================================================
def prettify_feature_names(raw_names):
    """Turn 'num__age' → 'age', 'cat__position_Centre-Forward' → 'position=Centre-Forward'."""
    out = []
    for n in raw_names:
        n = n.split("__", 1)[-1]
        if n.startswith("position_"):
            n = "position=" + n[len("position_"):]
        out.append(n)
    return out


# ==================================================================
# Cached data + training
# ==================================================================
@st.cache_data(show_spinner="Loading data…")
def load_data():
    return pd.read_csv("df.csv")


@st.cache_resource(show_spinner="Training models… (first run only)")
def train_models(df):
    X = df.drop(columns=["is_loaned"])
    y = df["is_loaned"].astype(int)

    categorical_features = ["position"]
    numerical_features = [
        "age", "minutes_played", "goals", "assists",
        "market_value", "previous_transfers",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=10, sparse_output=False,
        )),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    models = {
        "Logistic Regression": LogisticRegression(
            C=0.03, penalty="l2", solver="liblinear",
            class_weight="balanced", max_iter=2000,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=20, min_samples_split=5,
            class_weight="balanced", random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=20,
            max_features=0.5, class_weight=None,
            random_state=42, n_jobs=1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=41, weights="distance", p=1),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.01,
            subsample=0.9, colsample_bytree=0.9,
            min_child_weight=20, reg_lambda=15,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=1,
        ),
    }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    trained, rows, X_train_prep, X_test_prep, feat_names = {}, [], None, None, None

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier",   clone(model)),
        ])
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        pipe.fit(X_train, y_train)

        # Cache preprocessed matrices & feature names once
        if X_train_prep is None:
            prep = pipe.named_steps["preprocessor"]
            X_train_prep = prep.transform(X_train)
            X_test_prep  = prep.transform(X_test)
            feat_names   = prettify_feature_names(prep.get_feature_names_out())

        test_prob    = pipe.predict_proba(X_test)[:, 1]
        test_pred_50 = (test_prob >= 0.5).astype(int)

        oof_prob = cross_val_predict(
            clone(pipe), X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            method="predict_proba", n_jobs=-1,
        )[:, 1]
        thresholds = np.arange(0.20, 0.71, 0.01)
        f1s = [f1_score(y_train, (oof_prob >= t).astype(int),
                        zero_division=0) for t in thresholds]
        thr = float(thresholds[int(np.argmax(f1s))])
        test_pred_thr = (test_prob >= thr).astype(int)

        rows.append({
            "Model":         name,
            "CV AUC mean":   cv_scores.mean(),
            "CV AUC std":    cv_scores.std(),
            "Test ROC-AUC":  roc_auc_score(y_test, test_prob),
            "Test PR-AUC":   average_precision_score(y_test, test_prob),
            "F1@0.50":       f1_score(y_test, test_pred_50, zero_division=0),
            "OOF Threshold": thr,
            "F1@thr":        f1_score(y_test, test_pred_thr, zero_division=0),
            "Precision@thr": precision_score(y_test, test_pred_thr, zero_division=0),
            "Recall@thr":    recall_score(y_test, test_pred_thr, zero_division=0),
            "Accuracy@thr":  accuracy_score(y_test, test_pred_thr),
        })
        trained[name] = pipe

    results = (
        pd.DataFrame(rows)
          .sort_values("CV AUC mean", ascending=False)
          .reset_index(drop=True)
    )
    return (trained, results, X_train, X_test, y_train, y_test,
            X_train_prep, X_test_prep, feat_names)


# ==================================================================
# SHAP helpers
# ==================================================================
def compute_shap_explanation(model_name, pipe, X_train_prep, X_sample, feature_names):
    """Return a single-class (positive = 'Loan') SHAP Explanation object."""
    clf = pipe.named_steps["classifier"]

    if model_name in ("Decision Tree", "Random Forest", "XGBoost"):
        explainer = shap.TreeExplainer(clf)
        try:
            expl = explainer(X_sample)
            if hasattr(expl, "values") and expl.values.ndim == 3:
                expl = expl[:, :, 1]
            return expl
        except Exception:
            sv = explainer.shap_values(X_sample)
            if isinstance(sv, list):
                sv = sv[1]
            base = explainer.expected_value
            if isinstance(base, (list, np.ndarray)) and len(base) == 2:
                base = base[1]
            return shap.Explanation(
                values=sv, base_values=base,
                data=X_sample, feature_names=feature_names,
            )

    if model_name == "Logistic Regression":
        # Explicit 100-row background so SHAP doesn't silently subsample
        background = shap.sample(X_train_prep, 100, random_state=42)
        explainer = shap.LinearExplainer(clf, background)
        sv = explainer.shap_values(X_sample)
        return shap.Explanation(
            values=sv, base_values=explainer.expected_value,
            data=X_sample, feature_names=feature_names,
        )

    # KNN → KernelExplainer is slow, cap the sample size
    X_sample = X_sample[: min(15, len(X_sample))]
    background = shap.sample(X_train_prep, 30, random_state=42)
    explainer = shap.KernelExplainer(clf.predict_proba, background)
    sv = explainer.shap_values(X_sample, nsamples=100, silent=True)
    if isinstance(sv, list):
        sv = sv[1]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)) and len(base) == 2:
        base = base[1]
    return shap.Explanation(
        values=sv, base_values=base,
        data=X_sample, feature_names=feature_names,
    )


def shap_single(model_name, pipe, X_train_prep, X_single, feature_names):
    """SHAP explanation for exactly one preprocessed row."""
    full = compute_shap_explanation(
        model_name, pipe, X_train_prep, X_single, feature_names,
    )
    if full.values.ndim > 1:
        return full[0]
    return full


# ------------------------------------------------------------------
# CACHED wrappers — the actual fix for the slow-on-feature-change bug
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_shap_global_cached(model_name, n_shap,
                               _trained, _X_train_prep, _X_test_prep, feat_names):
    """Cache SHAP by (model, n_shap). Prevents recompute on every Streamlit rerun."""
    pipe = _trained[model_name]
    return compute_shap_explanation(
        model_name, pipe, _X_train_prep,
        _X_test_prep[:n_shap], feat_names,
    )


@st.cache_data(show_spinner=False)
def shap_single_cached(model_name, row_bytes,
                       _trained, _X_train_prep, feat_names):
    """Cache single-row SHAP keyed by (model, row bytes)."""
    row_prep = np.frombuffer(row_bytes, dtype=np.float64).reshape(1, -1)
    pipe = _trained[model_name]
    return shap_single(model_name, pipe, _X_train_prep, row_prep, feat_names)


# ==================================================================
# LIME helpers
# ==================================================================
def lime_single(pipe, X_train_prep, X_single_prep, feature_names,
                num_features=10, num_samples=3000):
    explainer = LimeTabularExplainer(
        X_train_prep,
        feature_names=feature_names,
        class_names=["Non-Loan", "Loan"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    return explainer.explain_instance(
        X_single_prep[0],
        pipe.named_steps["classifier"].predict_proba,
        num_features=num_features,
        num_samples=num_samples,
    )


@st.cache_data(show_spinner=False)
def lime_cached(model_name, row_bytes,
                _trained, _X_train_prep, feat_names,
                num_features=10, num_samples=3000):
    """Cache LIME by (model, row bytes)."""
    row_prep = np.frombuffer(row_bytes, dtype=np.float64).reshape(1, -1)
    pipe = _trained[model_name]
    return lime_single(
        pipe, _X_train_prep, row_prep, feat_names,
        num_features=num_features, num_samples=num_samples,
    )


# ==================================================================
# Load + train
# ==================================================================
st.sidebar.title("⚽ Loan Prediction · XAI")
st.sidebar.markdown(
    "Football player loan prediction — model comparison + **SHAP** + **LIME**."
)

df = load_data()
st.sidebar.caption(f"**{len(df):,}** rows · **{df.shape[1]}** columns")
st.sidebar.caption(f"Positive rate: **{df['is_loaned'].mean():.1%}**")

(trained, results, X_train, X_test, y_train, y_test,
 X_train_prep, X_test_prep, feat_names) = train_models(df)

# ==================================================================
# Header
# ==================================================================
st.title("⚽ Football Player Loan Prediction — with Explainable AI")
tab_overview, tab_curves, tab_cm, tab_shap, tab_lime, tab_predict, tab_data = st.tabs([
    "📊 Model Comparison", "📈 ROC & PR Curves", "🔢 Confusion Matrices",
    "🧠 SHAP", "🔬 LIME", "🎯 Live Prediction", "📁 Data",
])

# ==================================================================
# Tab 1 — Overview
# ==================================================================
with tab_overview:
    st.subheader("Metrics table (sorted by CV AUC mean)")
    styled = (
        results.set_index("Model").style
        .background_gradient(cmap="Greens",
            subset=["CV AUC mean", "Test ROC-AUC", "Test PR-AUC"])
        .format({
            "CV AUC mean": "{:.3f}", "CV AUC std": "{:.3f}",
            "Test ROC-AUC": "{:.3f}", "Test PR-AUC": "{:.3f}",
            "F1@0.50": "{:.3f}", "OOF Threshold": "{:.2f}",
            "F1@thr": "{:.3f}", "Precision@thr": "{:.3f}",
            "Recall@thr": "{:.3f}", "Accuracy@thr": "{:.3f}",
        })
    )
    st.dataframe(styled, width='stretch')

    st.markdown("---")
    st.subheader("CV AUC mean ± std  vs  test metrics")

    plot_df = results.set_index("Model")[
        ["CV AUC mean", "Test ROC-AUC", "Test PR-AUC", "F1@0.50"]
    ].rename(columns={"Test ROC-AUC": "Test ROC-AUC",
                      "Test PR-AUC": "Test PR-AUC",
                      "F1@0.50": "F1@0.50"})

    fig, ax = plt.subplots(figsize=(11, 4.5))
    plot_df.plot(kind="bar", ax=ax, width=0.8)
    x = np.arange(len(plot_df))
    ax.errorbar(
        x - 0.24, plot_df["CV AUC mean"].values,
        yerr=results.set_index("Model").loc[plot_df.index, "CV AUC std"].values,
        fmt="none", ecolor="black", capsize=4, linewidth=1.2,
    )
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_xticklabels(plot_df.index, rotation=15, ha="right")
    ax.legend(loc="lower right")
    ax.set_title("Tuned Model Performance")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.info(
        "All five models sit within one standard deviation of the top — "
        "treat them as statistically tied. Logistic Regression is the "
        "recommended pick based on test ROC-AUC and PR-AUC."
    )

# ==================================================================
# Tab 2 — ROC / PR curves
# ==================================================================
with tab_curves:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC curves (test set)")
        fig, ax = plt.subplots(figsize=(6, 5.5))
        entries = []
        for name, pipe in trained.items():
            prob = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, prob)
            entries.append((roc_auc_score(y_test, prob), name, fpr, tpr))
        for auc, name, fpr, tpr in sorted(entries):
            ax.plot(fpr, tpr, label=f"{name} ({auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
        st.pyplot(fig, clear_figure=True)

    with col2:
        st.subheader("Precision-Recall curves (test set)")
        fig, ax = plt.subplots(figsize=(6, 5.5))
        pos_rate = y_test.mean()
        ax.axhline(pos_rate, color="gray", ls="--", lw=0.8,
                   label=f"Random ({pos_rate:.2f})")
        entries = []
        for name, pipe in trained.items():
            prob = pipe.predict_proba(X_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test, prob)
            entries.append((average_precision_score(y_test, prob), name, prec, rec))
        for ap, name, prec, rec in sorted(entries):
            ax.plot(rec, prec, label=f"{name} ({ap:.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=0.3)
        st.pyplot(fig, clear_figure=True)

# ==================================================================
# Tab 3 — Confusion matrices
# ==================================================================
with tab_cm:
    st.subheader("Confusion matrices — test set")
    st.caption(
        "Left: default threshold 0.50. Right: OOF-selected threshold. "
        "Lowering the threshold raises loan-class recall at the cost of overall accuracy."
    )

    selected_model = st.selectbox(
        "Model", list(trained.keys()),
        index=list(trained.keys()).index(results.iloc[0]["Model"]),
        key="cm_model",
    )
    thr = float(results.loc[results["Model"] == selected_model, "OOF Threshold"].iloc[0])

    c1, c2 = st.columns(2)
    prob = trained[selected_model].predict_proba(X_test)[:, 1]

    for col, t, label in zip([c1, c2], [0.50, thr],
                             ["default (headline)", "OOF (reference)"]):
        with col:
            pred = (prob >= t).astype(int)
            cm = confusion_matrix(y_test, pred)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            ConfusionMatrixDisplay(
                cm, display_labels=["Non-Loan", "Loan"],
            ).plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"{selected_model}\n{label} threshold = {t:.2f}")
            st.pyplot(fig, clear_figure=True)

# ==================================================================
# Tab 4 — SHAP  (cached — no more recompute on every interaction)
# ==================================================================
with tab_shap:
    st.subheader("🧠 SHAP — SHapley Additive exPlanations")
    st.markdown(
        "SHAP assigns each feature a contribution to the prediction using "
        "Shapley values from cooperative game theory. Contributions are "
        "**additive**: `f(x) = φ₀ + Σ φⱼ`."
    )

    st.markdown("### Global explanations")
    c1, c2 = st.columns([1, 2])
    with c1:
        shap_model = st.selectbox("Model", list(trained.keys()), key="shap_model")
    with c2:
        n_shap = st.slider(
            "Samples to explain (speed vs. coverage)",
            min_value=20, max_value=len(X_test),
            value=min(100, len(X_test)), step=10,
            key="shap_n",
        )

    if shap_model == "KNN":
        st.warning(
            "⚠️ KNN uses SHAP's **KernelExplainer** (model-agnostic). "
            "It's the slowest explainer — expect 15–30 s on first compute. "
            "Tree-based models (Decision Tree / RF / XGBoost) are ~100× faster. "
            "Results are cached after the first run."
        )

    with st.spinner(f"Computing SHAP values for {shap_model}…"):
        explanation = compute_shap_global_cached(
            shap_model, n_shap,
            trained, X_train_prep, X_test_prep, feat_names,
        )

    gc1, gc2 = st.columns(2)
    with gc1:
        st.markdown("**Feature importance — mean |SHAP| per feature**")
        fig = plt.figure(figsize=(8, 5))
        shap.plots.bar(explanation, max_display=10, show=False)
        st.pyplot(fig, clear_figure=True)

    with gc2:
        st.markdown("**Beeswarm — direction & dispersion of each effect**")
        fig = plt.figure(figsize=(8, 5))
        shap.plots.beeswarm(explanation, max_display=10, show=False)
        st.pyplot(fig, clear_figure=True)

    st.caption(
        "💡 Beeswarm: each dot is a test player. **Horizontal** position = "
        "SHAP value (push toward Loan or Non-Loan). **Color** = the feature "
        "value. Red on the right = high feature values push toward Loan."
    )

    st.markdown("### Local explanation — one player")
    max_idx = explanation.values.shape[0] - 1
    inst = st.slider("Test instance", 0, max_idx, 0, key="shap_inst")

    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation[inst], max_display=10, show=False)
    st.pyplot(fig, clear_figure=True)

    st.markdown(f"**Original features for test row #{inst}:**")
    st.dataframe(X_test.iloc[[inst]], width='stretch')
    p_inst = trained[shap_model].predict_proba(X_test.iloc[[inst]])[0, 1]
    st.metric(f"Predicted P(loan) — {shap_model}", f"{p_inst:.1%}")

    st.caption(
        "💡 Waterfall: starts from φ₀ (average model prediction) and adds each "
        "feature's contribution. Reaching the final bar gives the exact model output."
    )

# ==================================================================
# Tab 5 — LIME  (cached)
# ==================================================================
with tab_lime:
    st.subheader("🔬 LIME — Local Interpretable Model-agnostic Explanations")
    st.markdown(
        "LIME fits a **simple linear model** around a single prediction. "
        "The bars show which features (in original units) push this player "
        "toward Loan vs. Non-Loan."
    )

    lc1, lc2 = st.columns([1, 2])
    with lc1:
        lime_model = st.selectbox("Model", list(trained.keys()), key="lime_model")
    with lc2:
        lime_inst = st.slider(
            "Test instance",
            min_value=0, max_value=len(X_test) - 1, value=0,
            key="lime_inst",
        )

    row_bytes = X_test_prep[lime_inst].astype(np.float64).tobytes()

    with st.spinner("Fitting LIME explanation…"):
        exp = lime_cached(
            lime_model, row_bytes,
            trained, X_train_prep, feat_names,
        )

    st.markdown(f"**Original features — test row #{lime_inst}:**")
    st.dataframe(X_test.iloc[[lime_inst]], width='stretch')

    fig = exp.as_pyplot_figure()
    fig.set_size_inches(10, 5)
    st.pyplot(fig, clear_figure=True)

    p_inst = trained[lime_model].predict_proba(X_test.iloc[[lime_inst]])[0, 1]
    st.metric(f"Predicted P(loan) — {lime_model}", f"{p_inst:.1%}")

    st.markdown("**LIME feature contributions (table form):**")
    contrib_df = pd.DataFrame(
        exp.as_list(), columns=["Feature condition", "Contribution"],
    ).sort_values("Contribution")
    st.dataframe(contrib_df, width='stretch')


# ==================================================================
# Tab 6 — Live Prediction + XAI (cached on button click)
# ==================================================================
with tab_predict:
    st.subheader("Score a single player")
    st.markdown(
        "Enter attributes → predict with every tuned model → "
        "**explain** the prediction with SHAP and LIME."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 15, 45, 24)
        minutes = st.number_input("Minutes played", 0, 4000, 1500)
    with c2:
        goals = st.number_input("Goals", 0, 60, 5)
        assists = st.number_input("Assists", 0, 40, 3)
    with c3:
        market_value = st.number_input(
            "Market value (M€)", 0.0, 200.0, 10.0, step=0.5,
        )
        prev_transfers = st.number_input("Previous transfers", 0, 20, 2)
        position = st.selectbox(
            "Position", sorted(df["position"].dropna().unique().tolist()),
        )

    if st.button("Predict", type="primary"):
        row_df = pd.DataFrame([{
            "age": age, "minutes_played": minutes,
            "goals": goals, "assists": assists,
            "market_value": market_value,
            "previous_transfers": prev_transfers,
            "position": position,
        }])

        rows = []
        for name, pipe in trained.items():
            p = pipe.predict_proba(row_df)[0, 1]
            t = float(results.loc[results["Model"] == name,
                                  "OOF Threshold"].iloc[0])
            rows.append({
                "Model":         name,
                "P(loan)":       p,
                "Pred @0.50":    "Loan" if p >= 0.50 else "No loan",
                "OOF Threshold": t,
                "Pred @OOF":     "Loan" if p >= t else "No loan",
            })

        pred_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(
            pred_df.style.format({"P(loan)": "{:.3f}", "OOF Threshold": "{:.2f}"}),
            width='stretch',
        )

        avg_p = pred_df["P(loan)"].mean()
        st.metric("Average P(loan) across models", f"{avg_p:.1%}")
        if avg_p >= 0.5:
            st.success("Consensus: **loan likely**")
        elif avg_p >= 0.35:
            st.warning("Consensus: **borderline / uncertain**")
        else:
            st.info("Consensus: **loan unlikely**")

        fig, ax = plt.subplots(figsize=(8, 3.5))
        pred_df["P(loan)"].sort_values().plot(
            kind="barh", ax=ax, color="steelblue",
        )
        ax.axvline(0.5, color="red", ls="--", lw=0.8, label="0.50")
        ax.set_xlim(0, 1); ax.set_xlabel("P(loan)")
        ax.legend(); plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        # ----- XAI for this single prediction -----
        st.markdown("---")
        st.markdown("### 🧠 Explain this prediction")
        xai_model = st.selectbox(
            "Choose a model to explain", list(trained.keys()),
            index=list(trained.keys()).index("Logistic Regression"),
            key="live_xai_model",
        )
        pipe = trained[xai_model]

        # Preprocess once, share between SHAP and LIME
        row_prep = pipe.named_steps["preprocessor"].transform(row_df)
        row_bytes = row_prep.astype(np.float64).tobytes()

        xc1, xc2 = st.columns(2)

        with xc1:
            st.markdown(f"**SHAP — waterfall for {xai_model}**")
            with st.spinner("Computing SHAP…"):
                expl = shap_single_cached(
                    xai_model, row_bytes,
                    trained, X_train_prep, feat_names,
                )
            fig = plt.figure(figsize=(9, 5))
            shap.plots.waterfall(expl, max_display=10, show=False)
            st.pyplot(fig, clear_figure=True)

        with xc2:
            st.markdown(f"**LIME — local explanation for {xai_model}**")
            with st.spinner("Fitting LIME…"):
                exp = lime_cached(
                    xai_model, row_bytes,
                    trained, X_train_prep, feat_names,
                    num_features=10, num_samples=2000,
                )
            fig = exp.as_pyplot_figure()
            fig.set_size_inches(9, 5)
            st.pyplot(fig, clear_figure=True)

# ==================================================================
# Tab 7 — Data
# ==================================================================
with tab_data:
    st.subheader("Data preview")
    st.dataframe(df.head(50), width='stretch')

    st.subheader("Summary statistics")
    st.dataframe(df.describe(include="all").T, width='stretch')

    st.subheader("Target balance")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    df["is_loaned"].value_counts().plot(
        kind="bar", ax=ax, color=["#4c72b0", "#dd8452"],
    )
    ax.set_xticklabels(["Not loaned (0)", "Loaned (1)"], rotation=0)
    ax.set_ylabel("Count")
    st.pyplot(fig, clear_figure=True)
