"""Streamlit dashboard — football player loan prediction (lean build)."""

import gc
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")               # <-- non-interactive backend, lower memory
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_val_predict,
    StratifiedKFold,
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

sns.set_theme(style="whitegrid")
st.set_page_config(page_title="Loan Prediction Dashboard",
                   page_icon="⚽", layout="wide")


@st.cache_data(show_spinner="Loading data…")
def load_data():
    return pd.read_csv("df.csv")


@st.cache_resource(show_spinner="Training models… (first run only, ~30 s)")
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

    # ------------------------------------------------------------------
    # LIGHTER MODELS — halved tree counts, n_jobs=1 everywhere
    # ------------------------------------------------------------------
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
            n_estimators=50, max_depth=5, min_samples_leaf=20,   # 50, not 100
            max_features=0.5, class_weight=None,
            random_state=42, n_jobs=1,                            # was 1, keep
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=41, weights="distance", p=1, n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.01,    # 100, not 200
            subsample=0.9, colsample_bytree=0.9,
            min_child_weight=20, reg_lambda=15,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=1, tree_method="hist",         # hist = lower memory
        ),
    }

    # ------------------------------------------------------------------
    # SINGLE 5-FOLD CV (not 5×2 repeated) — halves memory
    # ------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    trained, rows = {}, []
    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier",   clone(model)),
        ])

        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=cv,
            scoring="roc_auc", n_jobs=1,          # was -1, now 1 → less peak RAM
        )
        pipe.fit(X_train, y_train)

        test_prob    = pipe.predict_proba(X_test)[:, 1]
        test_pred_50 = (test_prob >= 0.5).astype(int)

        oof_prob = cross_val_predict(
            clone(pipe), X_train, y_train, cv=cv,
            method="predict_proba", n_jobs=1,
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
        gc.collect()                                # free intermediate buffers

    results = (
        pd.DataFrame(rows)
          .sort_values("CV AUC mean", ascending=False)
          .reset_index(drop=True)
    )

    # Free big things we no longer need
    del X, X_train
    gc.collect()

    return trained, results, X_test, y_train, y_test


# ======================================================================
# Boot
# ======================================================================
df = load_data()
st.sidebar.title("⚽ Loan Prediction Dashboard")
st.sidebar.markdown("Football player loan prediction — comparison & live scoring.")
st.sidebar.caption(f"**{len(df):,}** rows · **{df.shape[1]}** columns")
st.sidebar.caption(f"Positive rate: **{df['is_loaned'].mean():.1%}**")

trained, results, X_test, y_train, y_test = train_models(df)

st.title("⚽ Football Player Loan Prediction")
st.markdown(
    "Model comparison on stratified CV ROC-AUC. All five models are "
    "statistically tied within one standard deviation — **Logistic "
    "Regression** is preferred for test-set ROC-AUC and PR-AUC."
)

t1, t2, t3, t4, t5 = st.tabs([
    "📊 Model Comparison", "📈 ROC & PR Curves",
    "🔢 Confusion Matrices", "🎯 Live Prediction", "📁 Data",
])


def _close(fig):
    """Render then release the matplotlib figure."""
    st.pyplot(fig)
    plt.close(fig)
    gc.collect()


# ----------------------------------------------------------------------
# Tab 1
# ----------------------------------------------------------------------
with t1:
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
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")
    st.subheader("CV AUC mean ± std  vs  test metrics")

    plot_df = results.set_index("Model")[
        ["CV AUC mean", "Test ROC-AUC", "Test PR-AUC", "F1@0.50"]
    ]
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
    _close(fig)

    st.info(
        "All five models sit within one standard deviation of the top — "
        "treat them as statistically tied. Logistic Regression is the "
        "recommended pick based on test ROC-AUC and PR-AUC."
    )

# ----------------------------------------------------------------------
# Tab 2
# ----------------------------------------------------------------------
with t2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROC curves (test set)")
        fig, ax = plt.subplots(figsize=(6, 5))
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
        plt.tight_layout(); _close(fig)

    with c2:
        st.subheader("Precision-Recall curves (test set)")
        fig, ax = plt.subplots(figsize=(6, 5))
        pos_rate = y_test.mean()
        ax.axhline(pos_rate, color="gray", ls="--", lw=0.8,
                   label=f"Random ({pos_rate:.2f})")
        entries = []
        for name, pipe in trained.items():
            prob = pipe.predict_proba(X_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test, prob)
            entries.append((average_precision_score(y_test, prob),
                            name, prec, rec))
        for ap, name, prec, rec in sorted(entries):
            ax.plot(rec, prec, label=f"{name} ({ap:.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); _close(fig)

# ----------------------------------------------------------------------
# Tab 3
# ----------------------------------------------------------------------
with t3:
    st.subheader("Confusion matrices — test set")
    st.caption(
        "Left: default threshold 0.50 (what we deploy). "
        "Right: OOF-selected threshold, shown for transparency."
    )
    selected = st.selectbox(
        "Model", list(trained.keys()),
        index=list(trained.keys()).index(results.iloc[0]["Model"]),
    )
    thr = float(results.loc[results["Model"] == selected, "OOF Threshold"].iloc[0])

    c1, c2 = st.columns(2)
    prob = trained[selected].predict_proba(X_test)[:, 1]

    for col, t, label in zip(
        [c1, c2], [0.50, thr], ["default (headline)", "OOF (reference)"]
    ):
        with col:
            pred = (prob >= t).astype(int)
            cm = confusion_matrix(y_test, pred)
            fig, ax = plt.subplots(figsize=(4.2, 4))
            ConfusionMatrixDisplay(
                cm, display_labels=["Non-Loan", "Loan"],
            ).plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"{selected}\n{label} threshold = {t:.2f}")
            plt.tight_layout(); _close(fig)

# ----------------------------------------------------------------------
# Tab 4
# ----------------------------------------------------------------------
with t4:
    st.subheader("Score a single player")
    st.caption(
        "Runs every model and shows the predicted loan probability at "
        "0.50 and at each model's OOF-selected threshold."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 15, 45, 24)
        minutes = st.number_input("Minutes played", 0, 4000, 1500)
    with c2:
        goals = st.number_input("Goals", 0, 60, 5)
        assists = st.number_input("Assists", 0, 40, 3)
    with c3:
        market_value = st.number_input("Market value (M€)", 0.0, 200.0, 10.0, step=0.5)
        prev_transfers = st.number_input("Previous transfers", 0, 20, 2)
        position = st.selectbox(
            "Position", sorted(df["position"].dropna().unique().tolist()),
        )

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{
            "age": age, "minutes_played": minutes,
            "goals": goals, "assists": assists,
            "market_value": market_value,
            "previous_transfers": prev_transfers,
            "position": position,
        }])
        rows = []
        for name, pipe in trained.items():
            p = pipe.predict_proba(row)[0, 1]
            t = float(results.loc[results["Model"] == name, "OOF Threshold"].iloc[0])
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
            use_container_width=True,
        )
        avg = pred_df["P(loan)"].mean()
        st.metric("Average P(loan) across models", f"{avg:.1%}")
        if avg >= 0.5:
            st.success("Consensus: **loan likely**")
        elif avg >= 0.35:
            st.warning("Consensus: **borderline / uncertain**")
        else:
            st.info("Consensus: **loan unlikely**")

        fig, ax = plt.subplots(figsize=(8, 3))
        pred_df["P(loan)"].sort_values().plot(
            kind="barh", ax=ax, color="steelblue",
        )
        ax.axvline(0.5, color="red", ls="--", lw=0.8, label="0.50")
        ax.set_xlim(0, 1); ax.set_xlabel("P(loan)")
        ax.legend(); plt.tight_layout(); _close(fig)

# ----------------------------------------------------------------------
# Tab 5
# ----------------------------------------------------------------------
with t5:
    st.subheader("Data preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Summary statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.subheader("Target balance")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    df["is_loaned"].value_counts().plot(
        kind="bar", ax=ax, color=["#4c72b0", "#dd8452"],
    )
    ax.set_xticklabels(["Not loaned (0)", "Loaned (1)"], rotation=0)
    ax.set_ylabel("Count")
    plt.tight_layout(); _close(fig)
