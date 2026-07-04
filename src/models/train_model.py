"""
Modeling stage: trains an order-level bad_review risk classifier on top of
the leakage-safe feature table (see src/features/build_features.py), and
produces a secondary, non-point-in-time seller risk table for Ops.

Run:
    python -m src.models.train_model

Outputs (under output/model/):
    metrics.md                     model comparison, precision/recall@k,
                                     calibration -- numbers only, no
                                     conclusions (see docs/ once written)
    feature_importance.png
    permutation_importance.csv      permutation importance (test set, shipped model)
    lr_coefficients.csv             logistic regression standardized coefficients
    calibration.png                 reliability diagram (shipped model)
    test_predictions.csv            order-level scored test set
    seller_risk_table.csv           current (non point-in-time) seller risk ranking
    model.joblib                    fitted pipeline of the shipped model (HGB by
                                     default; random_forest if it clears the
                                     decision rule -- see build_pipelines())

Evaluated primarily on RANKING quality (ROC-AUC / PR-AUC / precision@k), per
docs/problem_framing.md section 3 -- this is a ~14.7%-positive-rate
classification problem meant to prioritize a limited pool of Ops
interventions, not a balanced-accuracy problem.
"""

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data.loader import RAW_DIR, load_raw, build_order_level_table, build_seller_level_table
from src.features.build_features import build_feature_table, ALLOWED_FEATURE_COLUMNS, K_SELLER

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "model"

CATEGORICAL_COLS = ["primary_category", "primary_payment_type", "customer_state", "purchase_season"]
NUMERIC_COLS = [c for c in ALLOWED_FEATURE_COLUMNS if c not in CATEGORICAL_COLS]

K_TOP_FRACTIONS = [0.01, 0.05, 0.10, 0.20]
RANDOM_STATE = 42

# Decision rule for whether random_forest replaces HGB as the shipped model
# (see docs/modeling_findings.md "Random forest cross-check"): RF only ships
# if it beats HGB's PR-AUC by >= RF_PR_AUC_MARGIN (the smallest real jump
# already seen between rungs: LR-over-seller_heuristic was +0.016), holds
# recall@20%, and keeps brier_score <= RF_MAX_BRIER -- guarding against the
# same class-weight-driven calibration collapse that ruled out
# logistic_regression despite its competitive ranking metrics.
DEFAULT_SHIP = "hist_gradient_boosting"
RF_PR_AUC_MARGIN = 0.015
RF_MAX_BRIER = 0.105


def build_pipelines() -> dict:
    """Fitted-model candidates, all consuming the same raw feature columns
    via their own ColumnTransformer. HGB gets native categorical handling
    (ordinal-encoded, negative unknown/missing sentinels never actually
    occur since these are fixed, dataset-wide category vocabularies with no
    missing categoricals in the feature table). LR gets the standard
    one-hot + impute + scale treatment since it can't handle either
    categoricals or NaN natively. random_forest reuses HGB's ordinal
    encoding (trees don't need one-hot) plus a median imputer, since unlike
    HGB it can't accept NaN natively (customer_seller_distance_km has real
    missingness -- see src/features/build_features.py). ExtraTrees would be
    a one-line variant of the same pipeline but isn't added here -- a second
    bagging candidate that only differs in split randomness doesn't earn
    its keep as a distinct check."""
    hgb_pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_COLS),
        ("num", "passthrough", NUMERIC_COLS),
    ])
    cat_mask = [True] * len(CATEGORICAL_COLS) + [False] * len(NUMERIC_COLS)
    hgb_clf = HistGradientBoostingClassifier(
        categorical_features=cat_mask,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )
    hgb_pipeline = Pipeline([("pre", hgb_pre), ("clf", hgb_clf)])

    lr_pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC_COLS),
    ])
    lr_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    lr_pipeline = Pipeline([("pre", lr_pre), ("clf", lr_clf)])

    rf_pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_COLS),
        ("num", SimpleImputer(strategy="median"), NUMERIC_COLS),
    ])
    rf_clf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    rf_pipeline = Pipeline([("pre", rf_pre), ("clf", rf_clf)])

    return {
        "logistic_regression": lr_pipeline,
        "hist_gradient_boosting": hgb_pipeline,
        "random_forest": rf_pipeline,
    }


def evaluate_scores(y_true, y_score, name) -> dict:
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "brier_score": brier_score_loss(y_true, y_score),
    }


def precision_recall_at_k(y_true, y_score, k_fracs, order_value) -> pd.DataFrame:
    """For each k in k_fracs, if Ops could only act on the top-k% riskiest
    orders (ranked by y_score): what fraction of flagged orders are actually
    bad (precision), what fraction of all bad reviews does that catch
    (recall), and what R$ order value of correctly-flagged bad orders would
    that let Ops proactively address (the business-facing number)."""
    y_true = np.asarray(y_true)
    order_value = np.asarray(order_value)
    order = np.argsort(-np.asarray(y_score))
    y_sorted = y_true[order]
    val_sorted = order_value[order]

    n = len(y_true)
    total_pos = y_sorted.sum()
    total_bad_value = val_sorted[y_sorted.astype(bool)].sum()

    rows = []
    for k in k_fracs:
        cutoff = max(1, int(round(n * k)))
        top_y = y_sorted[:cutoff]
        top_val = val_sorted[:cutoff]
        captured_value = top_val[top_y.astype(bool)].sum()
        rows.append({
            "k_pct": int(k * 100),
            "n_flagged": cutoff,
            "precision": top_y.sum() / cutoff,
            "recall": (top_y.sum() / total_pos) if total_pos > 0 else np.nan,
            "bad_order_value_captured": captured_value,
            "pct_of_total_bad_value_captured": (captured_value / total_bad_value * 100) if total_bad_value > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def calibration_table(y_true, y_score, n_bins=10) -> pd.DataFrame:
    tmp = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)})
    tmp["decile"] = pd.qcut(tmp["score"], n_bins, labels=False, duplicates="drop")
    grp = tmp.groupby("decile").agg(n=("y", "size"), mean_predicted=("score", "mean"), mean_actual=("y", "mean"))
    return grp.reset_index()


def permutation_importance_table(pipeline, X_test, y_test, feature_names) -> pd.DataFrame:
    result = permutation_importance(
        pipeline, X_test, y_test, scoring="average_precision",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=1,
    )
    return pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def lr_coefficients(lr_pipeline) -> pd.DataFrame:
    pre = lr_pipeline.named_steps["pre"]
    clf = lr_pipeline.named_steps["clf"]
    out = pd.DataFrame({"feature": pre.get_feature_names_out(), "coefficient": clf.coef_[0]})
    return out.reindex(out["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def build_current_seller_risk_table(raw, order_df, k=K_SELLER) -> pd.DataFrame:
    """Non point-in-time (uses the FULL dataset's history) seller risk
    ranking -- a cheap, directly Ops-consumable secondary deliverable per
    docs/problem_framing.md section 3, distinct from the point-in-time
    seller_bad_review_rate_smoothed feature used for model training."""
    seller_stats = build_seller_level_table(raw, order_df)
    overall_rate = order_df["bad_review"].astype(float).mean()
    seller_stats["bad_review_rate_smoothed"] = (
        (seller_stats["bad_review_count"].fillna(0) + k * overall_rate) / (seller_stats["n_orders"] + k)
    )
    return seller_stats.sort_values("bad_review_rate_smoothed", ascending=False).reset_index(drop=True)


def df_to_md_table(df: pd.DataFrame, float_fmt="{:.4f}") -> str:
    # Format column-by-column (not via df.iterrows()) -- iterrows returns each
    # row as a single-dtype Series, which silently upcasts int columns (e.g.
    # k_pct, n_flagged) to float when other columns in the same row are float,
    # printing "5" as "5.0000". Same class of dtype-unification gotcha already
    # flagged in src/data/loader.py.
    cols = df.columns.tolist()
    formatted = {
        c: df[c].map(float_fmt.format) if pd.api.types.is_float_dtype(df[c]) else df[c].astype(str)
        for c in cols
    }
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for i in range(len(df)):
        lines.append("| " + " | ".join(formatted[c].iloc[i] for c in cols) + " |")
    return "\n".join(lines)


def main():
    print("[model] Loading raw tables...")
    raw = load_raw(RAW_DIR)

    print("[model] Building feature table...")
    features = build_feature_table(raw)

    print("[model] Building order-level table (for the seller risk table)...")
    order_df = build_order_level_table(raw)

    train = features[features["split"] == "train"].copy()
    test = features[features["split"] == "test"].copy()
    print(f"[model] Train: {len(train):,} rows, Test: {len(test):,} rows "
          f"(train positive rate {train['bad_review'].mean()*100:.2f}%, "
          f"test positive rate {test['bad_review'].mean()*100:.2f}%)")

    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
    X_train, y_train = train[feature_cols], train["bad_review"].astype(int)
    X_test, y_test = test[feature_cols], test["bad_review"].astype(int)

    predictions = pd.DataFrame({"order_id": test["order_id"].values, "actual": y_test.values})
    results = []

    print("[model] Scoring constant baseline...")
    const_score = np.full(len(test), y_train.mean())
    results.append(evaluate_scores(y_test, const_score, "constant_baseline"))

    print("[model] Scoring seller-heuristic baseline...")
    seller_score = test["seller_bad_review_rate_smoothed"].fillna(y_train.mean()).values
    results.append(evaluate_scores(y_test, seller_score, "seller_heuristic"))
    predictions["seller_heuristic_score"] = seller_score

    pipelines = build_pipelines()
    fitted, scores = {}, {}
    for name, pipe in pipelines.items():
        print(f"[model] Fitting {name}...")
        pipe.fit(X_train, y_train)
        score = pipe.predict_proba(X_test)[:, 1]
        results.append(evaluate_scores(y_test, score, name))
        predictions[f"{name}_score"] = score
        fitted[name] = pipe
        scores[name] = score

    metrics_df = pd.DataFrame(results)

    order_value_test = (test["total_item_price"].fillna(0) + test["total_freight_value"].fillna(0)).values
    pr_at_k = {
        name: precision_recall_at_k(y_test.values, score, K_TOP_FRACTIONS, order_value_test)
        for name, score in {"seller_heuristic": seller_score, **scores}.items()
    }

    # Apply the pre-committed decision rule (see constants above
    # build_pipelines()): does random_forest earn the shipped-model slot
    # away from HGB, or does it stay a documented-but-not-shipped cross-check?
    metrics_by_name = metrics_df.set_index("model")
    hgb_row, rf_row = metrics_by_name.loc["hist_gradient_boosting"], metrics_by_name.loc["random_forest"]
    recall20 = {n: pr_at_k[n].set_index("k_pct").loc[20, "recall"] for n in ("hist_gradient_boosting", "random_forest")}
    rf_beats_pr_auc = rf_row["pr_auc"] >= hgb_row["pr_auc"] + RF_PR_AUC_MARGIN
    rf_holds_recall20 = recall20["random_forest"] >= recall20["hist_gradient_boosting"]
    rf_calibration_ok = rf_row["brier_score"] <= RF_MAX_BRIER
    rf_ships = rf_beats_pr_auc and rf_holds_recall20 and rf_calibration_ok
    shipped_name = "random_forest" if rf_ships else DEFAULT_SHIP
    shipped_pipeline, shipped_score = fitted[shipped_name], scores[shipped_name]
    print(f"[model] Decision rule: random_forest {'ships' if rf_ships else 'does not ship'} "
          f"(PR-AUC gain {rf_row['pr_auc'] - hgb_row['pr_auc']:+.4f}, "
          f"recall@20 {'held' if rf_holds_recall20 else 'regressed'}, "
          f"brier {rf_row['brier_score']:.4f}). Shipping: {shipped_name}.")

    predictions["risk_decile"] = pd.qcut(shipped_score, 10, labels=False, duplicates="drop")

    print("[model] Computing calibration table...")
    calib = calibration_table(y_test.values, shipped_score)

    print(f"[model] Computing permutation importance ({shipped_name}, test set)...")
    imp = permutation_importance_table(shipped_pipeline, X_test, y_test, feature_cols)
    coefs = lr_coefficients(fitted["logistic_regression"])

    print("[model] Building current seller risk table...")
    seller_risk_table = build_current_seller_risk_table(raw, order_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    seller_risk_table.to_csv(OUTPUT_DIR / "seller_risk_table.csv", index=False)
    imp.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)
    coefs.to_csv(OUTPUT_DIR / "lr_coefficients.csv", index=False)
    joblib.dump(shipped_pipeline, OUTPUT_DIR / "model.joblib")

    fig, ax = plt.subplots(figsize=(6, 5))
    top_imp = imp.head(15).sort_values("importance_mean")
    ax.barh(top_imp["feature"], top_imp["importance_mean"], xerr=top_imp["importance_std"], color="#4C72B0")
    ax.set_xlabel("Permutation importance (avg. precision drop)")
    ax.set_title(f"{shipped_name} feature importance (test set)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(calib["mean_predicted"], calib["mean_actual"], marker="o", color="#C44E52", label=shipped_name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Mean actual bad_review rate")
    ax.set_title("Calibration (test set, by predicted-risk decile)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "calibration.png", bbox_inches="tight")
    plt.close(fig)

    lines = ["# Modeling Summary\n", "Auto-generated by `src/models/train_model.py`. Do not hand-edit.\n"]
    lines.append(f"\nTrain: {len(train):,} orders, Test: {len(test):,} orders "
                 f"(time-based split -- see `output/features/schema.md`).\n")
    lines.append(f"\nPositive rate: train {y_train.mean()*100:.2f}%, test {y_test.mean()*100:.2f}%.\n")
    lines.append(f"\nShipped model: **{shipped_name}** (see decision rule constants in `src/models/train_model.py`).\n")

    lines.append("\n## Model comparison (test set)\n")
    lines.append(df_to_md_table(metrics_df))

    lines.append("\n## Precision / Recall @ top-k% risk score\n")
    lines.append("k% = the riskiest k% of test orders by that model's score. "
                  "`bad_order_value_captured` = R$ order value of orders that are "
                  "both flagged and actually bad_review -- the revenue Ops could "
                  "act on if limited to that pool.\n")
    for name, tbl in pr_at_k.items():
        lines.append(f"\n**{name}**\n")
        lines.append(df_to_md_table(tbl))

    lines.append(f"\n## Calibration ({shipped_name}, by predicted-risk decile)\n")
    lines.append(df_to_md_table(calib))

    lines.append("\n## Artifacts\n")
    lines.append(f"- `feature_importance.png` / `permutation_importance.csv` -- {shipped_name} permutation importance (test set, scored on average precision)")
    lines.append("- `lr_coefficients.csv` -- logistic regression standardized coefficients (interpretability cross-check)")
    lines.append("- `calibration.png` -- reliability diagram")
    lines.append("- `test_predictions.csv` -- order-level scored test set (all candidate models' scores + risk decile)")
    lines.append("- `seller_risk_table.csv` -- current (non point-in-time) seller risk ranking, secondary deliverable per docs/problem_framing.md section 3")
    lines.append(f"- `model.joblib` -- fitted {shipped_name} pipeline")

    (OUTPUT_DIR / "metrics.md").write_text("\n".join(lines))

    print(f"\n[model] Done. See {OUTPUT_DIR / 'metrics.md'} and other artifacts under {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
