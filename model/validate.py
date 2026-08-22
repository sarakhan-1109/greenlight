"""
Day 6 — rigorous re-validation of the box-office tier model.

Goes beyond the single train/test split in train.py to answer three questions
honestly (results feed the README's accuracy discussion):

  1. Is ~50% stable?           -> 5-fold stratified cross-validation (mean +/- std)
  2. Can it predict the FUTURE? -> temporal holdout: train on older films, test on
                                   the most recent years (the realistic scenario)
  3. WHERE does it err?        -> confusion matrix + accuracy broken down by era

Reuses the exact feature pipeline from train.py so this measures the deployed
model, not a different one.

Run:  ../backend/.venv/bin/python validate.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from train import CATEGORICAL, FEATURES, TIERS, build_star_power

CLEAN = "data/movies_clean.csv"

# Same hyperparameters as the deployed model (train.py).
PARAMS = dict(
    objective="multi:softprob",
    num_class=4,
    n_estimators=350,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=4,
    reg_lambda=1.5,
    enable_categorical=True,
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=42,
)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the leakage-free star-power level (prior films only)."""
    df = df.copy()
    star_level, _ = build_star_power(df)
    df["star_power_level"] = star_level.astype(int)
    return df


def make_X(df: pd.DataFrame, genre_categories) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=genre_categories)
    return X


def within_one(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))


def main() -> None:
    df = pd.read_csv(CLEAN)
    df = prep(df)
    genre_categories = sorted(df["genre"].unique().tolist())

    # =====================================================================
    # 1) 5-FOLD STRATIFIED CROSS-VALIDATION (global quartile tiers)
    # =====================================================================
    df["tier"] = pd.qcut(df["gross"], 4, labels=TIERS)
    y = df["tier"].cat.codes.to_numpy()
    X = make_X(df, genre_categories)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, adjs = [], []
    for tr, te in skf.split(X, y):
        model = XGBClassifier(**PARAMS)
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[te])
        accs.append(accuracy_score(y[te], pred))
        adjs.append(within_one(y[te], pred))

    print("\n=== 1) 5-FOLD CROSS-VALIDATION (random splits) ===")
    print(f"  exact-tier accuracy: {np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%")
    print(f"  within-one-tier:     {np.mean(adjs)*100:.1f}% +/- {np.std(adjs)*100:.1f}%")
    print(f"  per-fold exact:      {[round(a*100,1) for a in accs]}")
    print("  (random-guess baseline = 25%)")

    # =====================================================================
    # 2) TEMPORAL HOLDOUT — train on older films, test on recent years.
    #    Tier cutoffs are computed from the TRAINING period only (no peeking
    #    at the future), then applied to the test films.
    # =====================================================================
    df["release_key"] = df["year"] * 100 + df["release_month"]
    df = df.sort_values("release_key").reset_index(drop=True)

    # Split so ~80% (oldest) is train, ~20% (most recent) is test.
    cut = df["release_key"].quantile(0.80)
    train_df = df[df["release_key"] <= cut]
    test_df = df[df["release_key"] > cut]
    split_year = int(cut // 100)

    # Tiers from the TRAIN period's gross distribution only.
    _, bins = pd.qcut(train_df["gross"], 4, labels=TIERS, retbins=True)
    bins[0], bins[-1] = -np.inf, np.inf  # let test films fall outside train range

    def to_tier_code(g):
        return int(np.clip(np.digitize(g, bins[1:-1]), 0, 3))

    y_tr = train_df["gross"].apply(to_tier_code).to_numpy()
    y_te = test_df["gross"].apply(to_tier_code).to_numpy()
    X_tr = make_X(train_df, genre_categories)
    X_te = make_X(test_df, genre_categories)

    model = XGBClassifier(**PARAMS)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)

    print("\n=== 2) TEMPORAL HOLDOUT (predict the future) ===")
    print(f"  train: films through ~{split_year}  ({len(train_df)} films)")
    print(f"  test:  films after   ~{split_year}  ({len(test_df)} films)")
    print(f"  exact-tier accuracy: {acc*100:.1f}%")
    print(f"  within-one-tier:     {within_one(y_te, pred)*100:.1f}%")
    print("  (random-guess baseline = 25%)")

    print("\n  Confusion matrix (rows=true, cols=pred):")
    print("        " + "  ".join(f"{t[:4]:>5}" for t in TIERS))
    cm = confusion_matrix(y_te, pred, labels=[0, 1, 2, 3])
    for i, row in enumerate(cm):
        print(f"  {TIERS[i][:4]:>5} " + "  ".join(f"{v:>5}" for v in row))

    # =====================================================================
    # 3) ERROR ANALYSIS — accuracy by era (does it degrade over time?)
    # =====================================================================
    print("\n=== 3) ACCURACY BY ERA (5-fold CV predictions) ===")
    # Out-of-fold predictions across the whole dataset for a fair by-era view.
    oof = np.empty(len(y), dtype=int)
    for tr, te in skf.split(X, y):
        m = XGBClassifier(**PARAMS)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
    df_era = df.copy()
    df_era["correct"] = (oof == y)
    df_era["decade"] = (df_era["year"] // 10 * 10).astype(int)
    for dec, grp in df_era.groupby("decade"):
        print(f"  {dec}s: {grp['correct'].mean()*100:4.1f}% exact   (n={len(grp)})")


if __name__ == "__main__":
    main()
