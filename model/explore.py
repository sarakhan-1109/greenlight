"""
Day 2 — data exploration / profiling.

Goal: understand the raw dataset's shape and messiness BEFORE cleaning, and keep
notes (printed here) for the README + interview talking points. This script only
reads and reports; it changes nothing.

Run:  ../backend/.venv/bin/python explore.py
"""

import pandas as pd

RAW = "data/movies_raw.csv"

# Columns that are POST-RELEASE and must never become model inputs.
LEAKAGE_COLS = ["score", "votes"]
# Column used ONLY to build the target label, then dropped as a feature.
TARGET_SOURCE = "gross"

df = pd.read_csv(RAW)

print(f"\n=== SHAPE ===\n{df.shape[0]} rows x {df.shape[1]} cols")
print(f"Years: {int(df['year'].min())}–{int(df['year'].max())}")

print("\n=== MISSING VALUES (count / %) ===")
miss = df.isna().sum()
for col in df.columns:
    if miss[col]:
        print(f"  {col:10s} {miss[col]:5d}  ({miss[col] / len(df) * 100:4.1f}%)")

print("\n=== ZERO / SUSPICIOUS VALUES ===")
for col in ["budget", "gross", "runtime"]:
    zeros = (df[col] == 0).sum()
    print(f"  {col:8s} zeros: {zeros}")

print("\n=== BUDGET availability (our most important feature) ===")
has_budget = df["budget"].notna() & (df["budget"] > 0)
print(f"  usable budget: {has_budget.sum()} / {len(df)} ({has_budget.mean()*100:.1f}%)")

print("\n=== GROSS distribution (for tier cutoffs) ===")
gross = df.loc[df[TARGET_SOURCE].notna() & (df[TARGET_SOURCE] > 0), TARGET_SOURCE]
for q in [0.25, 0.50, 0.75, 0.90]:
    print(f"  {int(q*100)}th pct: ${gross.quantile(q):,.0f}")

print("\n=== GENRE counts ===")
print(df["genre"].value_counts().head(15).to_string())

print("\n=== Rows usable after requiring budget+gross+runtime ===")
usable = df[df["budget"].gt(0) & df["gross"].gt(0) & df["runtime"].gt(0)]
usable = usable.dropna(subset=["budget", "gross", "runtime", "genre", "released", "star"])
print(f"  {len(usable)} rows fully usable for modeling")

print("\n=== Lead-actor (star) coverage for prior-film proxy ===")
star_counts = usable["star"].value_counts()
print(f"  unique lead actors: {star_counts.size}")
print(f"  actors with >=2 films: {(star_counts >= 2).sum()}")
print(f"  median films per actor: {star_counts.median():.0f}")
