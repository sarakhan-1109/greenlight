"""
Day 2 — data cleaning.

Turns the raw, messy dataset into a tidy table ready for feature engineering and
training (Day 3). Every decision here is deliberate and commented, because the
*handling of messiness* is a core talking point for this project.

Run:  ../backend/.venv/bin/python clean_data.py
Input:  data/movies_raw.csv
Output: data/movies_clean.csv
"""

import re

import pandas as pd

RAW = "data/movies_raw.csv"
OUT = "data/movies_clean.csv"

# ---------------------------------------------------------------------------
# ANTI-LEAKAGE: these columns describe what happened AFTER a film released
# (IMDB score and accumulated vote counts). Using them as inputs would leak the
# outcome we're trying to predict. We drop them here so they can never reach the
# model. `gross` is kept ONLY long enough to build the target tier, then it is
# NOT passed to the model as a feature (see train.py).
# ---------------------------------------------------------------------------
LEAKAGE_COLS = ["score", "votes"]

MONTHS = {
    m: i
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"],
        start=1,
    )
}


def parse_release_month(released: str, year: float) -> int | None:
    """Extract release month from strings like 'June 13, 1980 (United States)'.

    The `released` field is free text and occasionally malformed, so we parse
    defensively and return None when we can't find a month name.
    """
    if not isinstance(released, str):
        return None
    match = re.match(r"\s*([A-Za-z]+)", released)
    if not match:
        return None
    return MONTHS.get(match.group(1).lower())


# Title patterns that strongly suggest a sequel / franchise entry. This is an
# APPROXIMATION (the dataset has no explicit franchise field) and we flag it as
# a known limitation in the README. Examples it catches: "Rocky II",
# "Toy Story 3", "The Dark Knight Rises" (via 'Part'), "Shrek: ...".
SEQUEL_PATTERNS = re.compile(
    r"(\b(II|III|IV|V|VI|VII|VIII|IX|X)\b|\b[2-9]\b|\bpart\b|\bchapter\b|:\s)",
    re.IGNORECASE,
)


def looks_like_franchise(name: str) -> int:
    """1 if the title looks like a sequel/franchise entry, else 0 (approx.)."""
    if not isinstance(name, str):
        return 0
    return int(bool(SEQUEL_PATTERNS.search(name)))


def main() -> None:
    df = pd.read_csv(RAW)
    start = len(df)
    print(f"Loaded {start} raw rows.")

    # 1) Drop post-release leakage columns outright.
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    print(f"Dropped leakage columns: {LEAKAGE_COLS}")

    # 2) Keep only rows with the essentials. Budget is our top feature; we do NOT
    #    impute it (imputing the most predictive feature would fabricate signal).
    before = len(df)
    df = df[(df["budget"] > 0) & (df["gross"] > 0) & (df["runtime"] > 0)]
    df = df.dropna(subset=["budget", "gross", "runtime", "genre", "released", "star", "year"])
    print(f"Kept rows with budget+gross+runtime+core fields: {len(df)} (dropped {before - len(df)})")

    # 3) Derive pre-release time features from the messy `released` text.
    df["release_month"] = df.apply(lambda r: parse_release_month(r["released"], r["year"]), axis=1)
    df = df.dropna(subset=["release_month"])
    df["release_month"] = df["release_month"].astype(int)

    # 4) Approximate franchise/sequel flag from the title (documented limitation).
    df["is_franchise"] = df["name"].apply(looks_like_franchise)

    # 5) Group rare genres so tiny classes (Western=3, etc.) don't destabilize
    #    the model. Genres with < 40 films fold into "Other".
    counts = df["genre"].value_counts()
    rare = counts[counts < 40].index
    df["genre"] = df["genre"].where(~df["genre"].isin(rare), "Other")
    print(f"Folded {len(rare)} rare genres into 'Other'.")

    # 6) Keep a tidy column set. `gross` stays for now to build the target in
    #    train.py; it is removed before the feature matrix is built.
    keep = ["name", "budget", "gross", "genre", "rating", "runtime", "star",
            "company", "country", "year", "release_month", "is_franchise"]
    df = df[keep].reset_index(drop=True)

    df.to_csv(OUT, index=False)
    print(f"\nWrote {len(df)} clean rows -> {OUT}")
    print(f"Franchise share: {df['is_franchise'].mean()*100:.1f}%")
    print(f"Genres: {df['genre'].nunique()}  |  Ratings: {df['rating'].nunique()}")


if __name__ == "__main__":
    main()
