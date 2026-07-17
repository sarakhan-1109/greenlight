# model/ — data science work

This folder holds the offline work that produces the trained model the backend
serves. It is separate from `backend/` so the web app stays lean.

```
data/         raw + cleaned datasets (gitignored; regenerable)
artifacts/    trained model + metadata the backend loads (gitignored)
```

## Plan

- **Day 2** — `clean_data.py` / a notebook: load raw TMDB data, clean it, and
  keep notes on what was messy. **Hard rule:** drop every post-release column
  (revenue is used ONLY to build the tier label, never as an input feature).
- **Day 3** — `train.py`: engineer pre-release features, define tier cutoffs
  from the revenue distribution, train XGBoost, save the model + feature list to
  `artifacts/`. The backend's `predictor.py` loads exactly those artifacts.

## The anti-leakage rule (why it matters)

`revenue` appears in the data but is **post-release**. We use it to *define the
target tier* (the thing we predict) and then remove it, so it can never sneak in
as an input. Same for ratings, vote counts, and popularity measured after
release. This is the project's central design decision.
