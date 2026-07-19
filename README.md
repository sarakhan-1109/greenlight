# 🎬 Greenlight — Pre-Release Box-Office Tier Predictor

**Most films lose money.** Greenlight estimates a film's box-office tier —
**Flop / Moderate / Hit / Blockbuster** — from *pre-release signals alone*
(budget, cast, timing, franchise, runtime), so you can see the commercial bet
before a single frame is shot.

> "Greenlighting" is the real industry term for a studio approving a film for
> production — a high-stakes bet. Greenlight is framed as decision support for
> that bet.

**Live demo:** _(coming Day 1 — deploying to Vercel + Render)_

---

## Why this project is interesting (the design decisions)

1. **Pre-release-only features (no data leakage).** The model uses *only*
   information knowable before a film releases. It never sees actual revenue,
   ratings, vote counts, or post-release popularity. Avoiding this "data
   leakage" is the classic mistake that makes a model look great in testing and
   fail in reality — designing against it is the headline decision here.

2. **ML predicts, the LLM only explains.** An XGBoost model produces the tier
   and the per-feature contributions. A separate LLM layer (Google Gemini)
   takes that finished output and writes a plain-English "analyst's note." The
   language model never makes the prediction. This layer is an isolated,
   swappable module — the project moved from Anthropic Claude to Gemini's free
   tier with a one-file change.

3. **Framed as a decision, not an exam.** The tool answers a real question — *is
   this a smart commercial bet?* — not "can I hit a leaderboard number."

## Architecture

```
┌─────────────┐      pre-release        ┌──────────────────┐
│   React     │  ──── features ─────▶   │   FastAPI        │
│  (Vercel)   │                         │   (Render)       │
│             │  ◀── tier + factors ─── │                  │
│  one page   │      + analyst's note   │  ┌────────────┐  │
└─────────────┘                         │  │ XGBoost    │  │ ← predicts the tier
                                        │  │ predictor  │  │
                                        │  └─────┬──────┘  │
                                        │        │ output  │
                                        │  ┌─────▼──────┐  │
                                        │  │ Gemini     │  │ ← explains it (only)
                                        │  │ interpreter│  │
                                        │  └────────────┘  │
                                        └──────────────────┘
```

## Tech stack

| Layer     | Choice                              |
|-----------|-------------------------------------|
| Data      | TMDB movie data (historical)        |
| Model     | XGBoost (single gradient-boosted tree model) |
| Backend   | FastAPI (Python)                    |
| Frontend  | React (Vite), single page           |
| LLM       | Google Gemini (interpretation only, free tier) |
| Deploy    | Vercel (frontend) + Render (backend)|

No database, no auth — Greenlight is a stateless predictor.

## Repository layout

```
backend/      FastAPI app (main.py + app/: schemas, predictor, interpreter)
frontend/     React single-page app (Vite)
model/        Data-science work: data/ (raw+clean), notebooks, training script
```

## Local development

```bash
# Backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001

# Frontend (separate terminal)
cd frontend
cp .env.example .env      # points at http://localhost:8001
npm install
npm run dev               # http://localhost:5173
```

## Status / roadmap

- [x] **Day 1** — Scaffold + trivial end-to-end slice (stub model) + deploy live.
- [ ] **Day 2** — Source & clean real TMDB data; verify no post-release leakage.
- [ ] **Day 3** — Train XGBoost; define tier cutoffs; real factor contributions.
- [ ] **Day 4** — Wire the Claude interpretation layer.
- [ ] **Day 5** — Finish the React UI; deploy the full version.
- [ ] **Day 6** — Full README: motivation, honest accuracy, "what I'd do differently."
- [ ] **Day 7** — Polish + interview walkthrough.

## Honest accuracy note

Box-office prediction is genuinely hard. A 4-tier model in the **~55–70%**
range is a good, honest result. A suspiciously high number (>90%) is treated as
a **leakage bug to investigate**, not a win.

## Future improvements (deliberately out of scope for the sprint)

ROI/profitability target · "compare two films" mode · more features · saved
history (would add a database) · richer genre/era data-story.
