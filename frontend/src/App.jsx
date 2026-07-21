import { useEffect, useState } from "react";
import { checkHealth, getMeta, predict } from "./api";
import "./App.css";

// Genres come from the backend (/api/meta) so the dropdown always matches the
// exact set the model was trained on. This short list is only a fallback shown
// before that fetch resolves.
const FALLBACK_GENRES = ["Action", "Comedy", "Drama"];
const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];
const TIER_BLURB = {
  Flop: "Likely to lose money — a risky commercial bet.",
  Moderate: "Modest returns — neither a clear win nor a disaster.",
  Hit: "Strong commercial performer.",
  Blockbuster: "Top-tier earner — the studio's home run.",
};

export default function App() {
  const [form, setForm] = useState({
    budget_usd: 100000000,
    genre: "Action",
    lead_star_power: 3,
    release_month: 7,
    is_franchise: false,
    runtime_min: 120,
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [meta, setMeta] = useState(null);

  const genres = meta?.genres ?? FALLBACK_GENRES;

  // On load, quietly ping the backend to warm it (Render free tier sleeps), and
  // fetch model metadata (genres) so the dropdown matches the deployed model.
  useEffect(() => {
    checkHealth().catch(() => {});
    getMeta().then(setMeta).catch(() => setMeta(null));
  }, []);

  function update(field, value) {
    setForm((f) => ({ ...f, [field]: value }));
  }

  async function onPredict() {
    setLoading(true);
    setError(null);
    try {
      const data = await predict(form);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <p className="kicker">Pre-release box-office intelligence</p>
        <h1>Greenlight</h1>
        <div className="intro">
          <p>
            Greenlight is a data-driven simulation model trained on historical film
            performance (1980–2020) to estimate a project's box-office tier before release.
          </p>
          <p>
            Using only pre-release inputs ( including budget, genre, cast track record,
            release timing, and franchise status ) the model compares your film against
            patterns observed across thousands of past releases to classify its likely
            commercial outcome (Flop, Moderate, Hit, or Blockbuster).
          </p>
        </div>
      </header>

      <section className="panel inputs">
        <h2>The film</h2>
        <div className="grid">
          <label>
            <span className="field-label">Production budget</span>
            <span className="budget-val">${Number(form.budget_usd).toLocaleString()}</span>
            <input
              type="range" min="1000000" max="400000000" step="1000000"
              value={form.budget_usd}
              onChange={(e) => update("budget_usd", Number(e.target.value))}
            />
          </label>

          <label>
            <span className="field-label">Primary genre</span>
            <select value={form.genre} onChange={(e) => update("genre", e.target.value)}>
              {genres.map((g) => <option key={g} value={g}>{g}</option>)}
            </select>
          </label>

          <label>
            <span className="field-label">Lead-actor star power</span>
            <select value={form.lead_star_power} onChange={(e) => update("lead_star_power", Number(e.target.value))}>
              <option value={1}>1 — Unknown</option>
              <option value={2}>2 — Emerging</option>
              <option value={3}>3 — Established</option>
              <option value={4}>4 — A-list</option>
              <option value={5}>5 — Global superstar</option>
            </select>
          </label>

          <label>
            <span className="field-label">Release month</span>
            <select value={form.release_month} onChange={(e) => update("release_month", Number(e.target.value))}>
              {MONTHS.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}
            </select>
          </label>

          <label>
            <span className="field-label">Runtime</span>
            <span className="budget-val">{form.runtime_min} min</span>
            <input
              type="range" min="70" max="210" step="5"
              value={form.runtime_min}
              onChange={(e) => update("runtime_min", Number(e.target.value))}
            />
          </label>

          <label className="toggle">
            <input
              type="checkbox" checked={form.is_franchise}
              onChange={(e) => update("is_franchise", e.target.checked)}
            />
            Part of a franchise / sequel
          </label>
        </div>

        <button className="predict-btn" onClick={onPredict} disabled={loading}>
          {loading ? "Analyzing…" : "Predict box-office tier"}
        </button>
        {error && <p className="error">⚠ {error}</p>}
      </section>

      {result && (
        <section className="panel result">
          <div className="verdict">
            <span className="verdict-kicker">Projected tier</span>
            <span className={`tier tier-${result.tier.toLowerCase()}`}>{result.tier}</span>
            <p className="blurb">{TIER_BLURB[result.tier]}</p>
            <p className="confidence">Model confidence — {Math.round(result.confidence * 100)}%</p>
            <div className="conf-meter">
              <div className="conf-fill" style={{ width: `${Math.round(result.confidence * 100)}%` }} />
            </div>
          </div>

          <div className="factors">
            <h3>What drove this prediction</h3>
            {(() => {
              // Scale bars relative to the strongest factor so the chart is
              // always well-proportioned regardless of raw SHAP magnitudes.
              const maxAbs = Math.max(...result.factors.map((f) => Math.abs(f.contribution)), 0.01);
              return result.factors.map((f) => {
              const pct = Math.max((Math.abs(f.contribution) / maxAbs) * 100, 2);
              const positive = f.contribution >= 0;
              return (
                <div className="factor-row" key={f.feature}>
                  <span className="factor-name">{f.feature}</span>
                  <div className="bar-track">
                    <div className={`bar ${positive ? "pos" : "neg"}`} style={{ width: `${pct}%` }} />
                  </div>
                  <span className="factor-val">{positive ? "+" : ""}{f.contribution}</span>
                </div>
              );
            });
            })()}
            <p className="hint">Green = pushed the prediction up · Red = pushed it down</p>
          </div>

          <div className="interpretation">
            <h3>Model Interpretation</h3>
            <p>{result.interpretation}</p>
            <span className="model-tag">model: {result.model_version}</span>
          </div>
        </section>
      )}

      <footer className="footer">
        <p>
          Predictions are generated by a trained machine learning model; the LLM is
          used only to explain the result.
        </p>
        <p>
          The model relies exclusively on pre-release inputs, with post-release signals
          removed to prevent data leakage.
        </p>
        <p>
          Observed accuracy: ~50% exact tier and ~90% within one tier (vs. 25% random
          baseline).
        </p>
        <p>
          Box-office outcomes remain inherently uncertain, and results should be
          interpreted as probabilistic estimates rather than definitive forecasts.
        </p>
      </footer>
    </div>
  );
}
