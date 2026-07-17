import { useEffect, useState } from "react";
import { checkHealth, predict } from "./api";
import "./App.css";

// Dropdown option sources. GENRES is a starter list; on Day 3 it will be
// aligned to whatever genres the trained model actually knows about.
const GENRES = [
  "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama",
  "Family", "Fantasy", "Horror", "Romance", "Science Fiction", "Thriller",
];
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
  const [backendOk, setBackendOk] = useState(null); // null=checking, true/false
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

  // On load, ping the backend so the header can honestly show connection state.
  useEffect(() => {
    checkHealth().then(() => setBackendOk(true)).catch(() => setBackendOk(false));
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
        <h1>🎬 Greenlight</h1>
        <p className="tagline">
          Most films lose money. Greenlight estimates a project's box-office tier
          from pre-release signals alone — budget, cast, timing, franchise — so you
          can see the commercial bet before a single frame is shot.
        </p>
        <span className={`status ${backendOk ? "ok" : backendOk === false ? "down" : ""}`}>
          {backendOk === null ? "connecting…" : backendOk ? "backend connected" : "backend offline"}
        </span>
      </header>

      <section className="panel inputs">
        <h2>The film</h2>
        <div className="grid">
          <label>
            Production budget
            <span className="budget-val">${Number(form.budget_usd).toLocaleString()}</span>
            <input
              type="range" min="1000000" max="400000000" step="1000000"
              value={form.budget_usd}
              onChange={(e) => update("budget_usd", Number(e.target.value))}
            />
          </label>

          <label>
            Primary genre
            <select value={form.genre} onChange={(e) => update("genre", e.target.value)}>
              {GENRES.map((g) => <option key={g} value={g}>{g}</option>)}
            </select>
          </label>

          <label>
            Lead-actor star power
            <select value={form.lead_star_power} onChange={(e) => update("lead_star_power", Number(e.target.value))}>
              <option value={1}>1 — Unknown</option>
              <option value={2}>2 — Emerging</option>
              <option value={3}>3 — Established</option>
              <option value={4}>4 — A-list</option>
              <option value={5}>5 — Global superstar</option>
            </select>
          </label>

          <label>
            Release month
            <select value={form.release_month} onChange={(e) => update("release_month", Number(e.target.value))}>
              {MONTHS.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}
            </select>
          </label>

          <label>
            Runtime
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

        <button className="predict-btn" onClick={onPredict} disabled={loading || backendOk === false}>
          {loading ? "Analyzing…" : "Predict box-office tier"}
        </button>
        {error && <p className="error">⚠ {error}</p>}
      </section>

      {result && (
        <section className="panel result">
          <div className="verdict">
            <span className={`tier tier-${result.tier.toLowerCase()}`}>{result.tier}</span>
            <div className="verdict-meta">
              <p className="blurb">{TIER_BLURB[result.tier]}</p>
              <p className="confidence">Model confidence: <strong>{Math.round(result.confidence * 100)}%</strong></p>
            </div>
          </div>

          <div className="factors">
            <h3>What drove this prediction</h3>
            {result.factors.map((f) => {
              const pct = Math.min(Math.abs(f.contribution) * 40, 100);
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
            })}
            <p className="hint">Green = pushed the prediction up · Red = pushed it down</p>
          </div>

          <div className="interpretation">
            <h3>Analyst's note</h3>
            <p>{result.interpretation}</p>
            <span className="model-tag">model: {result.model_version}</span>
          </div>
        </section>
      )}

      <footer className="footer">
        <p>ML predicts the tier · an LLM only explains it · pre-release features only (no data leakage)</p>
      </footer>
    </div>
  );
}
