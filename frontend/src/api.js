// Central place for talking to the Greenlight backend.
//
// The backend URL differs between local dev (http://localhost:8000) and
// production (the Render URL). We read it from a Vite env var so the same code
// works in both without edits. See frontend/.env and Vercel env settings.
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function predict(features) {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features),
  });
  if (!res.ok) throw new Error(`Prediction failed: ${res.status}`);
  return res.json();
}
