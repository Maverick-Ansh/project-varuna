// Thin client for the Varuna FastAPI backend. Base is same-origin (Vite proxies /api in dev).
const BASE = import.meta.env.VITE_API_BASE || "";

const q = (area) => (area ? `?area=${encodeURIComponent(area)}` : "");

async function get(path) {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}
async function post(path, body) {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} -> ${r.status}: ${await r.text()}`);
  return r.json();
}

export const api = {
  areas: () => get("/api/areas"),
  meta: (area) => get(`/api/meta${q(area)}`),
  sinks: (area) => get(`/api/sinks${q(area)}`),
  recharge: (area) => get(`/api/recharge${q(area)}`),
  alerts: (area) => get(`/api/alerts${q(area)}`),
  validation: (area) => get(`/api/validation${q(area)}`),
  canalPlan: (area) => get(`/api/canal_plan${q(area)}`),
  whatif: (rain_mm, dig_sites, area) => post("/api/whatif", { rain_mm, dig_sites, area }),
  canals: (rain_mm, n_canals, area) => post("/api/canals", { rain_mm, n_canals, area }),
  storage: (rain_mm, area) => post("/api/storage", { rain_mm, area }),
  optimize: (design_rain, budget_m3, area) => post("/api/optimize", { design_rain, budget_m3, area }),
  costbenefit: (rain_mm, area) => post("/api/costbenefit", { rain_mm, area }),
  exposure: (rain_mm, area) => post("/api/exposure", { rain_mm, area }),
  report: (rain_mm, area) => post("/api/report", { rain_mm, area }),
  chat: (message, history, area) => post("/api/chat", { message, history, area }),
};
