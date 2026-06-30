// Thin client for the Varuna FastAPI backend. Base is same-origin (Vite proxies /api in dev).
const BASE = import.meta.env.VITE_API_BASE || "";

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
  meta: () => get("/api/meta"),
  sinks: () => get("/api/sinks"),
  recharge: () => get("/api/recharge"),
  alerts: () => get("/api/alerts"),
  validation: () => get("/api/validation"),
  canalPlan: () => get("/api/canal_plan"),
  whatif: (rain_mm, dig_sites) => post("/api/whatif", { rain_mm, dig_sites }),
  canals: (rain_mm, n_canals) => post("/api/canals", { rain_mm, n_canals }),
  optimize: (design_rain, budget_m3) => post("/api/optimize", { design_rain, budget_m3 }),
  chat: (message, history) => post("/api/chat", { message, history }),
};
