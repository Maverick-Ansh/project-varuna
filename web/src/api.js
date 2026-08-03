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
  flowfield: (rain_mm, area, opts = {}) => post("/api/flowfield", { rain_mm, area, ...opts }),
  canals: (rain_mm, n_canals, area) => post("/api/canals", { rain_mm, n_canals, area }),
  storage: (rain_mm, area) => post("/api/storage", { rain_mm, area }),
  optimize: (design_rain, budget_m3, area) => post("/api/optimize", { design_rain, budget_m3, area }),
  costbenefit: (rain_mm, area) => post("/api/costbenefit", { rain_mm, area }),
  exposure: (rain_mm, area) => post("/api/exposure", { rain_mm, area }),
  route: (start, end, rain_mm, area) => post("/api/route", { start, end, rain_mm, area }),
  report: (rain_mm, area) => post("/api/report", { rain_mm, area }),
  chat: (message, history, area) => post("/api/chat", { message, history, area }),

  // v2 live layer
  weather: (area) => get(`/api/weather${q(area)}`),
  alertsLive: (area) => get(`/api/alerts_live${q(area)}`),
  reports: (area, hours = 24) => get(`/api/reports${q(area)}${area ? "&" : "?"}hours=${hours}`),
  reportFlood: (body) => post("/api/reports", body),
  advisory: (area) => get(`/api/advisory${q(area)}`),
  waterbalance: (area, rain_mm, efficiency = 1) =>
    get(`/api/waterbalance${q(area)}${area ? "&" : "?"}rain_mm=${rain_mm}&efficiency=${efficiency}`),
  storagePlan: (area) => get(`/api/storage_plan${q(area)}`),
  nightlights: (area) => get(`/api/nightlights${q(area)}`),
  learningLog: (area) => get(`/api/learning_log${q(area)}`),
  depthValidation: (area) => get(`/api/depth_validation${q(area)}`),
  city: (cityId, rain_mm) =>
    get(`/api/city?city=${encodeURIComponent(cityId)}${rain_mm != null ? `&rain_mm=${rain_mm}` : ""}`),
  imageUrl: (name, area) => `${BASE}/api/image/${name}${q(area)}`,

  // v3 "Bhujal": metered recharge plan, state screen, quarantined news
  rechargePlan: (area) => get(`/api/recharge_plan${q(area)}`),
  stateScreen: (state = "karnataka") => get(`/api/state_screen?state=${encodeURIComponent(state)}`),
  news: (area, hours = 24) => get(`/api/news${q(area)}${area ? "&" : "?"}hours=${hours}`),
};
