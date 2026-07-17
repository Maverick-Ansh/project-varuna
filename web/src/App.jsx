import React, { useEffect, useRef, useState } from "react";
import {
  MapContainer, TileLayer, GeoJSON, CircleMarker, Polyline, ImageOverlay, Rectangle, Tooltip, useMap, useMapEvents,
} from "react-leaflet";
import L from "leaflet";
import { api } from "./api.js";
import { usePolling } from "./hooks.js";
import {
  LivePanel, ReportPanel, WaterBalancePanel, AdvisoryPanel, NightLightsPanel, CityPanel, WB,
} from "./components/panels.jsx";

const LEVEL_COLOR = { RED: "#e23", AMBER: "#f90", GREEN: "#2a4" };

// Marker palette — one hue per physical thing, CVD-validated. Drain LINES stay the
// project purple; inlets (water enters) are amber; dug storage is pink; safe low
// ground teal. Never reuse a family across meanings.
const C = {
  drain: "#7b2fbe",
  inlet: { color: "#92400e", fillColor: "#d97706" },
  pit: { color: "#9d174d", fillColor: "#ec4899" },
  lowland: { color: "#115e59", fillColor: "#0d9488" },
  boundary: { color: "#555", fillColor: "#999" },
  route: "#1d4ed8", routeCasing: "#ffffff", shortest: "#64748b",
};
const OUTFALL_STYLE = { lowland: C.lowland, pit: C.pit, boundary: C.boundary };

// citizen-report depth -> sequential blues (light ankle -> dark chest), white ring on the map
const REPORT_COLOR = { ankle: "#93c5fd", knee: "#60a5fa", waist: "#3b82f6", chest: "#1e40af" };

function Dot({ c }) {
  return <span className="lg-dot" style={{ background: c.fillColor || c, borderColor: c.color || c }} />;
}
function Line({ c, dash }) {
  return <span className="lg-line" style={{ background: dash ? "none" : c, borderTop: dash ? `2px dashed ${c}` : "none" }} />;
}

function Legend({ show, canal, alerts, exposure, dig, route, reports, nightlights }) {
  const rows = [];
  if (show.flood) rows.push([<span className="lg-flood" key="s" />, "flood depth"]);
  if (show.reports && reports?.length > 0) rows.push([<Dot c={{ color: "#fff", fillColor: "#3b82f6" }} key="s" />, "citizen report (darker = deeper)"]);
  if (show.nightlights && nightlights) rows.push([<span className="lg-line" key="s" style={{ background: "#ef4444", height: 8 }} />, "power outage (night lights)"]);
  if (show.canal && canal) {
    rows.push([<Line c={C.drain} key="s" />, "storm drain (flow-weighted)"]);
    if (canal.network) rows.push([<Dot c={C.inlet} key="s" />, "drain inlet"]);
    rows.push([<Dot c={C.pit} key="s" />, "storage pit"]);
    if (canal.network) rows.push([<Dot c={C.lowland} key="s" />, "outfall: safe lowland"]);
  }
  if (show.dig && dig) rows.push([<Dot c={{ color: "#630", fillColor: "#c96" }} key="s" />, "dig site"]);
  if (show.alerts && alerts.length > 0) rows.push([<Dot c={{ fillColor: "#e23", color: "#e23" }} key="s" />, "sink alert (R/A/G)"]);
  if (show.roads && exposure) rows.push([<Line c="#c30" key="s" />, "flooded road"]);
  if (show.buildings && exposure) rows.push([<Dot c={{ color: "#900", fillColor: "#e33" }} key="s" />, "at-risk building"]);
  if (route) {
    rows.push([<Line c={C.route} key="s" />, "flood-safe route"]);
    rows.push([<Line c={C.shortest} dash key="s" />, "shortest (ignores water)"]);
  }
  if (!rows.length) return null;
  return (
    <div className="legend">
      {rows.map(([swatch, label], i) => (
        <div key={i} className="lg-row">{swatch}<span>{label}</span></div>
      ))}
    </div>
  );
}

// map clicks -> report pin (priority) or route endpoints
function MapClicks({ reporting, onReportPin, picking, onPick }) {
  useMapEvents({
    click: (e) => {
      if (reporting) onReportPin([e.latlng.lat, e.latlng.lng]);
      else if (picking) onPick([e.latlng.lat, e.latlng.lng]);
    },
  });
  return null;
}

function Panel({ title, children }) {
  return (
    <div className="panel">
      <h3>{title}</h3>
      {children}
    </div>
  );
}

function Recenter({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    if (center) map.setView(center, zoom || map.getZoom());
  }, [center && center[0], center && center[1], zoom]);
  return null;
}

function ValidationPanel({ v, learn }) {
  if (!v) return <Panel title="Validation"><p className="muted">loading…</p></Panel>;
  const stat = v.static_depth_vs_sar;
  const rep = v.calibration_report;
  const fmt = (x) => (x == null ? "—" : Number(x).toFixed(3));
  const last = learn?.entries?.length ? learn.entries[learn.entries.length - 1] : null;
  return (
    <Panel title="Validation vs Sentinel-1 SAR">
      {stat && (
        <p>Static depth-map CSI <b>{fmt(stat.csi)}</b> (POD {fmt(stat.pod)}, FAR {fmt(stat.far)})</p>
      )}
      {rep ? (
        <div className="kv">
          <div>Held-out CSI</div>
          <div><span className="muted">textbook</span> {fmt(rep.baseline?.mean_csi_test)}
            {" → "}<b>{fmt(rep.calibrated?.mean_csi_test)}</b> <span className="muted">calibrated</span></div>
        </div>
      ) : <p className="muted">SAR validation available for Patna only.</p>}
      {last && (
        <p className="muted" style={{ marginBottom: 0 }}>
          Self-improvement {last.date?.slice(0, 10)}:{" "}
          {last.accepted ? <b style={{ color: "#2a4" }}>model updated</b> : "update held back"}
          {" "}— {last.reason}
        </p>
      )}
    </Panel>
  );
}

export default function App() {
  const [areas, setAreas] = useState([]);
  const [view, setView] = useState(null);              // area id | "city:<id>"
  const isCity = !!view && view.startsWith("city:");
  const area = isCity ? null : view;
  const cityId = isCity ? view.slice(5) : null;

  const [meta, setMeta] = useState(null);
  const [sinks, setSinks] = useState(null);
  const [recharge, setRecharge] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [validation, setValidation] = useState(null);
  const [canal, setCanal] = useState(null);

  const [storage, setStorage] = useState(null);
  const [dig, setDig] = useState(null);
  const [cost, setCost] = useState(null);
  const [exposure, setExposure] = useState(null);
  const [report, setReport] = useState(null);

  const [rain, setRain] = useState(100);
  const [flood, setFlood] = useState(null);     // {overlay_png, bounds, summary}
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  // v2: live weather / citizen reports / water budget / advisory / night lights / city
  const [live, setLive] = useState(false);
  const [weather, setWeather] = useState(null);
  const [liveAlerts, setLiveAlerts] = useState(null);
  const [reports, setReports] = useState([]);
  const [reporting, setReporting] = useState(false);
  const [reportDraft, setReportDraft] = useState(null);
  const [reportStatus, setReportStatus] = useState(null);
  const [adv, setAdv] = useState(null);
  const [wb, setWb] = useState(null);
  const [storagePlan, setStoragePlan] = useState(null);
  const [unitM3, setUnitM3] = useState(50);
  const [eff, setEff] = useState(0.8);
  const [nl, setNl] = useState(null);
  const [citySummary, setCitySummary] = useState(null);
  const [learn, setLearn] = useState(null);

  const [show, setShow] = useState({
    flood: true, alerts: true, sinks: false, recharge: false, canal: true, dig: true,
    buildings: true, roads: true, route: true, reports: true, nightlights: true,
    containers: false,
  });
  const [chat, setChat] = useState([]);
  const [msg, setMsg] = useState("");
  const debounce = useRef(null);
  const wbDebounce = useRef(null);

  // evacuation routing: two picked points -> /api/route
  const [picking, setPicking] = useState(false);
  const [routePts, setRoutePts] = useState([]);        // [[lat,lon], [lat,lon]]
  const [routeRes, setRouteRes] = useState(null);
  const [routeErr, setRouteErr] = useState(null);
  const routeDebounce = useRef(null);

  function pickPoint(pt) {
    setRoutePts((pts) => {
      if (pts.length >= 2) return pts;
      const next = [...pts, pt];
      if (next.length === 2) setPicking(false);
      return next;
    });
  }
  function clearRoute() {
    setPicking(false); setRoutePts([]); setRouteRes(null); setRouteErr(null);
  }
  useEffect(() => {                                    // (re)route on points / rain / area
    if (routePts.length !== 2) return;
    if (routeDebounce.current) clearTimeout(routeDebounce.current);
    routeDebounce.current = setTimeout(() => {
      setRouteErr(null);
      api.route(routePts[0], routePts[1], rain, area)
        .then(setRouteRes)
        .catch((e) => { setRouteRes(null); setRouteErr(String(e)); });
    }, 400);
    return () => clearTimeout(routeDebounce.current);
  }, [routePts, rain, area]);
  useEffect(() => { clearRoute(); }, [view]);

  useEffect(() => {
    api.areas().then((list) => {
      setAreas(list);
      const first = list.find((a) => a.built) || list[0];
      if (first) setView(first.id);
    }).catch((e) => setErr(String(e)));
  }, []);

  // (re)load layers whenever the area changes; clear derived results
  useEffect(() => {
    if (!area) return;
    setErr(null);
    setCanal(null); setFlood(null); setStorage(null); setDig(null);
    setCost(null); setExposure(null); setReport(null);
    setWb(null); setStoragePlan(null); setNl(null); setAdv(null); setLearn(null);
    setReports([]); setReporting(false); setReportDraft(null); setReportStatus(null);
    api.meta(area).then(setMeta).catch((e) => setErr(String(e)));
    api.sinks(area).then(setSinks).catch(() => setSinks(null));
    api.recharge(area).then(setRecharge).catch(() => setRecharge(null));
    api.alerts(area).then(setAlerts).catch(() => setAlerts([]));
    api.validation(area).then(setValidation).catch(() => setValidation(null));
    api.canalPlan(area).then(setCanal).catch(() => setCanal(null));
    api.storagePlan(area).then(setStoragePlan).catch(() => setStoragePlan(null));
    api.nightlights(area).then(setNl).catch(() => setNl(null));
    api.advisory(area).then(setAdv).catch(() => setAdv(null));
    api.learningLog(area).then(setLearn).catch(() => setLearn(null));
  }, [area]);

  useEffect(() => {
    if (!area) return;
    if (debounce.current) clearTimeout(debounce.current);
    debounce.current = setTimeout(() => {
      setBusy(true); setErr(null);
      api.whatif(rain, null, area).then((r) => setFlood(r)).catch((e) => setErr(String(e))).finally(() => setBusy(false));
    }, 350);
    return () => clearTimeout(debounce.current);
  }, [rain, area]);

  // water budget follows the storm slider (read-only interpolation — cheap)
  useEffect(() => {
    if (!area) return;
    if (wbDebounce.current) clearTimeout(wbDebounce.current);
    wbDebounce.current = setTimeout(() => {
      api.waterbalance(area, rain, eff).then(setWb).catch(() => setWb(null));
    }, 350);
    return () => clearTimeout(wbDebounce.current);
  }, [rain, area, eff]);

  // citizen reports: always polling (60 s) so new pins appear for everyone
  const refreshReports = () => {
    if (area) api.reports(area).then((r) => setReports(r.reports)).catch(() => {});
  };
  usePolling(refreshReports, 60_000, !!area, [area]);

  // LIVE mode: forecast + live alerts every 15 min; forecast drives the slider
  usePolling(() => {
    if (!area) return;
    api.weather(area).then((w) => {
      setWeather(w);
      setRain(Math.max(0, Math.min(250, Math.round(w.rain_24h_mm / 5) * 5)));
    }).catch(() => {});
    api.alertsLive(area).then(setLiveAlerts).catch(() => setLiveAlerts(null));
  }, 900_000, live && !!area, [area]);

  // city aggregate view
  usePolling(() => {
    if (cityId) api.city(cityId).then(setCitySummary).catch((e) => setErr(String(e)));
  }, 300_000, !!cityId, [cityId]);

  async function submitReport(band, note) {
    if (!reportDraft || !area) return;
    try {
      const hp = document.getElementById("website")?.value || "";
      const r = await api.reportFlood({
        area, lat: reportDraft[0], lon: reportDraft[1], depth_band: band,
        note: note || null, client_ts: new Date().toISOString(),
        ...(hp ? { website: hp } : {}),
      });
      setReportStatus({ ok: true, msg: "Thanks — your report is live on the map and will teach tonight's model." });
      setReportDraft(null); setReporting(false);
      setReports((rs) => [{ ...r }, ...rs]);
    } catch (e) {
      const s = String(e);
      setReportStatus({
        ok: false,
        msg: s.includes("429") ? "Too many reports from your connection — please wait a while."
          : s.includes("outside_area") ? "That pin is outside this area's model window."
          : s,
      });
    }
  }

  const cityGroups = [...new Set(areas.map((a) => a.city).filter(Boolean))];
  const mumbaiTiles = areas.filter((a) => a.city && citySummary && a.city === cityId);
  const center = isCity ? [19.10, 72.90] : (meta ? meta.center : [25.605, 85.14]);
  const toggle = (k) => setShow((s) => ({ ...s, [k]: !s[k] }));

  // run a heavy endpoint with a shared busy/err guard
  async function run(fn, set) {
    setBusy(true); setErr(null);
    try { set(await fn()); } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  }
  const runCanals = () => run(() => api.canals(rain, 3, area), setCanal);
  const runStorage = () => run(() => api.storage(rain, area), setStorage);
  const runDig = () => run(() => api.optimize(rain, 150000, area), setDig);
  const runCost = () => run(() => api.costbenefit(rain, area), setCost);
  const runExposure = () => run(() => api.exposure(rain, area), setExposure);
  const runReport = () => run(() => api.report(rain, area), setReport);

  async function sendChat() {
    if (!msg.trim()) return;
    const m = msg; setMsg(""); setChat((c) => [...c, { role: "user", text: m }]);
    try {
      const r = await api.chat(m, chat.map((c) => ({ role: c.role, content: c.text })), area);
      setChat((c) => [...c, { role: "assistant", text: r.reply }]);
    } catch (e) {
      setChat((c) => [...c, { role: "assistant", text: `(LLM unavailable: ${e})` }]);
    }
  }

  const cityTileColor = (ti) => {
    const v = ti.summary ? ti.summary.flooded_volume_m3 : 0;
    const max = Math.max(1, ...(citySummary?.tiles || []).map((x) => x.summary?.flooded_volume_m3 || 0));
    const f = Math.sqrt(v / max);
    return { color: "#3b82f6", weight: 1.5, fillColor: "#3b82f6", fillOpacity: 0.08 + 0.45 * f };
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>Varuna<span className="muted"> · FloodTwin</span></h1>

        <Panel title="Area">
          <select value={view || ""} onChange={(e) => setView(e.target.value)} style={{ width: "100%" }}>
            {cityGroups.map((cid) => (
              <option key={`city:${cid}`} value={`city:${cid}`}>
                ▦ {cid === "mumbai" ? "Mumbai (city) — all tiles" : `${cid} (city)`}
              </option>
            ))}
            {areas.map((a) => (
              <option key={a.id} value={a.id} disabled={!a.built}>
                {a.name}{a.built ? "" : " (not built)"}
              </option>
            ))}
          </select>
          {area && <p className="muted">{areas.find((a) => a.id === area)?.note}</p>}
        </Panel>

        {isCity && <CityPanel city={citySummary} onSelectTile={(tid) => setView(tid)} />}

        {!isCity && (
          <>
            <LivePanel live={live} setLive={setLive} weather={weather} liveAlerts={liveAlerts} />

            <ReportPanel reporting={reporting} setReporting={setReporting}
                         draft={reportDraft} setDraft={setReportDraft}
                         submit={submitReport} status={reportStatus} />

            <Panel title="Storm what-if">
              <label>Rainfall: <b>{rain} mm</b> / 24h {live && <span className="badge gnn">LIVE</span>}{" "}
                {busy && <span className="muted">…</span>}</label>
              <input type="range" min="0" max="250" step="5" value={rain}
                     onChange={(e) => { setLive(false); setRain(Number(e.target.value)); }} />
              {flood && (
                <div className="kv">
                  <div>Flooded area</div><div>{(flood.summary.flooded_area_m2 / 1e6).toFixed(2)} km²</div>
                  <div>Flooded volume</div><div>{(flood.summary.flooded_volume_m3 / 1e6).toFixed(2)} M m³</div>
                  <div>Peak depth</div><div>{flood.summary.peak_depth_m} m</div>
                </div>
              )}
              <p className="muted">Live U-Net emulator (milliseconds).</p>
            </Panel>

            <WaterBalancePanel wb={wb} plan={storagePlan} unitM3={unitM3} setUnitM3={setUnitM3}
                               eff={eff} setEff={setEff} rain={rain} />

            <Panel title="Layers">
              {Object.keys(show).map((k) => (
                <label key={k} className="chk"
                       title={k === "recharge" && recharge?.sample ? recharge.gw_status?.reason : undefined}>
                  <input type="checkbox" checked={show[k]} onChange={() => toggle(k)} /> {k}
                  {k === "recharge" && recharge?.sample && <span className="badge sample">sample</span>}
                </label>
              ))}
            </Panel>

            <Panel title={`Interventions @ ${rain} mm`}>
              <div className="row">
                <button onClick={runCanals} disabled={busy}>Canals</button>
                <button onClick={runStorage} disabled={busy}>Storage</button>
                <button onClick={runDig} disabled={busy}>Excavate</button>
              </div>
              {canal && (
                <p>
                  Drains: net cut <b>{canal.reduction_pct}%</b> · {canal.n_canals} to {canal.outfalls}
                  {canal.network && <> · <b>{canal.network.n_inlets}</b> street inlets ({(canal.network.total_length_m / 1000).toFixed(1)} km)</>}
                </p>
              )}
              {storage && storage.targets && (
                <p>Storage: {Object.entries(storage.targets).map(([k, v]) => `${k}≈${v.sites} sites`).join(", ")}</p>
              )}
              {dig && <p>Excavation: cut <b>{dig.reduction_pct}%</b> · {dig.total_excavation_m3.toLocaleString()} m³</p>}
            </Panel>

            <Panel title="Cost-benefit (₹ per m³ removed)">
              <button onClick={runCost} disabled={busy}>Rank interventions</button>
              {cost && cost.interventions && (
                <table className="cb">
                  <thead><tr><th>#</th><th>Move</th><th>₹/m³</th><th>cut</th><th>₹cr</th></tr></thead>
                  <tbody>
                    {cost.interventions.map((it) => (
                      <tr key={it.name}>
                        <td>{it.rank}</td><td>{it.name}</td>
                        <td>{it.cost_per_m3_reduced_inr ?? "—"}</td>
                        <td>{it.reduction_pct}%</td><td>{it.cost_crore_inr}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Panel>

            <Panel title="Exposure / evacuation">
              <button onClick={runExposure} disabled={busy}>Assess @ {rain} mm</button>
              {exposure && (
                <div className="kv">
                  <div>At-risk buildings</div>
                  <div>{exposure.buildings.at_risk} / {exposure.buildings.total_in_domain} ({exposure.buildings.at_risk_pct}%)</div>
                  <div>Flooded roads</div><div>{exposure.roads.flooded} / {exposure.roads.total}</div>
                  {exposure.cached && (<><div>Source</div><div>cached @ {exposure.rain_mm} mm</div></>)}
                </div>
              )}
            </Panel>

            <Panel title="Evacuation route (GNN)">
              <div className="row">
                <button onClick={() => { clearRoute(); setPicking(true); }}>
                  {picking ? `Click map: ${routePts.length ? "destination" : "start"}…` : "Pick start & end"}
                </button>
                {(routePts.length > 0 || routeRes) && <button onClick={clearRoute}>Clear</button>}
              </div>
              {routeRes && (
                <>
                  <p>
                    <span className={`badge ${routeRes.backend}`}>{routeRes.backend === "gnn" ? "GNN" : "emulator"}</span>
                    {" "}<b>{(routeRes.route.length_m / 1000).toFixed(1)} km</b>
                    {" "}(+{routeRes.detour_pct}% vs shortest)
                  </p>
                  <div className="kv">
                    <div>Flooded street crossed</div>
                    <div>{routeRes.route.wet_length_m} m <span className="muted">vs {routeRes.shortest.wet_length_m} m</span></div>
                    <div>Max depth on route</div><div>{routeRes.route.max_depth_m} m</div>
                  </div>
                </>
              )}
              {routeErr && <p className="err">{routeErr}</p>}
              <p className="muted">Streets scored by learned flood risk at {rain} mm; drag the rainfall slider to re-plan.</p>
            </Panel>

            <Panel title="AI plan report">
              <button onClick={runReport} disabled={busy}>Generate report</button>
              {report && (
                <>
                  <p className="muted">backend: {report.backend}</p>
                  <pre className="report">{report.markdown}</pre>
                </>
              )}
            </Panel>

            <AdvisoryPanel adv={adv} onRefresh={() => api.advisory(area).then(setAdv).catch(() => {})} />
            <NightLightsPanel nl={nl} />

            <ValidationPanel v={validation} learn={learn} />

            <Panel title="Ask the model">
              <div className="chat">
                {chat.map((c, i) => <div key={i} className={`bubble ${c.role}`}>{c.text}</div>)}
              </div>
              <div className="row">
                <input value={msg} placeholder="e.g. where should we dig?"
                       onChange={(e) => setMsg(e.target.value)} onKeyDown={(e) => e.key === "Enter" && sendChat()} />
                <button onClick={sendChat}>Send</button>
              </div>
            </Panel>
          </>
        )}

        {err && <p className="err">{err}</p>}
      </aside>

      <main className="map">
        <MapContainer center={center} zoom={12} style={{ height: "100%", width: "100%" }}>
          <Recenter center={center} zoom={isCity ? 11 : undefined} />
          <MapClicks reporting={reporting} picking={picking} onPick={pickPoint}
                     onReportPin={(pt) => { setReportDraft(pt); setReporting(false); }} />
          <TileLayer attribution="© OpenStreetMap"
                     url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

          {/* city aggregate: one rectangle per tile, shaded by flooded volume */}
          {isCity && citySummary && citySummary.tiles.filter((t) => t.built && t.bounds).map((t) => (
            <Rectangle key={t.id} bounds={t.bounds} pathOptions={cityTileColor(t)}
                       eventHandlers={{ click: () => setView(t.id) }}>
              <Tooltip>
                {t.name}{t.summary ? ` · ${(t.summary.flooded_area_m2 / 1e6).toFixed(1)} km² flooded @ ${Math.round(t.rain_mm)} mm` : ""}
                {t.reports_24h ? ` · ${t.reports_24h} reports` : ""}
              </Tooltip>
            </Rectangle>
          ))}

          {!isCity && meta && <Rectangle bounds={meta.domain_bounds} pathOptions={{ color: "#888", weight: 1, fill: false, dashArray: "4" }} />}

          {!isCity && show.flood && flood && (
            <ImageOverlay url={flood.overlay_png} bounds={flood.bounds} opacity={0.75} />
          )}

          {/* night-lights power outages (red = dark vs 90-day normal) */}
          {!isCity && show.nightlights && nl && nl.bounds && (
            <ImageOverlay url={api.imageUrl("nightlights.png", area)} bounds={nl.bounds} opacity={0.8} />
          )}

          {/* citizen reports — sequential blues by depth, white ring for map contrast */}
          {!isCity && show.reports && reports.map((r) => (
            <CircleMarker key={r.id} center={[r.lat, r.lon]} radius={7}
                          pathOptions={{ color: "#ffffff", weight: 2,
                                         fillColor: REPORT_COLOR[r.depth_band] || "#3b82f6", fillOpacity: 0.95 }}>
              <Tooltip>
                {r.depth_band}-deep ({r.depth_m} m) · {new Date(r.ts).toLocaleTimeString()}
                {r.note ? <><br />“{r.note}”</> : null}
              </Tooltip>
            </CircleMarker>
          ))}
          {!isCity && reportDraft && (
            <CircleMarker center={reportDraft} radius={9}
                          pathOptions={{ color: "#3aa0ff", dashArray: "3", fillColor: "#3aa0ff", fillOpacity: 0.4 }}>
              <Tooltip permanent>your report</Tooltip>
            </CircleMarker>
          )}

          {!isCity && show.roads && exposure && exposure.roads.dry_lines.map((ln, i) => (
            <Polyline key={`dry${i}`} positions={ln} pathOptions={{ color: "#2a7", weight: 1.5, opacity: 0.5 }} />
          ))}
          {!isCity && show.roads && exposure && exposure.roads.flooded_lines.map((ln, i) => (
            <Polyline key={`fl${i}`} positions={ln} pathOptions={{ color: "#c30", weight: 2, opacity: 0.7 }} />
          ))}
          {!isCity && show.buildings && exposure && exposure.buildings.points.map((b, i) => (
            <CircleMarker key={`b${i}`} center={[b.lat, b.lon]} radius={3}
                          pathOptions={{ color: "#900", fillColor: "#e33", fillOpacity: 0.8, weight: 0 }}>
              <Tooltip>at-risk building · {b.depth_m} m</Tooltip>
            </CircleMarker>
          ))}

          {!isCity && show.alerts && alerts.map((a, i) => (
            <CircleMarker key={i} center={[a.lat, a.lon]} radius={6}
                          pathOptions={{ color: LEVEL_COLOR[a.level] || "#39c", fillOpacity: 0.8 }}>
              <Tooltip>sink {a.sink_id}: {a.level} (fill {a.fill_ratio})</Tooltip>
            </CircleMarker>
          ))}

          {!isCity && show.sinks && sinks && (
            <GeoJSON key={area} data={sinks}
                     pointToLayer={(f, latlng) => L.circleMarker(latlng, { radius: 3, color: "#36c" })} />
          )}

          {!isCity && show.recharge && recharge && recharge.features.map((f, i) => (
            <CircleMarker key={i} center={[f.geometry.coordinates[1], f.geometry.coordinates[0]]}
                          radius={4 + 6 * (f.properties.rsi || 0)}
                          pathOptions={recharge.sample
                            ? { color: "#888", fillOpacity: 0.3, dashArray: "3 3" }
                            : { color: "#2a7", fillOpacity: 0.6 }}>
              <Tooltip>recharge RSI {Number(f.properties.rsi).toFixed(2)}
                {recharge.sample ? " — SAMPLE groundwater input; ranking not meaningful" : ""}</Tooltip>
            </CircleMarker>
          ))}

          {/* storm-drain spiderweb along real streets: trunk thick, branches thin (flow-weighted) */}
          {!isCity && show.canal && canal && canal.network && canal.network.edges.map((e, i) => (
            <Polyline key={`net${i}`} positions={e.path_latlon}
                      pathOptions={{ color: "#7b2fbe", weight: 2 + 5 * e.weight, opacity: 0.9 }}>
              <Tooltip>storm drain · carries {e.drained_m3.toLocaleString()} m³</Tooltip>
            </Polyline>
          ))}
          {!isCity && show.canal && canal && canal.network && canal.network.inlets.map((p, i) => (
            <CircleMarker key={`in${i}`} center={p.latlon} radius={4}
                          pathOptions={{ ...C.inlet, fillOpacity: 0.95, weight: 1.5 }}>
              <Tooltip>drain inlet · collects {p.drained_m3.toLocaleString()} m³</Tooltip>
            </CircleMarker>
          ))}
          {!isCity && show.canal && canal && canal.network && canal.network.outfall_points.map((o, i) => (
            <CircleMarker key={`of${i}`} center={o.latlon} radius={7}
                          pathOptions={{ ...(OUTFALL_STYLE[o.kind] || C.boundary), fillOpacity: 0.95, weight: 2 }}>
              <Tooltip>outfall → {o.kind === "lowland" ? "safe low ground" : o.kind === "pit" ? "storage pit" : o.kind}</Tooltip>
            </CircleMarker>
          ))}
          {/* legacy single-line canals (bundles without a road graph) */}
          {!isCity && show.canal && canal && !canal.network && canal.canals && canal.canals.map((c, i) => (
            <Polyline key={i} positions={c.path_latlon} pathOptions={{ color: "#7b2fbe", weight: 3 }}>
              <Tooltip>canal {Math.round(c.length_m)} m → outfall</Tooltip>
            </Polyline>
          ))}
          {!isCity && show.canal && canal && canal.storage_sites && canal.storage_sites.map((s, i) => (
            <CircleMarker key={`p${i}`} center={s.latlon} radius={5}
                          pathOptions={{ ...C.pit, fillOpacity: 0.9, weight: 1.5 }}>
              <Tooltip>storage pit {s.excavation_m3.toLocaleString()} m³</Tooltip>
            </CircleMarker>
          ))}

          {/* planned container sites: buildable detention cells, deepest (rank 1) first */}
          {!isCity && show.containers && storagePlan && storagePlan.sites &&
            storagePlan.sites.filter((s) => s.latlon).slice(0, 800).map((s) => (
            <CircleMarker key={`ct${s.rank}`} center={s.latlon}
                          radius={Math.max(2.5, Math.min(9, 1.5 * Math.sqrt(s.site_m3 / 500)))}
                          pathOptions={{ color: "#0369a1", fillColor: "#0ea5e9",
                                         fillOpacity: 0.75, weight: 1 }}>
              <Tooltip>container #{s.rank} · {s.site_m3.toLocaleString()} m³
                {s.on_road ? " · under street" : ""}</Tooltip>
            </CircleMarker>
          ))}

          {!isCity && show.dig && dig && dig.dig_plan && dig.dig_plan.map((s, i) => (
            <CircleMarker key={`d${i}`} center={[s.lat, s.lon]} radius={4 + 3 * s.dig_depth_m}
                          pathOptions={{ color: "#630", fillColor: "#c96", fillOpacity: 0.85 }}>
              <Tooltip>dig {s.dig_depth_m} m ({s.excavation_m3.toLocaleString()} m³)</Tooltip>
            </CircleMarker>
          ))}

          {/* evacuation route: white casing keeps the line readable over the flood overlay */}
          {!isCity && show.route && routeRes && (
            <>
              <Polyline positions={routeRes.shortest.path_latlon}
                        pathOptions={{ color: C.shortest, weight: 3, opacity: 0.7, dashArray: "6 6" }} />
              <Polyline positions={routeRes.route.path_latlon}
                        pathOptions={{ color: C.routeCasing, weight: 7, opacity: 0.9 }} />
              <Polyline positions={routeRes.route.path_latlon}
                        pathOptions={{ color: C.route, weight: 4, opacity: 0.95 }}>
                <Tooltip>flood-safe route · {(routeRes.route.length_m / 1000).toFixed(1)} km · max {routeRes.route.max_depth_m} m</Tooltip>
              </Polyline>
            </>
          )}
          {!isCity && show.route && routePts.map((p, i) => (
            <CircleMarker key={`rp${i}`} center={p} radius={7}
                          pathOptions={{ color: C.route, fillColor: i === 0 ? "#fff" : C.route,
                                         fillOpacity: 1, weight: 3 }}>
              <Tooltip>{i === 0 ? "start" : "destination"}</Tooltip>
            </CircleMarker>
          ))}
        </MapContainer>
        {!isCity && (
          <Legend show={show} canal={canal} alerts={alerts} exposure={exposure} dig={dig}
                  route={show.route && routeRes} reports={reports} nightlights={nl} />
        )}
      </main>
    </div>
  );
}
