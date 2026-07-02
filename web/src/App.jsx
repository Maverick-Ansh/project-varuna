import React, { useEffect, useRef, useState } from "react";
import {
  MapContainer, TileLayer, GeoJSON, CircleMarker, Polyline, ImageOverlay, Rectangle, Tooltip, useMap,
} from "react-leaflet";
import L from "leaflet";
import { api } from "./api.js";

const LEVEL_COLOR = { RED: "#e23", AMBER: "#f90", GREEN: "#2a4" };

function Panel({ title, children }) {
  return (
    <div className="panel">
      <h3>{title}</h3>
      {children}
    </div>
  );
}

function Recenter({ center }) {
  const map = useMap();
  useEffect(() => {
    if (center) map.setView(center);
  }, [center && center[0], center && center[1]]);
  return null;
}

function ValidationPanel({ v }) {
  if (!v) return <Panel title="Validation"><p className="muted">loading…</p></Panel>;
  const stat = v.static_depth_vs_sar;
  const rep = v.calibration_report;
  const fmt = (x) => (x == null ? "—" : Number(x).toFixed(3));
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
    </Panel>
  );
}

export default function App() {
  const [areas, setAreas] = useState([]);
  const [area, setArea] = useState(null);

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

  const [show, setShow] = useState({
    flood: true, alerts: true, sinks: false, recharge: false, canal: true, dig: true,
    buildings: true, roads: true,
  });
  const [chat, setChat] = useState([]);
  const [msg, setMsg] = useState("");
  const debounce = useRef(null);

  useEffect(() => {
    api.areas().then((list) => {
      setAreas(list);
      const first = list.find((a) => a.built) || list[0];
      if (first) setArea(first.id);
    }).catch((e) => setErr(String(e)));
  }, []);

  // (re)load layers whenever the area changes; clear derived results
  useEffect(() => {
    if (!area) return;
    setErr(null);
    setCanal(null); setFlood(null); setStorage(null); setDig(null);
    setCost(null); setExposure(null); setReport(null);
    api.meta(area).then(setMeta).catch((e) => setErr(String(e)));
    api.sinks(area).then(setSinks).catch(() => setSinks(null));
    api.recharge(area).then(setRecharge).catch(() => setRecharge(null));
    api.alerts(area).then(setAlerts).catch(() => setAlerts([]));
    api.validation(area).then(setValidation).catch(() => setValidation(null));
    api.canalPlan(area).then(setCanal).catch(() => setCanal(null));
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

  const center = meta ? meta.center : [25.605, 85.14];
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

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>Varuna<span className="muted"> · FloodTwin</span></h1>

        <Panel title="Area">
          <select value={area || ""} onChange={(e) => setArea(e.target.value)} style={{ width: "100%" }}>
            {areas.map((a) => (
              <option key={a.id} value={a.id} disabled={!a.built}>
                {a.name}{a.built ? "" : " (not built)"}
              </option>
            ))}
          </select>
          {area && <p className="muted">{areas.find((a) => a.id === area)?.note}</p>}
        </Panel>

        <Panel title="Storm what-if">
          <label>Rainfall: <b>{rain} mm</b> / 24h {busy && <span className="muted">…</span>}</label>
          <input type="range" min="0" max="250" step="5" value={rain}
                 onChange={(e) => setRain(Number(e.target.value))} />
          {flood && (
            <div className="kv">
              <div>Flooded area</div><div>{(flood.summary.flooded_area_m2 / 1e6).toFixed(2)} km²</div>
              <div>Flooded volume</div><div>{(flood.summary.flooded_volume_m3 / 1e6).toFixed(2)} M m³</div>
              <div>Peak depth</div><div>{flood.summary.peak_depth_m} m</div>
            </div>
          )}
          <p className="muted">Live U-Net emulator (milliseconds).</p>
        </Panel>

        <Panel title="Layers">
          {Object.keys(show).map((k) => (
            <label key={k} className="chk">
              <input type="checkbox" checked={show[k]} onChange={() => toggle(k)} /> {k}
            </label>
          ))}
        </Panel>

        <Panel title={`Interventions @ ${rain} mm`}>
          <div className="row">
            <button onClick={runCanals} disabled={busy}>Canals</button>
            <button onClick={runStorage} disabled={busy}>Storage</button>
            <button onClick={runDig} disabled={busy}>Excavate</button>
          </div>
          {canal && <p>Canals: net cut <b>{canal.reduction_pct}%</b> · {canal.n_canals} to {canal.outfalls}</p>}
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

        <Panel title="AI plan report">
          <button onClick={runReport} disabled={busy}>Generate report</button>
          {report && (
            <>
              <p className="muted">backend: {report.backend}</p>
              <pre className="report">{report.markdown}</pre>
            </>
          )}
        </Panel>

        <ValidationPanel v={validation} />

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

        {err && <p className="err">{err}</p>}
      </aside>

      <main className="map">
        <MapContainer center={center} zoom={12} style={{ height: "100%", width: "100%" }}>
          <Recenter center={center} />
          <TileLayer attribution="© OpenStreetMap"
                     url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

          {meta && <Rectangle bounds={meta.domain_bounds} pathOptions={{ color: "#888", weight: 1, fill: false, dashArray: "4" }} />}

          {show.flood && flood && (
            <ImageOverlay url={flood.overlay_png} bounds={flood.bounds} opacity={0.75} />
          )}

          {show.roads && exposure && exposure.roads.dry_lines.map((ln, i) => (
            <Polyline key={`dry${i}`} positions={ln} pathOptions={{ color: "#2a7", weight: 1.5, opacity: 0.5 }} />
          ))}
          {show.roads && exposure && exposure.roads.flooded_lines.map((ln, i) => (
            <Polyline key={`fl${i}`} positions={ln} pathOptions={{ color: "#c30", weight: 2, opacity: 0.7 }} />
          ))}
          {show.buildings && exposure && exposure.buildings.points.map((b, i) => (
            <CircleMarker key={`b${i}`} center={[b.lat, b.lon]} radius={3}
                          pathOptions={{ color: "#900", fillColor: "#e33", fillOpacity: 0.8, weight: 0 }}>
              <Tooltip>at-risk building · {b.depth_m} m</Tooltip>
            </CircleMarker>
          ))}

          {show.alerts && alerts.map((a, i) => (
            <CircleMarker key={i} center={[a.lat, a.lon]} radius={6}
                          pathOptions={{ color: LEVEL_COLOR[a.level] || "#39c", fillOpacity: 0.8 }}>
              <Tooltip>sink {a.sink_id}: {a.level} (fill {a.fill_ratio})</Tooltip>
            </CircleMarker>
          ))}

          {show.sinks && sinks && (
            <GeoJSON key={area} data={sinks}
                     pointToLayer={(f, latlng) => L.circleMarker(latlng, { radius: 3, color: "#36c" })} />
          )}

          {show.recharge && recharge && recharge.features.map((f, i) => (
            <CircleMarker key={i} center={[f.geometry.coordinates[1], f.geometry.coordinates[0]]}
                          radius={4 + 6 * (f.properties.rsi || 0)}
                          pathOptions={{ color: "#2a7", fillOpacity: 0.6 }}>
              <Tooltip>recharge RSI {Number(f.properties.rsi).toFixed(2)}</Tooltip>
            </CircleMarker>
          ))}

          {show.canal && canal && canal.canals && canal.canals.map((c, i) => (
            <Polyline key={i} positions={c.path_latlon} pathOptions={{ color: "#e23", weight: 3 }}>
              <Tooltip>canal {Math.round(c.length_m)} m → outfall</Tooltip>
            </Polyline>
          ))}
          {show.canal && canal && canal.storage_sites && canal.storage_sites.map((s, i) => (
            <CircleMarker key={`p${i}`} center={s.latlon} radius={5}
                          pathOptions={{ color: "#093", fillColor: "#3e6", fillOpacity: 0.9 }}>
              <Tooltip>storage pit {s.excavation_m3.toLocaleString()} m³</Tooltip>
            </CircleMarker>
          ))}

          {show.dig && dig && dig.dig_plan && dig.dig_plan.map((s, i) => (
            <CircleMarker key={`d${i}`} center={[s.lat, s.lon]} radius={4 + 3 * s.dig_depth_m}
                          pathOptions={{ color: "#630", fillColor: "#c96", fillOpacity: 0.85 }}>
              <Tooltip>dig {s.dig_depth_m} m ({s.excavation_m3.toLocaleString()} m³)</Tooltip>
            </CircleMarker>
          ))}
        </MapContainer>
      </main>
    </div>
  );
}
