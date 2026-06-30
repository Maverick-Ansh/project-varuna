import React, { useEffect, useRef, useState } from "react";
import {
  MapContainer, TileLayer, GeoJSON, CircleMarker, Polyline, ImageOverlay, Rectangle, Tooltip,
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
      ) : <p className="muted">Run the SAR calibration to populate calibrated CSI.</p>}
      {Object.keys(v.dynamic_twin_vs_sar || {}).length > 0 && (
        <p className="muted">dynamic-twin scored dates: {Object.keys(v.dynamic_twin_vs_sar).join(", ")}</p>
      )}
    </Panel>
  );
}

export default function App() {
  const [meta, setMeta] = useState(null);
  const [sinks, setSinks] = useState(null);
  const [recharge, setRecharge] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [validation, setValidation] = useState(null);
  const [canal, setCanal] = useState(null);

  const [rain, setRain] = useState(100);
  const [flood, setFlood] = useState(null);     // {overlay_png, bounds, summary}
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  const [show, setShow] = useState({ flood: true, alerts: true, sinks: false, recharge: false, canal: true });
  const [chat, setChat] = useState([]);
  const [msg, setMsg] = useState("");
  const debounce = useRef(null);

  useEffect(() => {
    api.meta().then(setMeta).catch((e) => setErr(String(e)));
    api.sinks().then(setSinks).catch(() => {});
    api.recharge().then(setRecharge).catch(() => {});
    api.alerts().then(setAlerts).catch(() => {});
    api.validation().then(setValidation).catch(() => {});
    api.canalPlan().then(setCanal).catch(() => {});
  }, []);

  // live what-if on rainfall change (debounced)
  useEffect(() => {
    if (debounce.current) clearTimeout(debounce.current);
    debounce.current = setTimeout(() => {
      setBusy(true); setErr(null);
      api.whatif(rain).then((r) => setFlood(r)).catch((e) => setErr(String(e))).finally(() => setBusy(false));
    }, 350);
    return () => clearTimeout(debounce.current);
  }, [rain]);

  const center = meta ? meta.center : [25.605, 85.14];
  const toggle = (k) => setShow((s) => ({ ...s, [k]: !s[k] }));

  async function runCanals() {
    setBusy(true);
    try { setCanal(await api.canals(rain, 3)); } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  }
  async function sendChat() {
    if (!msg.trim()) return;
    const m = msg; setMsg(""); setChat((c) => [...c, { role: "user", text: m }]);
    try {
      const r = await api.chat(m, chat.map((c) => ({ role: c.role, content: c.text })));
      setChat((c) => [...c, { role: "assistant", text: r.reply }]);
    } catch (e) {
      setChat((c) => [...c, { role: "assistant", text: `(LLM unavailable: ${e})` }]);
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>Varuna<span className="muted"> · Patna FloodTwin</span></h1>

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

        <Panel title="Interventions">
          <button onClick={runCanals} disabled={busy}>Plan canals @ {rain} mm</button>
          {canal && <p>net flood cut <b>{canal.reduction_pct}%</b> · {canal.n_canals} canals ({canal.outfalls || "pits"})</p>}
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
          <TileLayer attribution="© OpenStreetMap"
                     url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

          {meta && <Rectangle bounds={meta.domain_bounds} pathOptions={{ color: "#888", weight: 1, fill: false, dashArray: "4" }} />}

          {show.flood && flood && (
            <ImageOverlay url={flood.overlay_png} bounds={flood.bounds} opacity={0.75} />
          )}

          {show.alerts && alerts.map((a, i) => (
            <CircleMarker key={i} center={[a.lat, a.lon]} radius={6}
                          pathOptions={{ color: LEVEL_COLOR[a.level] || "#39c", fillOpacity: 0.8 }}>
              <Tooltip>sink {a.sink_id}: {a.level} (fill {a.fill_ratio})</Tooltip>
            </CircleMarker>
          ))}

          {show.sinks && sinks && (
            <GeoJSON data={sinks}
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
        </MapContainer>
      </main>
    </div>
  );
}
