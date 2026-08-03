import React, { useMemo, useState } from "react";
import { CircleMarker, Marker, TileLayer, Tooltip, useMapEvents } from "react-leaflet";
import L from "leaflet";

// --- basemaps -------------------------------------------------------------------------------
// Satellite is what makes a zoomed-in flood legible: on a street map an arrow sits in white
// space, on imagery it sits on a visible road between visible buildings.
export const BASEMAPS = {
  map: {
    label: "Map",
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: "© OpenStreetMap",
    maxZoom: 19,
  },
  satellite: {
    label: "Satellite",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
    maxZoom: 19,
  },
};

// Zoom at which "auto" flips to imagery — roughly where individual streets become readable.
export const SATELLITE_ZOOM = 15;

export function useMapZoom(initial = 12) {
  const [zoom, setZoom] = useState(initial);
  useMapEvents({ zoomend: (e) => setZoom(e.target.getZoom()) });
  return zoom;
}

export function Basemap({ mode, zoom }) {
  const key = mode === "auto" ? (zoom >= SATELLITE_ZOOM ? "satellite" : "map") : mode;
  const b = BASEMAPS[key] || BASEMAPS.map;
  // `key` forces a real layer swap rather than a URL mutation on the same tile layer
  return <TileLayer key={key} url={b.url} attribution={b.attribution} maxZoom={b.maxZoom} />;
}

// --- flow arrows ----------------------------------------------------------------------------

// Sequential blues by depth, matching the citizen-report ramp so the map has one depth language.
const DEPTH_COLORS = [
  [0.15, "#93c5fd"], [0.40, "#60a5fa"], [0.80, "#3b82f6"], [1.30, "#1d4ed8"], [Infinity, "#172554"],
];
export const depthColor = (m) => DEPTH_COLORS.find(([t]) => m < t)[1];

// Drawing every arrow at every zoom is a hairball at city scale and too sparse at street scale.
const arrowBudget = (zoom) => (zoom >= 16 ? 900 : zoom >= 15 ? 500 : zoom >= 14 ? 260 : zoom >= 13 ? 130 : 70);

// divIcons are DOM nodes, so the icon HTML is memoised across arrows that round to the same
// bearing/colour/size — a few dozen distinct icons instead of several hundred.
const iconCache = new Map();
function arrowIcon(bearing, color, size) {
  const b = Math.round(bearing / 5) * 5;
  const key = `${b}|${color}|${size}`;
  let icon = iconCache.get(key);
  if (!icon) {
    icon = L.divIcon({
      className: "flow-arrow",
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
      html:
        `<svg width="${size}" height="${size}" viewBox="0 0 24 24" ` +
        `style="transform:rotate(${b}deg)">` +
        `<path d="M12 21 L12 4 M12 3 L6.5 11 M12 3 L17.5 11" stroke="${color}" ` +
        `stroke-width="3.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    });
    iconCache.set(key, icon);
  }
  return icon;
}

/**
 * Water-direction arrows. `street=true` renders the road-projected set (bigger, named).
 *
 * Depths are shown with the emulator's held-out RMSE, never bare: a 60 m surrogate reporting
 * "690 mm" reads as a survey unless the error bar travels with it.
 */
export function FlowArrows({ arrows, zoom, street = false, rain, rmse }) {
  const shown = useMemo(() => (arrows || []).slice(0, arrowBudget(zoom) * (street ? 1 : 0.7)),
                        [arrows, zoom, street]);
  if (!shown.length) return null;
  const size = street ? (zoom >= 15 ? 26 : 20) : (zoom >= 15 ? 20 : 15);
  return (
    <>
      {shown.map((a, i) => (
        <Marker key={`${street ? "s" : "f"}${i}`} position={a.latlon} interactive
                icon={arrowIcon(a.bearing, depthColor(a.depth_m), size)}
                zIndexOffset={street ? 400 : 200}>
          <Tooltip direction="top" offset={[0, -6]}>
            <b>{a.street || (street ? "unnamed street" : "overland flow")}</b><br />
            {Math.round(a.depth_m * 1000).toLocaleString()} mm deep
            {rmse ? <span className="muted"> ± {Math.round(rmse * 1000)} mm</span> : null}
            {rain != null && <span className="muted"> @ {rain} mm rain</span>}<br />
            flowing {compass(a.bearing)} at {a.speed_ms.toFixed(2)} m/s
            {a.near_water && (
              <><br /><span className="warn-tip">next to permanent water — depth may be the
                lake's, not the street's</span></>
            )}
          </Tooltip>
        </Marker>
      ))}
    </>
  );
}

const POINTS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
export const compass = (deg) => POINTS[Math.round(((deg % 360) + 360) % 360 / 22.5) % 16];

// --- danger zones ---------------------------------------------------------------------------

export const SEVERITY = {
  EXTREME: { color: "#7f1d1d", fillColor: "#dc2626", r: 13 },
  HIGH: { color: "#7c2d12", fillColor: "#ea580c", r: 11 },
  MODERATE: { color: "#78350f", fillColor: "#f59e0b", r: 9 },
  LOW: { color: "#1e3a8a", fillColor: "#3b82f6", r: 7 },
  MINIMAL: { color: "#334155", fillColor: "#64748b", r: 6 },
};

/**
 * Ranked danger-zone pins. The top `labelTop` carry a PERMANENT label — the whole point is that
 * you can read the worst places without hovering anything — and the rest reveal on hover so the
 * map does not turn into a wall of text.
 */
export function DangerPins({ zones, labelTop = 5, rain }) {
  if (!zones || !zones.length) return null;
  return (
    <>
      {zones.map((z) => {
        const s = SEVERITY[z.severity] || SEVERITY.LOW;
        const permanent = z.rank <= labelTop;
        return (
          <CircleMarker key={`z${z.rank}`} center={z.latlon} radius={s.r}
                        pathOptions={{ color: s.color, fillColor: s.fillColor, fillOpacity: 0.85,
                                       weight: 2.5 }}>
            <Tooltip permanent={permanent} direction="top" offset={[0, -s.r - 2]}
                     className={`zone-tip sev-${z.severity.toLowerCase()}`}>
              <b>{z.place}</b>
              <span className="zone-sev"> {z.band}-deep</span>
              {!permanent && (
                <>
                  <br />peak {z.peak_depth_m} m · {z.built_area_km2} km² built
                  <br />{z.volume_m3.toLocaleString()} m³ standing
                </>
              )}
              {permanent && z.landmark && z.landmark !== z.place_name && (
                <span className="zone-near"> · near {z.landmark}</span>
              )}
              {rain != null && !permanent && (
                <><br /><span className="muted">at {rain} mm / 24 h</span></>
              )}
            </Tooltip>
          </CircleMarker>
        );
      })}
    </>
  );
}

/** Sidebar list of the ranked zones — the same data as the pins, readable without the map. */
export function DangerPanel({ zones, rain, onFly }) {
  if (!zones) return <p className="muted">move the rainfall slider…</p>;
  if (!zones.length) return <p className="muted">no danger zones at {rain} mm.</p>;
  return (
    <table className="cb zones">
      <thead><tr><th>#</th><th>Place</th><th>Depth</th><th>m³</th></tr></thead>
      <tbody>
        {zones.map((z) => (
          <tr key={z.rank} onClick={() => onFly && onFly(z)} title={
            `${z.severity} · peak ${z.peak_depth_m} m · ${z.built_area_km2} km² built` +
            (z.landmark ? ` · near ${z.landmark}` : "")}>
            <td>{z.rank}</td>
            <td>
              <i className="sw" style={{ background: (SEVERITY[z.severity] || SEVERITY.LOW).fillColor }} />
              {z.place}{!z.named && <span className="muted"> ?</span>}
            </td>
            <td>{z.peak_depth_m} m</td>
            <td>{Math.round(z.volume_m3 / 1000).toLocaleString()}k</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// --- depth validation ------------------------------------------------------------------------

/**
 * The twin scored against REAL reported depths (DEPTH_VALIDATION.md). This panel exists to put
 * the unflattering number in front of anyone using the map: the model currently has NEGATIVE
 * skill on point depths, and a dashboard that shows a confident blue overlay without saying so
 * is misleading by omission.
 */
export function DepthValidationPanel({ v, area }) {
  if (!v) return <p className="muted">no depth validation run yet.</p>;
  const m = (v.area_metrics && v.area_metrics.n ? v.area_metrics : v.overall) || {};
  const scope = v.area_metrics && v.area_metrics.n ? area : "all areas";
  if (!m.n) return <p className="muted">no observations scored.</p>;
  const skill = m.skill_vs_best_baseline;
  const good = skill > 0;
  const d = v.subgrid_dilution;
  return (
    <>
      <div className="kv">
        <div>Observations</div><div>{m.n} <span className="muted">({scope})</span></div>
        <div>Mean abs error</div><div>{(m.model.mae_m * 100).toFixed(0)} cm</div>
        <div>Bias</div>
        <div>{m.model.bias_m > 0 ? "+" : ""}{(m.model.bias_m * 100).toFixed(0)} cm
          <span className="muted"> {m.model.bias_m < 0 ? "(under-predicts)" : "(over-predicts)"}</span></div>
        <div>Skill</div>
        <div><b style={{ color: good ? "#2a4" : "#f77" }}>{skill}</b>
          <span className="muted"> {good ? "beats" : "worse than"} a flat guess</span></div>
        <div>Rank correlation</div><div>{m.correlation}</div>
      </div>
      {d && (
        <p className="muted" style={{ marginTop: 8 }}>
          Depths read <b>{d.median_observed_over_predicted}×</b> shallower than reported: a 60 m
          cell averages a pond of about <b>{d.implied_ponding_area_m2.toLocaleString()} m²</b>
          {" "}({d.implied_ponding_span_m} m across) over {d.cell_area_m2.toLocaleString()} m² of ground.
        </p>
      )}
      <p className="muted" style={{ marginBottom: 0 }}>
        Scored against real crowdsourced depth reports, forced with the rain that actually fell.
        Negative skill means the depth numbers on this map are not yet trustworthy per street —
        treat the map as relative risk, not measurement. See DEPTH_VALIDATION.md.
      </p>
    </>
  );
}
