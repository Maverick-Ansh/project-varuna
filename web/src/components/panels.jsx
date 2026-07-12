import React, { useState } from "react";

// v2 sidebar panels: live weather, citizen reports, water budget + storage designer,
// advisory, night lights, city aggregate. Chart colors validated for the dark panel
// surface (#18202e): nature-absorbed #16a34a, ponded #3b82f6, storable #ec4899.

export const WB = { absorbed: "#16a34a", ponded: "#3b82f6", storable: "#ec4899" };

export function Panel({ title, children }) {
  return (
    <div className="panel">
      <h3>{title}</h3>
      {children}
    </div>
  );
}

const fmtM3 = (v) => (v == null ? "—" : v >= 1e6 ? `${(v / 1e6).toFixed(2)} M m³`
  : v >= 1e3 ? `${(v / 1e3).toFixed(0)} k m³` : `${Math.round(v)} m³`);
const ago = (iso) => {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  return s < 90 ? "just now" : s < 5400 ? `${Math.round(s / 60)} min ago` : `${Math.round(s / 3600)} h ago`;
};

// ---------------------------------------------------------------- live weather

export function LivePanel({ live, setLive, weather, liveAlerts }) {
  const hy = weather?.hyetograph;
  const max = hy ? Math.max(0.2, ...hy.precip_mm) : 1;
  return (
    <Panel title="Live weather">
      <div className="row" style={{ marginTop: 0 }}>
        <button className={live ? "live-on" : ""} onClick={() => setLive(!live)}>
          {live ? "● LIVE — forecast drives the storm" : "Go live (real forecast)"}
        </button>
      </div>
      {live && weather && (
        <>
          <div className="kv">
            <div>Next 24 h (worst corner)</div><div><b>{weather.rain_24h_mm} mm</b></div>
            <div>Past 24 h (observed)</div><div>{weather.past24_mm} mm</div>
            <div>Next 48 h (centre)</div><div>{weather.rain_center_48h_mm} mm</div>
          </div>
          {hy && (
            <div className="hyeto" title="hourly forecast rain, next 48 h">
              {hy.precip_mm.map((v, i) => (
                <span key={i} style={{ height: `${Math.max(4, (v / max) * 100)}%`,
                                       opacity: v > 0 ? 1 : 0.25 }}
                      title={`${hy.times[i].slice(11, 16)} — ${v.toFixed(1)} mm`} />
              ))}
            </div>
          )}
          {liveAlerts && (
            <p style={{ margin: "6px 0 0" }}>
              Outlook: <b style={{ color: "#e23" }}>{liveAlerts.summary.red} RED</b> ·{" "}
              <b style={{ color: "#f90" }}>{liveAlerts.summary.amber} AMBER</b> sinks
              <span className="muted"> at {liveAlerts.forecast_rain_mm} mm</span>
            </p>
          )}
          <p className="muted" style={{ marginBottom: 0 }}>
            Open-Meteo, refreshed every 15 min · fetched {ago(weather.fetched_at)}
          </p>
        </>
      )}
      {!live && <p className="muted" style={{ marginBottom: 0 }}>
        Sets the rainfall slider from the real 24-h forecast and keeps it updated.</p>}
    </Panel>
  );
}

// ---------------------------------------------------------------- citizen reports

const BANDS = [
  ["ankle", "Ankle · 0.1 m"], ["knee", "Knee · 0.4 m"],
  ["waist", "Waist · 0.8 m"], ["chest", "Chest · 1.3 m"],
];

export function ReportPanel({ reporting, setReporting, draft, setDraft, submit, status }) {
  const [band, setBand] = useState("knee");
  const [note, setNote] = useState("");
  const reset = () => { setReporting(false); setDraft(null); setNote(""); };
  return (
    <Panel title="Report flooding">
      {!reporting && !draft && (
        <>
          <button onClick={() => setReporting(true)}>Report flooding here</button>
          <p className="muted" style={{ marginBottom: 0 }}>
            See water on your street? Drop a pin — reports appear on the map instantly and
            teach the model overnight.</p>
        </>
      )}
      {reporting && !draft && <p><b>Click the map</b> where the water is…{" "}
        <a href="#" onClick={(e) => { e.preventDefault(); reset(); }}>cancel</a></p>}
      {draft && (
        <>
          <p style={{ margin: 0 }}>Pin: {draft[0].toFixed(4)}, {draft[1].toFixed(4)}</p>
          <div className="depth-btns">
            {BANDS.map(([id, label]) => (
              <button key={id} className={band === id ? "sel" : "ghost"}
                      onClick={() => setBand(id)}>{label}</button>
            ))}
          </div>
          <input style={{ width: "100%", marginTop: 6 }} maxLength={280} value={note}
                 placeholder="optional note (e.g. 'rising fast near the market')"
                 onChange={(e) => setNote(e.target.value)} />
          {/* honeypot for form-filling bots — humans never see it */}
          <input type="text" tabIndex={-1} autoComplete="off" aria-hidden="true"
                 style={{ position: "absolute", left: -9999, width: 1, height: 1 }}
                 id="website" onChange={() => {}} />
          <div className="row">
            <button onClick={() => { submit(band, note); setNote(""); }}>Submit report</button>
            <button className="ghost" onClick={reset}>Cancel</button>
          </div>
        </>
      )}
      {status && <p className={status.ok ? "muted" : "err"} style={{ marginBottom: 0 }}>{status.msg}</p>}
    </Panel>
  );
}

// ------------------------------------------------- water budget + storage designer

export function WaterBalancePanel({ wb, plan, unitM3, setUnitM3, eff, setEff, rain }) {
  if (!wb) {
    return (
      <Panel title="Where does the rain go?">
        <p className="muted" style={{ marginBottom: 0 }}>
          No water-balance ladder in this bundle yet (built during the area build / nightly job).</p>
      </Panel>
    );
  }
  const total = Math.max(wb.rain_m3, 1);
  const segs = [
    ["absorbed by soil & vegetation", wb.infiltrated_m3, WB.absorbed],
    ["ponded on the surface", wb.ponded_final_m3, WB.ponded],
  ];
  const target = plan?.targets?.["50%"];
  const usable = target?.storage_m3 ? target.storage_m3 * eff : null;
  const units = (t) => (t?.storage_m3 ? Math.ceil(t.storage_m3 / (unitM3 * eff)) : null);
  return (
    <Panel title={`Where does the rain go? @ ${rain} mm`}>
      <div className="stack-bar" role="img"
           aria-label={`Of ${fmtM3(wb.rain_m3)} of rain, ${fmtM3(wb.infiltrated_m3)} is absorbed and ${fmtM3(wb.ponded_final_m3)} ponds`}>
        {segs.map(([label, v, c]) => (
          <div key={label} className="seg" title={`${label}: ${fmtM3(v)}`}
               style={{ width: `${(100 * v) / total}%`, background: c }} />
        ))}
      </div>
      <div className="lg-inline">
        {segs.map(([label, v, c]) => (
          <span key={label}><i style={{ background: c }} /> {label} <b>{fmtM3(v)}</b></span>
        ))}
      </div>
      {wb.reduced_by_nature_m3 > 0 && (
        <p style={{ margin: "6px 0 0" }}>
          Without green cover this storm would pond{" "}
          <b style={{ color: WB.absorbed }}>{fmtM3(wb.reduced_by_nature_m3)} more</b>{" "}
          <span className="muted">(all-concrete counterfactual)</span>
        </p>
      )}
      {wb.infil_by_class && (
        <div className="kv">
          <div>· vegetation (trees/shrub/crops)</div><div>{fmtM3(wb.infil_by_class.vegetation)}</div>
          <div>· built-up ground</div><div>{fmtM3(wb.infil_by_class.built)}</div>
          <div>· bare / other</div>
          <div>{fmtM3((wb.infil_by_class.bare || 0) + (wb.infil_by_class.other || 0))}</div>
        </div>
      )}

      {plan?.targets && (
        <>
          <h3 style={{ marginTop: 12 }}>Storage container designer</h3>
          {usable != null && (
            <>
              <div className="stack-bar thin" title="how much of the ponded water containers can hold">
                <div className="seg" style={{ width: "100%", background: "#26314a" }} />
                <div className="seg over" style={{
                  width: `${Math.min(100, (100 * usable) / Math.max(wb.ponded_final_m3, 1))}%`,
                  background: WB.storable }} />
              </div>
              <p style={{ margin: "4px 0 0" }}>
                <i className="sw" style={{ background: WB.storable }} /> Containers for the 50% plan
                hold <b>{fmtM3(usable)}</b> usable{" "}
                <span className="muted">({Math.round(eff * 100)}% of {fmtM3(target.storage_m3)} nominal)</span>
              </p>
            </>
          )}
          <label style={{ marginTop: 8 }}>Container unit size: <b>{unitM3} m³</b></label>
          <input type="range" min="10" max="500" step="10" value={unitM3}
                 onChange={(e) => setUnitM3(Number(e.target.value))} />
          <label>Efficiency factor: <b>{Math.round(eff * 100)}%</b>
            <span className="muted"> (silting / maintenance)</span></label>
          <input type="range" min="30" max="100" step="5" value={Math.round(eff * 100)}
                 onChange={(e) => setEff(Number(e.target.value) / 100)} />
          <table className="cb">
            <thead><tr><th>flood cut</th><th>sites</th><th>volume</th><th>≈ units</th></tr></thead>
            <tbody>
              {Object.entries(plan.targets).map(([pct, t]) => (
                <tr key={pct}>
                  <td>{pct}</td><td>{t.sites ?? "—"}</td>
                  <td>{fmtM3(t.storage_m3)}</td>
                  <td>{units(t) != null ? units(t).toLocaleString() : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ marginBottom: 0 }}>
            Sizing at {plan.rain_mm} mm design storm; units = volume ÷ (unit size × efficiency).</p>

          {plan.phases && plan.phases.some((p) => p.reachable) && (
            <>
              <h3 style={{ marginTop: 12 }}>Phased build-out plan</h3>
              <table className="cb">
                <thead><tr><th>phase</th><th>cut</th><th>+ sites</th><th>+ volume</th><th>≈ cost</th></tr></thead>
                <tbody>
                  {plan.phases.map((p) => (
                    <tr key={p.phase}>
                      <td>{p.phase}</td>
                      <td>{p.target_cut_pct}%</td>
                      <td>{p.reachable ? `+${p.add_sites.toLocaleString()}` : "—"}</td>
                      <td>{p.reachable ? fmtM3(p.add_storage_m3) : "—"}</td>
                      <td>{p.reachable ? `₹${p.add_cost_crore_inr} cr` : "unreachable"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted" style={{ marginBottom: 0 }}>
                Each phase adds sites deepest-first on buildable ground (no buildings, no water;
                under-street detention allowed). Costs at ₹{(plan.storage_inr_per_m3 || 6000).toLocaleString()}/m³
                RCC detention — indicative, not a bid. Turn on the “containers” map layer to see
                every planned site.</p>
            </>
          )}
        </>
      )}
    </Panel>
  );
}

// ---------------------------------------------------------------- advisory

const SEV = { low: "#2a4", moderate: "#f90", high: "#e23" };
const LANG_LABEL = { en: "English", hi: "हिन्दी", mr: "मराठी" };

export function AdvisoryPanel({ adv, onRefresh }) {
  const [lang, setLang] = useState("en");
  if (!adv) return null;
  const langs = Object.keys(adv.advisory || {});
  const cur = adv.advisory?.[langs.includes(lang) ? lang : langs[0]];
  return (
    <Panel title="Public advisory">
      <p style={{ margin: 0 }}>
        <span className="badge" style={{ background: SEV[adv.severity] + "33", color: SEV[adv.severity] }}>
          {adv.severity}
        </span>{" "}
        <span className={`badge ${adv.backend === "hosted" ? "gnn" : "emulator"}`}>
          {adv.backend === "hosted" ? "AI" : "template"}
        </span>
      </p>
      <div className="row" style={{ gap: 4 }}>
        {langs.map((l) => (
          <button key={l} className={l === (langs.includes(lang) ? lang : langs[0]) ? "sel" : "ghost"}
                  onClick={() => setLang(l)}>{LANG_LABEL[l] || l}</button>
        ))}
      </div>
      <p style={{ marginBottom: 4 }}>{cur}</p>
      <p className="muted" style={{ margin: 0 }}>
        {ago(adv.generated_at)} · grounded in live forecast + alerts + citizen reports{" "}
        <a href="#" onClick={(e) => { e.preventDefault(); onRefresh(); }}>refresh</a>
      </p>
    </Panel>
  );
}

// ---------------------------------------------------------------- night lights

export function NightLightsPanel({ nl }) {
  if (!nl) return null;
  const days = Math.round((Date.now() - new Date(nl.date) / 1) / 86400000);
  return (
    <Panel title="Power outages (night lights)">
      <div className="kv" style={{ marginTop: 0 }}>
        <div>Dark built-up cells</div>
        <div><b>{nl.n_outage_cells}</b> ({nl.pct_built_dark}% of lit built land)</div>
        <div>Data night</div><div>{nl.date} {days > 4 && <span className="err">({days} d old)</span>}</div>
      </div>
      <p className="muted" style={{ marginBottom: 0 }}>
        NASA Black Marble (VIIRS, 500 m): red = radiance &gt;{Math.round(nl.drop_frac * 100)}% below
        the 90-day normal. Satellite archive lags a few days — not real-time.</p>
    </Panel>
  );
}

// ---------------------------------------------------------------- city aggregate

export function CityPanel({ city, onSelectTile }) {
  if (!city) return <Panel title="City view"><p className="muted">loading…</p></Panel>;
  const t = city.totals;
  return (
    <Panel title={city.name}>
      <div className="kv" style={{ marginTop: 0 }}>
        <div>Flooded area</div><div><b>{(t.flooded_area_m2 / 1e6).toFixed(1)} km²</b></div>
        <div>Flooded volume</div><div>{fmtM3(t.flooded_volume_m3)}</div>
        <div>Critical / elevated sinks</div>
        <div><b style={{ color: "#e23" }}>{t.red}</b> / <b style={{ color: "#f90" }}>{t.amber}</b></div>
        <div>Citizen reports (24 h)</div><div>{t.reports_24h}</div>
        {t.outage_cells > 0 && (<><div>Power-outage cells</div><div>{t.outage_cells}</div></>)}
      </div>
      <table className="cb">
        <thead><tr><th>tile</th><th>rain</th><th>flood</th><th>R/A</th></tr></thead>
        <tbody>
          {city.tiles.map((ti) => (
            <tr key={ti.id} style={{ cursor: ti.built ? "pointer" : "default", opacity: ti.built ? 1 : 0.5 }}
                onClick={() => ti.built && onSelectTile(ti.id)}>
              <td>{ti.name.replace("Mumbai — ", "")}</td>
              <td>{ti.rain_mm != null ? `${Math.round(ti.rain_mm)} mm` : "—"}</td>
              <td>{ti.summary ? `${(ti.summary.flooded_area_m2 / 1e6).toFixed(1)} km²` : "…"}</td>
              <td>{ti.alerts ? `${ti.alerts.red}/${ti.alerts.amber}` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted" style={{ marginBottom: 0 }}>{city.note} Click a tile for the full model.</p>
    </Panel>
  );
}
