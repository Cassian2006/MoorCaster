import { useEffect, useMemo, useState } from "react";
import {
  Line,
  LineChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { CircleMarker, MapContainer, Popup, Rectangle, TileLayer } from "react-leaflet";
import { CalendarDays, Database, RefreshCw, Sparkles } from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE || "";
const FRONTEND_BUILD_TIME = typeof __BUILD_TIME__ !== "undefined" ? __BUILD_TIME__ : "-";

const ROI_BOUNDS = [
  [30.5, 121.9],
  [30.75, 122.25]
];
const ROI_CENTER = [30.625, 122.075];

const I18N = {
  zh: {
    title: "MoorCaster 控制台",
    subtitle: "AIS 主体 + Sentinel-1/YOLO 视觉证据",
    serverUpdated: "数据最后更新(UTC)",
    frontendUpdated: "前端构建时间(UTC)",
    refresh: "刷新",
    tabDashboard: "总览",
    tabForecast: "预测",
    tabEvidence: "证据卡",
    tabJobs: "任务",
    presence: "在场船舶",
    idle: "低速停留船舶",
    waitingMean: "平均等待",
    waitingP90: "P90等待",
    waitingP95: "P95等待",
    vesselUnit: "艘",
    hourUnit: "小时",
    mapTitle: "AIS 质量检查地图",
    congestionTrend: "拥堵日曲线",
    waitingTrend: "等待时长日曲线",
    selectDate: "选择预测日期(未来24天任意一天)",
    aisForecast: "AIS 拥堵预测",
    visionForecast: "视觉融合预测",
    waitingForecast: "等待时长预测",
    p90Explain: "P90: 90% 的等待事件低于该值。",
    p95Explain: "P95: 95% 的等待事件低于该值。",
    modelExplain: "说明: 预测曲线按日输出，内部计算单位为分钟，前端展示为小时/船舶。",
    forecastReadout: "预测解读",
    risk: "风险等级",
    consistency: "一致性(AIS vs 视觉)",
    noVision: "无视觉预测",
    crossDayHint: "出现跨日等待(>=24h)。在高拥堵锚地场景中，这种现象可能是正常的排队等待。",
    noData: "暂无数据",
    open: "查看",
    ingestHint: "Web 端不会下载数据；这里只会基于已落盘的 AIS/S1 数据刷新预测与可视化。",
    startPipeline: "刷新预测",
    running: "运行中",
    stopped: "已停止",
    progress: "同步状态"
  },
  en: {
    title: "MoorCaster Console",
    subtitle: "AIS primary + Sentinel-1/YOLO visual evidence",
    serverUpdated: "Data Last Updated (UTC)",
    frontendUpdated: "Frontend Build Time (UTC)",
    refresh: "Refresh",
    tabDashboard: "Dashboard",
    tabForecast: "Forecast",
    tabEvidence: "Evidence",
    tabJobs: "Jobs",
    presence: "Presence",
    idle: "Idle",
    waitingMean: "Mean Waiting",
    waitingP90: "P90 Waiting",
    waitingP95: "P95 Waiting",
    vesselUnit: "vessels",
    hourUnit: "hours",
    mapTitle: "AIS Quality Map",
    congestionTrend: "Daily Congestion",
    waitingTrend: "Daily Waiting",
    selectDate: "Forecast Date (any day in next 24 days)",
    aisForecast: "AIS Forecast",
    visionForecast: "Vision Blend Forecast",
    waitingForecast: "Waiting Forecast",
    p90Explain: "P90: 90% of waiting events are below this value.",
    p95Explain: "P95: 95% of waiting events are below this value.",
    modelExplain: "Notes: model is daily; internal waiting unit is minutes, UI displays hours/vessels.",
    forecastReadout: "Forecast Readout",
    risk: "Risk",
    consistency: "Consistency (AIS vs Vision)",
    noVision: "No visual forecast",
    crossDayHint: "Cross-day waiting detected (>=24h). This can be normal in anchorage queues during heavy congestion.",
    noData: "No data",
    open: "Open",
    ingestHint: "The web app does not download data; it only refreshes forecasts from already downloaded AIS/S1 data.",
    startPipeline: "Refresh Forecast",
    running: "Running",
    stopped: "Stopped",
    progress: "Ingestion Status"
  }
};
async function fetchJson(path, options) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return res.json();
}

function fmtNum(v, d = 1) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return n.toFixed(d);
}

function buildForecastNarrative(sel) {
  const ais = Number(sel?.c?.forecast_value || 0);
  const vision = Number(sel?.v?.vision_forecast || 0);
  const meanHr = Number(sel?.w?.mean_hours || 0);
  const p90Hr = Number(sel?.w?.p90_hours || 0);

  let risk = "low";
  if (ais >= 120 || p90Hr >= 18) risk = "high";
  else if (ais >= 80 || p90Hr >= 12) risk = "medium";

  const visionGap = Math.abs(ais - vision);
  const consistency = vision > 0 ? (visionGap <= 15 ? "high" : visionGap <= 35 ? "medium" : "low") : "no_vision";
  const crossDay = meanHr >= 24 || p90Hr >= 24;

  return { risk, consistency, crossDay, ais, vision, meanHr, p90Hr };
}

function StatCard({ label, value, unit, tone = "primary" }) {
  const toneMap = {
    primary: "border-l-primary bg-primary/5",
    secondary: "border-l-secondary bg-secondary/5",
    warning: "border-l-warning bg-warning/5"
  };
  return (
    <div className={`rounded-xl border border-border border-l-4 p-4 ${toneMap[tone] || toneMap.primary}`}>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-foreground">
        {value} <span className="text-sm font-medium text-muted-foreground">{unit}</span>
      </div>
    </div>
  );
}

export default function App() {
  const [lang, setLang] = useState("zh");
  const t = I18N[lang];

  const [tab, setTab] = useState("dashboard");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [meta, setMeta] = useState({});
  const [congestion, setCongestion] = useState([]);
  const [waitingDay, setWaitingDay] = useState([]);
  const [waitingSummary, setWaitingSummary] = useState({});
  const [fcCongestion, setFcCongestion] = useState([]);
  const [fcWaiting, setFcWaiting] = useState([]);
  const [fcVision, setFcVision] = useState([]);
  const [cards, setCards] = useState([]);
  const [selectedCard, setSelectedCard] = useState(null);
  const [mapGeo, setMapGeo] = useState({ type: "FeatureCollection", features: [] });
  const [jobs, setJobs] = useState({});
  const [downloadProgress, setDownloadProgress] = useState({});
  const [selectedDate, setSelectedDate] = useState("");

  const refreshAll = async () => {
    setLoading(true);
    setError("");
    try {
      const [
        metaRes,
        congestionRes,
        waitingDayRes,
        waitingSummaryRes,
        fcCongestionRes,
        fcWaitingRes,
        fcVisionRes,
        cardsRes,
        mapRes,
        jobsRes
      ] = await Promise.all([
        fetchJson("/api/meta"),
        fetchJson("/api/series/congestion?granularity=day"),
        fetchJson("/api/series/waiting/day"),
        fetchJson("/api/series/waiting/summary"),
        fetchJson("/api/forecast/congestion?horizon_days=24"),
        fetchJson("/api/forecast/waiting?horizon_days=24"),
        fetchJson("/api/forecast/vision?horizon_days=24"),
        fetchJson("/api/evidence/cards?limit=20"),
        fetchJson("/api/map/ais-points?limit=2500"),
        fetchJson("/api/jobs/status")
      ]);
      setMeta(metaRes || {});
      setCongestion(congestionRes?.items || []);
      setWaitingDay(waitingDayRes?.items || []);
      setWaitingSummary(waitingSummaryRes || {});
      setFcCongestion(fcCongestionRes?.items || []);
      setFcWaiting(fcWaitingRes?.items || []);
      setFcVision(fcVisionRes?.items || []);
      setCards(cardsRes?.items || []);
      setMapGeo(mapRes || { type: "FeatureCollection", features: [] });
      setJobs(jobsRes?.jobs || {});
      setDownloadProgress(jobsRes?.download_progress || {});
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  };

  const refreshJobStatus = async () => {
    try {
      const jobsRes = await fetchJson("/api/jobs/status");
      const metaRes = await fetchJson("/api/meta");
      setJobs(jobsRes?.jobs || {});
      setDownloadProgress(jobsRes?.download_progress || {});
      setMeta(metaRes || {});
    } catch {
      // transient backend error
    }
  };

  useEffect(() => {
    refreshAll();
    const timer = setInterval(refreshJobStatus, 5000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!selectedDate && fcCongestion.length > 0) {
      setSelectedDate(fcCongestion[0].time_bin);
    }
  }, [fcCongestion, selectedDate]);

  const latestCongestion = congestion.length > 0 ? congestion[congestion.length - 1] : {};

  const forecastDates = useMemo(() => fcCongestion.map((x) => x.time_bin), [fcCongestion]);

  const selectedForecast = useMemo(() => {
    const c = fcCongestion.find((x) => x.time_bin === selectedDate);
    const w = fcWaiting.find((x) => x.date === selectedDate);
    const v = fcVision.find((x) => x.time_bin === selectedDate);
    return { c, w, v };
  }, [selectedDate, fcCongestion, fcWaiting, fcVision]);

  const forecastNarrative = useMemo(() => buildForecastNarrative(selectedForecast), [selectedForecast]);

  const mergedForecastLine = useMemo(() => {
    const map = new Map();
    for (const r of fcCongestion) {
      map.set(r.time_bin, { date: r.time_bin, ais_idle: Number(r.forecast_value || 0) });
    }
    for (const r of fcVision) {
      const row = map.get(r.time_bin) || { date: r.time_bin };
      row.vision = Number(r.vision_forecast || 0);
      map.set(r.time_bin, row);
    }
    for (const r of fcWaiting) {
      const row = map.get(r.date) || { date: r.date };
      row.waiting_p90_hr = Number(r.p90_hours || 0);
      map.set(r.date, row);
    }
    return Array.from(map.values()).sort((a, b) => String(a.date).localeCompare(String(b.date)));
  }, [fcCongestion, fcVision, fcWaiting]);

  const startPipeline = async () => {
    try {
      await fetchJson("/api/jobs/pipeline/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ horizon_days: 24 })
      });
      await refreshJobStatus();
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-40 border-b border-border bg-card/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-4 px-4 py-4">
          <div>
            <h1 className="text-2xl font-semibold text-primary">{t.title}</h1>
            <p className="text-sm text-muted-foreground">{t.subtitle}</p>
          </div>
          <div className="text-right text-xs text-muted-foreground">
            <div>{t.serverUpdated}: {meta?.last_updated_utc || "-"}</div>
            <div>{t.frontendUpdated}: {FRONTEND_BUILD_TIME}</div>
          </div>
          <div className="flex items-center gap-2">
            <button
              className="inline-flex items-center gap-2 rounded-md border border-border bg-accent px-3 py-2 text-sm hover:bg-muted"
              onClick={refreshAll}
            >
              <RefreshCw className={`size-4 ${loading ? "animate-spin" : ""}`} />
              {t.refresh}
            </button>
            <button
              className="rounded-md border border-border bg-accent px-3 py-2 text-sm hover:bg-muted"
              onClick={() => setLang((x) => (x === "zh" ? "en" : "zh"))}
            >
              {lang === "zh" ? "EN" : "涓枃"}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] px-4 py-6">
        <div className="mb-4 grid w-full grid-cols-2 gap-2 rounded-lg border border-border bg-card p-1 md:grid-cols-4">
          {[
            ["dashboard", t.tabDashboard],
            ["forecast", t.tabForecast],
            ["evidence", t.tabEvidence],
            ["jobs", t.tabJobs]
          ].map(([k, label]) => (
            <button
              key={k}
              onClick={() => setTab(k)}
              className={`rounded-md px-3 py-2 text-sm transition ${tab === k ? "bg-primary text-primary-foreground" : "hover:bg-accent"}`}
            >
              {label}
            </button>
          ))}
        </div>

        {error ? (
          <div className="mb-4 rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error}
          </div>
        ) : null}

        {tab === "dashboard" && (
          <section className="space-y-4">
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-5">
              <StatCard label={t.presence} value={fmtNum(latestCongestion?.presence_mmsi || 0, 0)} unit={t.vesselUnit} tone="primary" />
              <StatCard label={t.idle} value={fmtNum(latestCongestion?.idle_mmsi || 0, 0)} unit={t.vesselUnit} tone="secondary" />
              <StatCard label={t.waitingMean} value={fmtNum(waitingSummary?.mean_hr || 0, 2)} unit={t.hourUnit} tone="warning" />
              <StatCard label={t.waitingP90} value={fmtNum(waitingSummary?.p90_hr || 0, 2)} unit={t.hourUnit} tone="warning" />
              <StatCard label={t.waitingP95} value={fmtNum(waitingSummary?.p95_hr || 0, 2)} unit={t.hourUnit} tone="warning" />
            </div>

            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
              <div className="rounded-xl border border-border bg-card p-4">
                <h3 className="mb-2 text-base font-semibold">{t.mapTitle}</h3>
                <MapContainer center={ROI_CENTER} zoom={11} style={{ height: 380, width: "100%" }}>
                  <TileLayer url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" />
                  <Rectangle bounds={ROI_BOUNDS} pathOptions={{ color: "#0d9488", weight: 2 }} />
                  {mapGeo?.features?.map((f, idx) => (
                    <CircleMarker
                      key={idx}
                      center={[f.geometry.coordinates[1], f.geometry.coordinates[0]]}
                      radius={2}
                      pathOptions={{ color: "#ea580c", fillOpacity: 0.65, weight: 0 }}
                    >
                      <Popup>
                        MMSI: {f.properties?.mmsi || "-"}<br />
                        Time: {f.properties?.postime || "-"}<br />
                        SOG: {fmtNum(f.properties?.sog || 0, 2)} kn
                      </Popup>
                    </CircleMarker>
                  ))}
                </MapContainer>
              </div>

              <div className="rounded-xl border border-border bg-card p-4">
                <h3 className="mb-2 text-base font-semibold">{t.congestionTrend}</h3>
                <ResponsiveContainer width="100%" height={380}>
                  <LineChart data={congestion}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time_bin" hide />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="presence_mmsi" stroke="#0c4a6e" dot={false} name={t.presence} />
                    <Line type="monotone" dataKey="idle_mmsi" stroke="#0d9488" dot={false} name={t.idle} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="rounded-xl border border-border bg-card p-4">
              <h3 className="mb-2 text-base font-semibold">{t.waitingTrend}</h3>
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={waitingDay}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="mean_hours" stroke="#0c4a6e" dot={false} name={`${t.waitingMean}(${t.hourUnit})`} />
                  <Line type="monotone" dataKey="p90_hours" stroke="#ea580c" dot={false} name={`${t.waitingP90}(${t.hourUnit})`} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>
        )}

        {tab === "forecast" && (
          <section className="space-y-4">
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="mb-2 flex items-center gap-2 text-sm text-muted-foreground">
                <CalendarDays className="size-4" />
                {t.selectDate}
              </div>
              <select
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
              >
                {forecastDates.map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
              <StatCard label={t.aisForecast} value={fmtNum(selectedForecast?.c?.forecast_value || 0, 2)} unit={t.vesselUnit} tone="primary" />
              <StatCard label={t.visionForecast} value={fmtNum(selectedForecast?.v?.vision_forecast || 0, 2)} unit={t.vesselUnit} tone="secondary" />
              <StatCard label={t.waitingMean} value={fmtNum(selectedForecast?.w?.mean_hours || 0, 2)} unit={t.hourUnit} tone="warning" />
              <StatCard label={t.waitingForecast} value={fmtNum(selectedForecast?.w?.p90_hours || 0, 2)} unit={t.hourUnit} tone="warning" />
            </div>

            <div className="rounded-xl border border-border bg-card p-4">
              <ResponsiveContainer width="100%" height={340}>
                <LineChart data={mergedForecastLine}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="ais_idle" stroke="#0c4a6e" dot={false} name={t.aisForecast} />
                  <Line type="monotone" dataKey="vision" stroke="#0d9488" dot={false} name={t.visionForecast} />
                  <Line type="monotone" dataKey="waiting_p90_hr" stroke="#ea580c" dot={false} name={`${t.waitingP90}(${t.hourUnit})`} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="rounded-xl border border-warning/30 bg-warning/5 p-4 text-sm">
              <div>{t.p90Explain}</div>
              <div>{t.p95Explain}</div>
              <div className="mt-1 text-muted-foreground">{t.modelExplain}</div>
            </div>

            <div className="rounded-xl border border-border bg-card p-4 text-sm">
              <div className="mb-2 font-semibold">{t.forecastReadout}</div>
              <div>
                {t.risk}:{" "}
                <span className={`rounded px-2 py-0.5 text-xs ${
                  forecastNarrative.risk === "high"
                    ? "bg-warning/20 text-warning"
                    : forecastNarrative.risk === "medium"
                    ? "bg-secondary/20 text-secondary"
                    : "bg-primary/15 text-primary"
                }`}>
                  {forecastNarrative.risk.toUpperCase()}
                </span>
              </div>
              <div className="mt-1 text-muted-foreground">
                AIS={fmtNum(forecastNarrative.ais, 1)} vessels, Vision={fmtNum(forecastNarrative.vision, 1)} vessels, Mean Waiting={fmtNum(forecastNarrative.meanHr, 2)} h, P90={fmtNum(forecastNarrative.p90Hr, 2)} h.
              </div>
              <div className="mt-1 text-muted-foreground">
                {t.consistency}: {forecastNarrative.consistency === "no_vision" ? t.noVision : forecastNarrative.consistency}.
              </div>
              {forecastNarrative.crossDay ? (
                <div className="mt-1 text-warning">{t.crossDayHint}</div>
              ) : null}
            </div>
          </section>
        )}

        {tab === "evidence" && (
          <section className="space-y-4">
            {cards.length === 0 ? (
              <div className="rounded-xl border border-border bg-card p-6 text-sm text-muted-foreground">{t.noData}</div>
            ) : (
              <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                {cards.map((c) => (
                  <div key={c.card_id || c._file} className="rounded-xl border border-border bg-card p-4">
                    <div className="mb-2 flex items-center justify-between">
                      <div className="font-semibold text-primary">{c.card_id || "-"}</div>
                      <button className="rounded-md border border-border px-2 py-1 text-xs hover:bg-accent" onClick={() => setSelectedCard(c)}>
                        {t.open}
                      </button>
                    </div>
                    <div className="text-sm text-muted-foreground">{c.t_anchor || "-"}</div>
                    <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                      <div>{t.presence}: {fmtNum(c?.congestion_snapshot?.presence_count || 0, 0)} {t.vesselUnit}</div>
                      <div>{t.idle}: {fmtNum(c?.congestion_snapshot?.idle_count || 0, 0)} {t.vesselUnit}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        {tab === "jobs" && (
          <section className="space-y-4">
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="mb-3 rounded-md border border-border bg-accent px-3 py-2 text-sm text-muted-foreground">
                {t.ingestHint}
              </div>
              <div className="mb-3 flex flex-wrap gap-2">
                <button className="rounded-md bg-secondary px-3 py-2 text-sm text-secondary-foreground hover:bg-secondary/90" onClick={startPipeline}>
                  {t.startPipeline}
                </button>
                <button className="rounded-md border border-border px-3 py-2 text-sm hover:bg-accent" onClick={refreshJobStatus}>
                  {t.refresh}
                </button>
              </div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Database className="size-4" />
                {t.progress}: files={downloadProgress?.count || 0}, latest={downloadProgress?.latest || "-"}, mtime={downloadProgress?.latest_mtime || "-"}
              </div>
              <div className="mt-2 text-xs text-muted-foreground">
                active model: {meta?.active_yolo_model || "-"}
              </div>
            </div>

            {Object.entries(jobs).map(([name, j]) => (
              <div className="rounded-xl border border-border bg-card p-4" key={name}>
                <div className="mb-2 flex items-center justify-between">
                  <h3 className="font-semibold">{name}</h3>
                  <span className={`rounded px-2 py-1 text-xs ${j.running ? "bg-secondary/15 text-secondary" : "bg-muted text-muted-foreground"}`}>
                    {j.running ? t.running : t.stopped}
                  </span>
                </div>
                <div className="grid grid-cols-1 gap-1 text-xs text-muted-foreground md:grid-cols-3">
                  <div>pid: {j.pid || "-"}</div>
                  <div>start: {j.started_at || "-"}</div>
                  <div>end: {j.finished_at || "-"}</div>
                </div>
                <pre className="mt-3 max-h-52 overflow-auto rounded-md border border-border bg-accent p-2 text-xs">
                  {(j.log_tail || []).slice(-20).join("\n")}
                </pre>
              </div>
            ))}
          </section>
        )}
      </main>

      {selectedCard && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={() => setSelectedCard(null)}>
          <div className="max-h-[85vh] w-full max-w-4xl overflow-auto rounded-xl border border-border bg-card p-4" onClick={(e) => e.stopPropagation()}>
            <div className="mb-3 flex items-center justify-between">
              <h3 className="font-semibold text-primary">{selectedCard.card_id || "-"}</h3>
              <button className="rounded-md border border-border px-2 py-1 text-xs hover:bg-accent" onClick={() => setSelectedCard(null)}>
                Close
              </button>
            </div>
            <pre className="rounded-md border border-border bg-accent p-3 text-xs">
              {JSON.stringify(selectedCard, null, 2)}
            </pre>
          </div>
        </div>
      )}

      <footer className="border-t border-border bg-card/90 px-4 py-4 text-center text-xs text-muted-foreground">
        <div className="inline-flex items-center gap-2">
          <Sparkles className="size-3.5" />
          ROI: lat 30.50-30.75, lon 121.90-122.25 | S1(GRD) + YOLO as visual evidence engine
        </div>
      </footer>
    </div>
  );
}
