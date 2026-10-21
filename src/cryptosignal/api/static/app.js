"use strict";

// Dashboard for the cryptosignal API. Plain JS + lightweight-charts (TradingView), no build step.
const LWC = window.LightweightCharts;
const $ = (id) => document.getElementById(id);

// Colour follows the entity, never its rank: each strategy keeps its slot everywhere.
const STRATEGY_COLORS = {
  logreg: "--series-1",
  buy_and_hold: "--series-2",
  sma_crossover: "--series-3",
  macd_crossover: "--series-4",
  momentum: "--series-5",
  hgb: "--series-7",
};
const HIDDEN_BY_DEFAULT = new Set(["momentum"]);

const state = { ticker: null, days: 365, price: null, equity: null, points: new Map() };

const fmt = {
  pct: (x, d = 1) =>
    x == null ? "-" : `${(x * 100).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d })}%`,
  num: (x, d = 2) =>
    x == null ? "-" : x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }),
  price: (x) => (x == null ? "-" : x.toLocaleString("en-US", { maximumFractionDigits: x < 10 ? 4 : 2 })),
  compact: (x) => (x == null ? "-" : new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(x)),
};

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const toTime = (isoDate) => Date.parse(`${isoDate}T00:00:00Z`) / 1000;
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};

async function api(path) {
  const res = await fetch(path, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) {
      /* not JSON */
    }
    const err = new Error(detail);
    err.status = res.status;
    err.retryAfter = Number(res.headers.get("Retry-After")) || null;
    throw err;
  }
  return res.json();
}

function notice(message, kind = "info") {
  const box = $("notice");
  box.hidden = !message;
  box.dataset.kind = kind;
  box.textContent = message || "";
}

// ---------------------------------------------------------------- charts

function baseOptions() {
  return {
    autoSize: true,
    layout: {
      background: { type: "solid", color: css("--surface") },
      textColor: css("--text-2"),
      fontFamily: getComputedStyle(document.body).fontFamily,
      panes: { separatorColor: css("--grid"), separatorHoverColor: css("--axis") },
    },
    grid: { vertLines: { color: css("--grid") }, horzLines: { color: css("--grid") } },
    rightPriceScale: { borderColor: css("--axis") },
    timeScale: { borderColor: css("--axis"), timeVisible: false },
    crosshair: { mode: LWC.CrosshairMode.Normal },
  };
}

const quiet = { priceLineVisible: false, lastValueVisible: false };

function createPriceChart() {
  const chart = LWC.createChart($("price-chart"), baseOptions());
  const s = {
    candles: chart.addSeries(LWC.CandlestickSeries, { borderVisible: false, priceLineVisible: false }),
    smaFast: chart.addSeries(LWC.LineSeries, { lineWidth: 2, ...quiet, crosshairMarkerVisible: false }),
    smaSlow: chart.addSeries(LWC.LineSeries, { lineWidth: 2, ...quiet, crosshairMarkerVisible: false }),
    volume: chart.addSeries(LWC.HistogramSeries, { priceFormat: { type: "volume" }, ...quiet }, 1),
    macdHist: chart.addSeries(LWC.HistogramSeries, quiet, 2),
    macd: chart.addSeries(LWC.LineSeries, { lineWidth: 2, ...quiet }, 2),
    macdSignal: chart.addSeries(LWC.LineSeries, { lineWidth: 2, ...quiet }, 2),
    rsi: chart.addSeries(LWC.LineSeries, { lineWidth: 2, ...quiet, autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 100 } }) }, 3),
  };
  s.rsiLines = [30, 70].map((price) => s.rsi.createPriceLine({ price, lineWidth: 1, lineStyle: LWC.LineStyle.Solid, axisLabelVisible: true, title: "" }));
  s.markers = LWC.createSeriesMarkers(s.candles, []);
  [0.56, 0.12, 0.16, 0.16].forEach((f, i) => chart.panes()[i].setStretchFactor(f));
  chart.subscribeCrosshairMove((param) => renderPriceLegend(param.time));
  return { chart, s };
}

function themePriceChart() {
  const { chart, s } = state.price;
  chart.applyOptions(baseOptions());
  const up = css("--up");
  const down = css("--down");
  s.candles.applyOptions({ upColor: up, downColor: down, wickUpColor: up, wickDownColor: down });
  s.smaFast.applyOptions({ color: css("--series-1") });
  s.smaSlow.applyOptions({ color: css("--series-2") });
  s.volume.applyOptions({ color: css("--volume") });
  s.macdHist.applyOptions({ color: css("--volume") });
  s.macd.applyOptions({ color: css("--series-1") });
  s.macdSignal.applyOptions({ color: css("--series-2") });
  s.rsi.applyOptions({ color: css("--series-7") });
  s.rsiLines.forEach((line) => line.applyOptions({ color: css("--axis") }));
  if (state.lastIndicators) setMarkers(state.lastIndicators.points);
}

function line(points, key) {
  return points.filter((p) => p[key] != null).map((p) => ({ time: toTime(p.date), value: p[key] }));
}

function setMarkers(points) {
  const markers = [];
  for (let i = 1; i < points.length; i += 1) {
    const prev = points[i - 1];
    const cur = points[i];
    if ([prev.macd, prev.macd_signal, cur.macd, cur.macd_signal].some((v) => v == null)) continue;
    if (cur.macd > cur.macd_signal && prev.macd <= prev.macd_signal) {
      markers.push({ time: toTime(cur.date), position: "belowBar", shape: "arrowUp", color: css("--up"), size: 0.6 });
    } else if (cur.macd < cur.macd_signal && prev.macd >= prev.macd_signal) {
      markers.push({ time: toTime(cur.date), position: "aboveBar", shape: "arrowDown", color: css("--down"), size: 0.6 });
    }
  }
  state.price.s.markers.setMarkers(markers);
}

function renderIndicators(data) {
  state.lastIndicators = data;
  const { chart, s } = state.price;
  const pts = data.points;
  state.points = new Map(pts.map((p) => [toTime(p.date), p]));
  state.windows = data.windows;
  s.candles.setData(pts.map((p) => ({ time: toTime(p.date), open: p.open, high: p.high, low: p.low, close: p.close })));
  s.smaFast.setData(line(pts, "sma_fast"));
  s.smaSlow.setData(line(pts, "sma_slow"));
  s.volume.setData(line(pts, "volume"));
  s.macdHist.setData(line(pts, "macd_hist"));
  s.macd.setData(line(pts, "macd"));
  s.macdSignal.setData(line(pts, "macd_signal"));
  s.rsi.setData(line(pts, "rsi"));
  setMarkers(pts);
  chart.timeScale().fitContent();
  renderPriceLegend();
}

function legendItem(colorVar, label, value, box = false) {
  const key = el("span", { className: box ? "key box" : "key" });
  key.style.background = css(colorVar);
  return el("span", { className: "legend-item" }, key, el("strong", { textContent: value }), document.createTextNode(label));
}

function renderPriceLegend(time) {
  const legend = $("price-legend");
  const last = state.lastIndicators?.points.at(-1);
  const p = (time != null && state.points.get(time)) || last;
  if (!p) return legend.replaceChildren();
  const w = state.windows;
  legend.replaceChildren(
    el("span", { className: "legend-item" }, el("strong", { textContent: p.date })),
    legendItem("--text", "close", fmt.price(p.close), true),
    legendItem("--series-1", `SMA ${w.sma_fast}`, fmt.price(p.sma_fast)),
    legendItem("--series-2", `SMA ${w.sma_slow}`, fmt.price(p.sma_slow)),
    legendItem("--volume", "volume", fmt.compact(p.volume), true),
    legendItem("--series-1", "MACD", fmt.num(p.macd)),
    legendItem("--series-2", `signal (${w.macd_signal})`, fmt.num(p.macd_signal)),
    legendItem("--series-7", `RSI ${w.rsi}`, fmt.num(p.rsi, 1)),
    legendItem("--muted", `${w.volatility}-day volatility`, p.volatility == null ? "-" : `${fmt.num(p.volatility)}%`, true),
  );
}

function createEquityChart() {
  const opts = baseOptions();
  opts.rightPriceScale.mode = LWC.PriceScaleMode.Logarithmic;
  const chart = LWC.createChart($("equity-chart"), opts);
  chart.subscribeCrosshairMove((param) => renderEquityLegend(param.time));
  return { chart, series: new Map(), data: null, visible: new Map() };
}

function renderBacktest(bt) {
  const eq = state.equity;
  eq.series.forEach((series) => eq.chart.removeSeries(series));
  eq.series.clear();
  eq.data = bt;
  eq.times = bt.equity.dates.map(toTime);
  eq.index = new Map(eq.times.map((t, i) => [t, i]));
  const order = [...bt.models, "buy_and_hold", "sma_crossover", "macd_crossover", "momentum"];
  for (const name of order) {
    const values = bt.equity.series[name];
    if (!values) continue;
    if (!eq.visible.has(name)) eq.visible.set(name, !HIDDEN_BY_DEFAULT.has(name));
    const series = eq.chart.addSeries(LWC.LineSeries, {
      color: css(STRATEGY_COLORS[name] || "--muted"),
      lineWidth: 2,
      visible: eq.visible.get(name),
      priceLineVisible: false,
      lastValueVisible: false,
      priceFormat: { type: "custom", formatter: (v) => `$${fmt.num(v)}` },
    });
    series.setData(values.map((value, i) => ({ time: eq.times[i], value })));
    eq.series.set(name, series);
  }
  eq.chart.timeScale().fitContent();
  renderEquityLegend();
  $("equity-sub").textContent =
    `Growth of $1 from ${bt.test_start} to ${bt.test_end}, out-of-sample, net of ${bt.cost_bps} bps per trade (log scale). Click a legend entry to toggle it.`;
  renderKpis(bt);
  renderTables(bt);
}

function renderEquityLegend(time) {
  const eq = state.equity;
  if (!eq.data) return;
  const i = time != null && eq.index.has(time) ? eq.index.get(time) : eq.times.length - 1;
  const items = [el("span", { className: "legend-item" }, el("strong", { textContent: eq.data.equity.dates[i] }))];
  for (const [name, series] of eq.series) {
    const key = el("span", { className: "key" });
    key.style.background = css(STRATEGY_COLORS[name] || "--muted");
    const button = el(
      "button",
      { type: "button", className: "legend-item", title: "Show or hide this series" },
      key,
      el("strong", { textContent: `$${fmt.num(eq.data.equity.series[name][i])}` }),
      document.createTextNode(eq.data.labels[name] || name),
    );
    button.setAttribute("aria-pressed", String(eq.visible.get(name)));
    button.addEventListener("click", () => {
      eq.visible.set(name, !eq.visible.get(name));
      series.applyOptions({ visible: eq.visible.get(name) });
      renderEquityLegend(time);
    });
    items.push(button);
  }
  $("equity-legend").replaceChildren(...items);
}

function themeEquityChart() {
  const eq = state.equity;
  const opts = baseOptions();
  opts.rightPriceScale.mode = LWC.PriceScaleMode.Logarithmic;
  eq.chart.applyOptions(opts);
  eq.series.forEach((series, name) => series.applyOptions({ color: css(STRATEGY_COLORS[name] || "--muted") }));
  renderEquityLegend();
}

// ---------------------------------------------------------------- tiles & tables

function renderSignal(sig) {
  const value = $("signal-value");
  value.replaceChildren(
    document.createTextNode(sig.signal === "long" ? "Long" : "Flat"),
    el("span", { className: "badge", textContent: `P(up) ${fmt.pct(sig.probability_up)}` }),
  );
  $("signal-sub").textContent = `For the ${sig.prediction_for} daily close, using bars through ${sig.as_of}.`;
  const stale = sig.stale ? " · data provider unavailable, showing cached data" : "";
  $("signal-meta").textContent = `${sig.model_version} · ${sig.training_samples} training days · source: ${sig.data_source}${stale}`;
}

function renderKpis(bt) {
  const model = bt.models[0];
  const c = bt.classification[model];
  const s = bt.strategies[model];
  const bh = bt.strategies.buy_and_hold;
  $("kpi-acc").textContent = fmt.pct(c.accuracy);
  $("kpi-acc-sub").textContent = `${bt.labels[model]} · always-up baseline ${fmt.pct(bt.up_rate)}`;
  $("kpi-auc").textContent = c.roc_auc == null ? "-" : fmt.num(c.roc_auc, 3);
  $("kpi-sharpe").textContent = fmt.num(s.sharpe);
  $("kpi-sharpe-sub").textContent = `Buy & hold: ${fmt.num(bh.sharpe)}`;
  $("kpi-dd").textContent = fmt.pct(s.max_drawdown);
  $("kpi-dd-sub").textContent = `Buy & hold: ${fmt.pct(bh.max_drawdown)}`;
}

function row(cells, highlight) {
  const tr = el("tr", highlight ? { className: "highlight" } : {});
  cells.forEach((text) => tr.append(el("td", { textContent: text })));
  return tr;
}

function renderTables(bt) {
  const models = new Set(bt.models);
  $("strategy-table").tBodies[0].replaceChildren(
    ...Object.entries(bt.strategies).map(([name, s]) =>
      row([bt.labels[name] || name, fmt.pct(s.total_return, 0), fmt.pct(s.cagr), fmt.num(s.sharpe), fmt.pct(s.max_drawdown), fmt.pct(s.exposure, 0), String(s.trades)], models.has(name)),
    ),
  );
  $("class-table").tBodies[0].replaceChildren(
    ...Object.entries(bt.classification).map(([name, c]) =>
      row([bt.labels[name] || name, fmt.pct(c.accuracy), `${fmt.pct(c.accuracy_ci95[0])} - ${fmt.pct(c.accuracy_ci95[1])}`, c.roc_auc == null ? "-" : fmt.num(c.roc_auc, 3), fmt.pct(c.up_rate_predicted, 0)], models.has(name)),
    ),
  );
  $("table-sub").textContent =
    `${bt.n_test} out-of-sample days (${bt.test_start} to ${bt.test_end}); model re-fitted every ${bt.retrain_every} days after a ${bt.min_train_days}-day first window; ${fmt.pct(bt.up_rate)} of days closed up.`;
}

// ---------------------------------------------------------------- data flow

async function withRetry(load, label) {
  try {
    return await load();
  } catch (err) {
    if (err.status === 503) {
      const wait = err.retryAfter || 60;
      notice(`${label}: ${err.message}. Retrying automatically in ${wait}s.`, "error");
      setTimeout(() => refresh(), wait * 1000);
    } else {
      notice(`${label}: ${err.message}`, "error");
    }
    return null;
  }
}

async function loadIndicators() {
  const chartBox = $("price-chart");
  chartBox.dataset.loading = "true";
  const data = await withRetry(() => api(`/indicators/${state.ticker}?days=${state.days}`), "Indicators");
  chartBox.dataset.loading = "false";
  if (data) renderIndicators(data);
}

async function refresh() {
  const ticker = state.ticker;
  notice("");
  const slow = setTimeout(
    () => notice(`Preparing ${ticker}: fetching market data and training the model. A cold start on the free hosting tier can take up to a minute.`),
    1500,
  );
  const sig = await withRetry(() => api(`/signal/${ticker}`), "Signal");
  clearTimeout(slow);
  if (ticker !== state.ticker) return;
  if (sig) {
    notice("");
    renderSignal(sig);
  }
  await loadIndicators();
  const equityBox = $("equity-chart");
  equityBox.dataset.loading = "true";
  const bt = await withRetry(() => api(`/backtest/${ticker}`), "Backtest");
  equityBox.dataset.loading = "false";
  if (bt && ticker === state.ticker) renderBacktest(bt);
}

async function init() {
  if (!LWC) {
    notice("The chart library could not be loaded (offline or blocked CDN). The API still works: see /docs.", "error");
    return;
  }
  state.price = createPriceChart();
  state.equity = createEquityChart();
  themePriceChart();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    themePriceChart();
    themeEquityChart();
  });

  const select = $("ticker");
  const tickers = await withRetry(() => api("/tickers"), "Tickers");
  if (!tickers) return;
  select.replaceChildren(...tickers.map((t) => el("option", { value: t, textContent: t })));
  state.ticker = tickers[0];
  select.addEventListener("change", () => {
    state.ticker = select.value;
    state.equity.visible.clear();
    refresh();
  });
  $("range").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-days]");
    if (!button) return;
    state.days = Number(button.dataset.days);
    for (const b of $("range").querySelectorAll("button")) b.setAttribute("aria-checked", String(b === button));
    loadIndicators();
  });
  refresh();
}

document.addEventListener("DOMContentLoaded", init);
