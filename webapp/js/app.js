/* ============================================================
   SOMS TRS Passage Planner — app.js
   Pure JS: no build step, no ONNX.
   IF scored via exported tree structure (model.json).
   ============================================================ */

'use strict';

/* ── Globals ──────────────────────────────────────────────── */
let CFG      = null;   // config.json
let MODEL    = null;   // model.json  { n, ms, trees[] }
let STRAIT   = null;   // trs_strait.json
let MONTHOUR = null;   // trs_monthour.json
let INCIDENTS= null;   // incidents.json

let leafletMap  = null;
let routeLayer  = null;
let incLayer    = null;
let transitChart= null;

const ROUTE_EAST = [
  [1.255, 103.570], [1.240, 103.700], [1.225, 103.820],
  [1.210, 103.950], [1.200, 104.040], [1.195, 104.120]
];
const ROUTE_WEST = [...ROUTE_EAST].reverse();

/* ── IF scoring helpers ───────────────────────────────────── */
function cFactor(n) {
  if (n <= 1) return 0.0;
  if (n === 2) return 1.0;
  return 2.0 * (Math.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n;
}

function pathLength(tree, x, node, depth) {
  node  = node  || 0;
  depth = depth || 0;
  if (tree.cl[node] === -1) {
    return depth + cFactor(tree.ns[node]);
  }
  const feat = tree.f[node];
  if (x[feat] <= tree.th[node]) {
    return pathLength(tree, x, tree.cl[node], depth + 1);
  } else {
    return pathLength(tree, x, tree.cr[node], depth + 1);
  }
}

function scoreSamples(model, xScaled) {
  const ms = model.ms;
  const c  = cFactor(ms);
  let sumPath = 0;
  for (const tree of model.trees) {
    sumPath += pathLength(tree, xScaled);
  }
  const meanPath = sumPath / model.n;
  return Math.pow(2, -meanPath / c);   // raw JS score (positive)
}

/* ── Scaler ───────────────────────────────────────────────── */
function applyScaler(raw17) {
  return raw17.map((v, i) => (v - CFG.SCALER_MEAN[i]) / CFG.SCALER_SCALE[i]);
}

/* ── TRS pipeline ─────────────────────────────────────────── */
function scoreTRS(raw17) {
  const scaled  = applyScaler(raw17);
  const jsRaw   = scoreSamples(MODEL, scaled);
  // Python convention: sklearn score_samples returns negative anomaly score
  // JS raw is positive average-path-based score; negate to match sklearn sign
  const skRaw   = -jsRaw;
  const trs = Math.max(0, Math.min(1,
    (skRaw - CFG.IF_RAW_MIN) / (CFG.IF_RAW_MAX - CFG.IF_RAW_MIN)
  ));
  return trs;
}

/* ── Risk band helpers ────────────────────────────────────── */
function riskBand(trs) {
  if (trs < CFG.T_LOW)  return 'LOW';
  if (trs < CFG.T_MOD)  return 'MODERATE';
  if (trs < CFG.T_HIGH) return 'HIGH';
  return 'CRITICAL';
}

function bandColour(band) {
  const map = { LOW: '#2ca02c', MODERATE: '#ff7f0e', HIGH: '#d62728', CRITICAL: '#7b0000' };
  return map[band] || '#888';
}

/* ── Haversine distance (km) ──────────────────────────────── */
function haversine([lat1, lon1], [lat2, lon2]) {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat/2)**2 +
            Math.cos(lat1 * Math.PI/180) * Math.cos(lat2 * Math.PI/180) *
            Math.sin(dLon/2)**2;
  return R * 2 * Math.asin(Math.sqrt(a));
}

/* ── Route geometry ───────────────────────────────────────── */
function buildRouteSegments(waypoints) {
  const segs = [];
  let cumDist = 0;
  for (let i = 1; i < waypoints.length; i++) {
    const d = haversine(waypoints[i-1], waypoints[i]);
    segs.push({ from: waypoints[i-1], to: waypoints[i], dist: d, cumStart: cumDist });
    cumDist += d;
  }
  return { segs, totalDist: cumDist };
}

function interpolateRoute(segs, totalDist, fracDist) {
  const target = fracDist * totalDist;
  for (const seg of segs) {
    const segEnd = seg.cumStart + seg.dist;
    if (target <= segEnd) {
      const t = (target - seg.cumStart) / seg.dist;
      const lat = seg.from[0] + t * (seg.to[0] - seg.from[0]);
      const lon = seg.from[1] + t * (seg.to[1] - seg.from[1]);
      return [lat, lon];
    }
  }
  const last = segs[segs.length - 1];
  return last.to;
}

/* ── Feature vector builder ───────────────────────────────── */
function buildFeatureVector(params) {
  // Build lookup keyed by exact IF_FEATURES column names from config.json
  const isDaytime = (params.sgtHour >= 6 && params.sgtHour < 18) ? 1 : 0;
  // SOMS jurisdiction: Singapore (0) east of ~103.85°E, Indonesia (2) west
  const jurCode = params.lon >= 103.85 ? 0 : 2;
  const monthEnv = CFG.MONTHLY_ENV[params.month]
                || CFG.MONTHLY_ENV[String(params.month)] || {};
  const visScore = monthEnv.Est_Visibility_Score ?? 3.0;

  const lookup = {
    'Lat_DD':                    params.lat,
    'Lon_DD':                    params.lon,
    'Hour_Sin':                  Math.sin(2 * Math.PI * params.sgtHour / 24),
    'Hour_Cos':                  Math.cos(2 * Math.PI * params.sgtHour / 24),
    'Month_Sin':                 Math.sin(2 * Math.PI * params.month / 12),
    'Month_Cos':                 Math.cos(2 * Math.PI * params.month / 12),
    'Wind_Speed_10m_kmh':        params.wind,
    'Cloud_Cover_%':             params.cloud,
    'Effective_Lunar_Illum_%':   params.lunar,
    'Is_Daytime_Int':            isDaytime,
    'Est_Visibility_Score':      visScore,
    'Traffic_Anomaly_%':         params.anomaly,
    'Ship_Type_Enc':             CFG.SHIP_TYPE_RISK[params.shipType] ?? 0,
    'Jurisdiction_Enc':          jurCode,
    'Lane_Eastbound Lane':       params.lane === 'Eastbound Lane'  ? 1 : 0,
    'Lane_Outside of TSS Lanes': 0,
    'Lane_Westbound Lane':       params.lane === 'Westbound Lane'  ? 1 : 0,
  };

  return CFG.IF_FEATURES.map(f => lookup[f] ?? 0);
}

/* ── Transit simulation ───────────────────────────────────── */
function runTransit() {
  const shipType = document.getElementById('t-ship').value;
  const lane     = getRadioVal('t-dir-group');
  const month    = parseInt(document.getElementById('t-month').value);
  const entryHour= parseInt(document.getElementById('t-hour').value);
  const speedKn  = parseFloat(document.getElementById('t-speed').value);
  const wind     = parseFloat(document.getElementById('t-wind').value);
  const cloud    = parseFloat(document.getElementById('t-cloud').value);
  const lunar    = parseFloat(document.getElementById('t-lunar').value);
  const anomaly  = parseFloat(document.getElementById('t-anomaly').value);

  const speedKmh = speedKn * 1.852;
  const waypoints= (lane === 'Eastbound Lane') ? ROUTE_EAST : ROUTE_WEST;
  const { segs, totalDist } = buildRouteSegments(waypoints);

  const stepMin  = 15;
  const stepH    = stepMin / 60;
  const steps    = [];
  let   elapsed  = 0;  // hours
  let   distDone = 0;

  while (distDone < totalDist) {
    const frac  = Math.min(distDone / totalDist, 1.0);
    const [lat, lon] = interpolateRoute(segs, totalDist, frac);
    const sgtHour = ((entryHour + elapsed) % 24 + 24) % 24;
    const sgtMin  = Math.round((sgtHour % 1) * 60);
    const sgtHourInt = Math.floor(sgtHour);
    const sgtStr  = `${String(sgtHourInt).padStart(2,'0')}:${String(sgtMin).padStart(2,'0')}`;

    // Jurisdiction label
    const jurLabel = (lon >= 103.83) ? 'Singapore' : 'International';

    // Score
    const trs  = scoreTRS(buildFeatureVector({ shipType, lane, month,
      sgtHour: sgtHourInt, lat, lon, wind, cloud, lunar, anomaly }));
    const band = riskBand(trs);

    steps.push({ elapsed, sgtStr, lat, lon, jurLabel, trs, band });

    distDone += speedKmh * stepH;
    elapsed  += stepH;
  }

  // Final position (exit)
  if (steps.length > 0) {
    const last = steps[steps.length - 1];
    if (last.lat.toFixed(3) !== waypoints[waypoints.length-1][0].toFixed(3)) {
      const [lat, lon] = waypoints[waypoints.length - 1];
      const sgtHour = ((entryHour + elapsed) % 24 + 24) % 24;
      const sgtHourInt = Math.floor(sgtHour);
      const sgtStr = `${String(sgtHourInt).padStart(2,'0')}:${String(Math.round((sgtHour%1)*60)).padStart(2,'0')}`;
      const trs  = scoreTRS(buildFeatureVector({ shipType, lane, month,
        sgtHour: sgtHourInt, lat, lon, wind, cloud, lunar, anomaly }));
      steps.push({ elapsed, sgtStr, lat, lon, jurLabel: 'International', trs, band: riskBand(trs) });
    }
  }

  renderResults(steps, totalDist, speedKn, entryHour);
}

/* ── Render results ───────────────────────────────────────── */
function renderResults(steps, totalDist, speedKn, entryHour) {
  document.getElementById('welcome-state').classList.add('hidden');
  document.getElementById('results-state').classList.remove('hidden');

  const peakStep  = steps.reduce((a, b) => a.trs > b.trs ? a : b, steps[0]);
  const entryTRS  = steps[0]?.trs ?? 0;
  const exitTRS   = steps[steps.length - 1]?.trs ?? 0;
  const duration  = steps[steps.length - 1]?.elapsed ?? 0;

  // KPI row
  const kpiRow = document.getElementById('kpi-row');
  kpiRow.innerHTML = `
    <div class="kpi-card">
      <div class="kpi-label">Distance</div>
      <div class="kpi-value">${totalDist.toFixed(1)}<span class="kpi-unit">km</span></div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Duration</div>
      <div class="kpi-value">${duration.toFixed(1)}<span class="kpi-unit">h</span></div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Peak TRS</div>
      <div class="kpi-value" style="color:${bandColour(peakStep.band)}">
        ${peakStep.trs.toFixed(3)}
        <span class="kpi-badge" style="background:${bandColour(peakStep.band)}">${peakStep.band}</span>
      </div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Entry → Exit TRS</div>
      <div class="kpi-value">
        <span style="color:${bandColour(riskBand(entryTRS))}">${entryTRS.toFixed(3)}</span>
        <span class="kpi-arrow">→</span>
        <span style="color:${bandColour(riskBand(exitTRS))}">${exitTRS.toFixed(3)}</span>
      </div>
    </div>
  `;

  buildTransitChart(steps, entryHour);
  updateLeafletMap(steps);
  buildTable(steps);
  buildBandSummary(steps);
}

/* ── Chart.js transit chart ──────────────────────────────── */
function buildTransitChart(steps, entryHour) {
  const ctx = document.getElementById('chart-transit').getContext('2d');
  if (transitChart) { transitChart.destroy(); }

  const labels = steps.map(s => s.sgtStr);
  const data   = steps.map(s => s.trs);
  const ptColors = steps.map(s => bandColour(s.band));

  // Band zone annotations (manual background segments via plugin-free gradient hack)
  // We use Chart.js dataset fill areas for band zones
  const T_LOW  = CFG.T_LOW;
  const T_MOD  = CFG.T_MOD;
  const T_HIGH = CFG.T_HIGH;

  // Night window: is the step in night hours (21–23 or 0–5)?
  function isNight(sgtStr) {
    const h = parseInt(sgtStr.split(':')[0]);
    return h >= 21 || h < 6;
  }
  function isDawn(sgtStr) {
    const h = parseInt(sgtStr.split(':')[0]);
    return h >= 4 && h < 7;
  }

  // Build background plugin for band zones
  const bandZonePlugin = {
    id: 'bandZones',
    beforeDraw(chart) {
      const { ctx: c, chartArea: { left, right, top, bottom }, scales } = chart;
      const yScale = scales.y;
      function yPx(v) { return yScale.getPixelForValue(v); }

      const zones = [
        { from: 0,      to: T_LOW,  color: 'rgba(44,160,44,0.08)'  },
        { from: T_LOW,  to: T_MOD,  color: 'rgba(255,127,14,0.10)' },
        { from: T_MOD,  to: T_HIGH, color: 'rgba(214,39,40,0.10)'  },
        { from: T_HIGH, to: 1.0,    color: 'rgba(123,0,0,0.12)'    },
      ];
      c.save();
      c.beginPath();
      c.rect(left, top, right - left, bottom - top);
      c.clip();
      zones.forEach(({ from, to, color }) => {
        c.fillStyle = color;
        c.fillRect(left, yPx(to), right - left, yPx(from) - yPx(to));
      });

      // Night shading along x-axis
      const xScale = scales.x;
      steps.forEach((s, i) => {
        if (isNight(s.sgtStr)) {
          const x0 = (i === 0) ? left : (xScale.getPixelForValue(i-1) + xScale.getPixelForValue(i)) / 2;
          const x1 = (i === steps.length-1) ? right : (xScale.getPixelForValue(i) + xScale.getPixelForValue(i+1)) / 2;
          c.fillStyle = 'rgba(0,0,60,0.15)';
          c.fillRect(x0, top, x1 - x0, bottom - top);
        }
      });

      // Threshold lines
      [{ v: T_LOW, label: 'LOW', col: '#2ca02c' },
       { v: T_MOD, label: 'MOD', col: '#ff7f0e' },
       { v: T_HIGH, label: 'HIGH', col: '#d62728' }].forEach(({ v, label, col }) => {
        c.setLineDash([4, 4]);
        c.strokeStyle = col + 'aa';
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(left, yPx(v));
        c.lineTo(right, yPx(v));
        c.stroke();
        c.setLineDash([]);
        c.fillStyle = col;
        c.font = '9px JetBrains Mono, monospace';
        c.fillText(label, left + 2, yPx(v) - 3);
      });
      c.restore();
    }
  };

  transitChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data,
        borderColor: '#c8a84b',
        borderWidth: 1.5,
        pointBackgroundColor: ptColors,
        pointRadius: 3,
        pointHoverRadius: 5,
        fill: false,
        tension: 0.3,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const s = steps[ctx.dataIndex];
              return ` TRS ${s.trs.toFixed(3)}  [${s.band}]  @ ${s.sgtStr} SGT`;
            }
          },
          backgroundColor: '#0d2240',
          borderColor: '#c8a84b44',
          borderWidth: 1,
          titleColor: '#b8cfe8',
          bodyColor:  '#e8f0f8',
        }
      },
      scales: {
        x: {
          ticks: { color: '#6a8aaa', font: { size: 10, family: 'JetBrains Mono, monospace' },
                   maxTicksLimit: 12, maxRotation: 0 },
          grid:  { color: 'rgba(255,255,255,0.05)' },
          title: { display: true, text: 'SGT', color: '#6a8aaa', font: { size: 11 } }
        },
        y: {
          min: 0, max: 1,
          ticks: { color: '#6a8aaa', font: { size: 10, family: 'JetBrains Mono, monospace' },
                   callback: v => v.toFixed(2) },
          grid:  { color: 'rgba(255,255,255,0.05)' },
          title: { display: true, text: 'TRS', color: '#6a8aaa', font: { size: 11 } }
        }
      }
    },
    plugins: [bandZonePlugin]
  });
}

/* ── Leaflet map ──────────────────────────────────────────── */
function buildLeafletMap(incidents) {
  if (leafletMap) return;   // already initialised

  leafletMap = L.map('map-route', { zoomControl: true, attributionControl: false })
    .setView([1.22, 103.85], 10);

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© CartoDB', subdomains: 'abcd', maxZoom: 14
  }).addTo(leafletMap);

  // Training incidents background layer
  incLayer = L.layerGroup().addTo(leafletMap);
  incidents.forEach(({ la, lo, b }) => {
    L.circleMarker([la, lo], {
      radius: 4, fillColor: bandColour(b), color: 'transparent',
      fillOpacity: 0.35, weight: 0
    }).addTo(incLayer);
  });

  routeLayer = L.layerGroup().addTo(leafletMap);
}

function updateLeafletMap(steps) {
  if (!leafletMap) buildLeafletMap(INCIDENTS);
  routeLayer.clearLayers();
  if (!steps || steps.length === 0) return;

  // Draw route line
  const latlngs = steps.map(s => [s.lat, s.lon]);
  L.polyline(latlngs, { color: '#c8a84b44', weight: 2, dashArray: '4 4' }).addTo(routeLayer);

  // Draw step circles
  steps.forEach((s, i) => {
    const isFirst = i === 0;
    const isLast  = i === steps.length - 1;
    const radius  = (isFirst || isLast) ? 7 : 4;
    const opacity = (isFirst || isLast) ? 0.95 : 0.75;
    const marker  = L.circleMarker([s.lat, s.lon], {
      radius, fillColor: bandColour(s.band), color: '#0d2240',
      fillOpacity: opacity, weight: 1
    });
    marker.bindTooltip(
      `<b>${s.sgtStr} SGT</b><br>TRS ${s.trs.toFixed(3)} [${s.band}]<br>${s.jurLabel}`,
      { className: 'map-tooltip', opacity: 0.95 }
    );
    marker.addTo(routeLayer);
  });

  // Entry / exit labels
  const entry = steps[0];
  const exit  = steps[steps.length - 1];
  L.marker([entry.lat, entry.lon], {
    icon: L.divIcon({ className: 'map-label', html: '<span class="map-label-text entry">ENTRY</span>', iconAnchor: [20, 6] })
  }).addTo(routeLayer);
  L.marker([exit.lat, exit.lon], {
    icon: L.divIcon({ className: 'map-label', html: '<span class="map-label-text exit">EXIT</span>', iconAnchor: [-4, 6] })
  }).addTo(routeLayer);

  // Fit bounds — invalidate first so Leaflet knows the real container size
  setTimeout(() => {
    leafletMap.invalidateSize();
    const bounds = L.latLngBounds(latlngs).pad(0.1);
    leafletMap.fitBounds(bounds);
  }, 50);
}

/* ── Step table ───────────────────────────────────────────── */
function buildTable(steps) {
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';
  steps.forEach(s => {
    const elH = Math.floor(s.elapsed);
    const elM = Math.round((s.elapsed % 1) * 60);
    const elStr = `${String(elH).padStart(2,'0')}h ${String(elM).padStart(2,'0')}m`;
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${elStr}</td>
      <td>${s.sgtStr}</td>
      <td>${s.lat.toFixed(4)}</td>
      <td>${s.lon.toFixed(4)}</td>
      <td>${s.jurLabel}</td>
      <td>${s.sgtStr}</td>
      <td>${s.trs.toFixed(3)}</td>
      <td><span class="band-chip" style="background:${bandColour(s.band)}20;color:${bandColour(s.band)};border-color:${bandColour(s.band)}44">${s.band}</span></td>
    `;
    tbody.appendChild(row);
  });
}

/* ── Band summary ─────────────────────────────────────────── */
function buildBandSummary(steps) {
  const counts = { LOW: 0, MODERATE: 0, HIGH: 0, CRITICAL: 0 };
  steps.forEach(s => counts[s.band]++);
  const total = steps.length;
  const bandSummary = document.getElementById('band-summary');
  bandSummary.innerHTML = Object.entries(counts).map(([band, n]) => {
    const pct = total > 0 ? (n / total * 100).toFixed(0) : 0;
    const col  = bandColour(band);
    return `
      <div class="bs-item">
        <div class="bs-label" style="color:${col}">${band}</div>
        <div class="bs-bar-wrap">
          <div class="bs-bar" style="width:${pct}%;background:${col}"></div>
        </div>
        <div class="bs-count">${n} step${n!==1?'s':''} · ${pct}%</div>
      </div>`;
  }).join('');
}

/* ── Strait Heatmap ───────────────────────────────────────── */
let straitHeatmapDrawn = false;

function drawStraitHeatmap() {
  if (!STRAIT || !CFG) return;

  const ship   = document.getElementById('hm-ship').value;
  const month  = document.getElementById('hm-month').value;
  const lane   = getRadioVal('hm-dir-group');
  const entryHour = parseInt(document.getElementById('hm-hour').value);
  const speedKn   = parseFloat(document.getElementById('hm-speed').value);

  const grid = (STRAIT[ship]?.[month]?.[lane]) || {};

  const canvas = document.getElementById('canvas-strait');
  const dpr    = window.devicePixelRatio || 1;
  const W = canvas.parentElement.clientWidth  || 600;
  const H = canvas.parentElement.clientHeight || 380;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const padL = 54, padR = 16, padT = 36, padB = 40;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const nHours = 24;
  const nPos   = 20;
  const cellW  = plotW / nHours;
  const cellH  = plotH / nPos;

  ctx.clearRect(0, 0, W, H);

  // Draw cells
  for (let h = 0; h < nHours; h++) {
    const row = grid[h] || Array(nPos).fill(0);
    for (let p = 0; p < nPos; p++) {
      const trs  = row[p] ?? 0;
      const band = riskBand(trs);
      ctx.fillStyle = bandColour(band) + (trs < CFG.T_LOW ? '55' : 'cc');
      ctx.fillRect(
        padL + h * cellW,
        padT + p * cellH,
        cellW - 0.5,
        cellH - 0.5
      );
    }
  }

  // Transit overlay
  const totalDist = 61.5;
  const speedKmh  = speedKn * 1.852;
  const durationH = totalDist / speedKmh;
  const exitHour  = (entryHour + durationH) % 24;

  ctx.save();
  ctx.beginPath();
  ctx.rect(padL, padT, plotW, plotH);
  ctx.clip();

  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth   = 2;
  ctx.shadowColor = '#ffffffaa';
  ctx.shadowBlur  = 4;
  ctx.beginPath();
  const x0 = padL + entryHour / 24 * plotW;
  const y0 = padT;
  const x1 = padL + (exitHour / 24) * plotW;
  const y1 = padT + plotH;
  // Handle wrap-around: draw two segments if exit > 24
  if (entryHour + durationH <= 24) {
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
  } else {
    // First segment
    const fracAtMidnight = (24 - entryHour) / durationH;
    const yMid = padT + fracAtMidnight * plotH;
    ctx.moveTo(x0, y0);
    ctx.lineTo(padL + plotW, yMid);
    // Second segment (wraps)
    const x1Wrap = padL + ((exitHour) / 24) * plotW;
    ctx.moveTo(padL, yMid);
    ctx.lineTo(x1Wrap, padT + plotH);
  }
  ctx.stroke();
  ctx.restore();
  ctx.shadowBlur = 0;
  ctx.setLineDash([]);

  // X-axis (hours)
  ctx.fillStyle = '#6a8aaa';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  for (let h = 0; h < nHours; h += 3) {
    ctx.fillText(`${String(h).padStart(2,'0')}`, padL + (h + 0.5) * cellW, padT + plotH + 14);
  }
  ctx.fillText('Hour of Entry (SGT)', padL + plotW / 2, H - 4);

  // Y-axis (position)
  ctx.textAlign = 'right';
  const posLabels = ['Entry', '25%', '50%', '75%', 'Exit'];
  posLabels.forEach((lbl, idx) => {
    const yFrac = idx / (posLabels.length - 1);
    ctx.fillText(lbl, padL - 4, padT + yFrac * plotH + 4);
  });

  // Title/subtitle
  ctx.fillStyle = '#b8cfe8';
  ctx.font = '600 11px Space Grotesk, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`${ship} · ${lane} · Month ${month}`, padL, padT - 10);

  straitHeatmapDrawn = true;
}

/* ── Month×Hour Heatmap ───────────────────────────────────── */
function drawMonthHourHeatmap() {
  if (!MONTHOUR || !CFG) return;

  const ship   = document.getElementById('hm-ship').value;
  const key    = MONTHOUR[ship] ? ship : 'All';
  const grid   = MONTHOUR[key] || {};

  const canvas = document.getElementById('canvas-monthour');
  const dpr    = window.devicePixelRatio || 1;
  const W = canvas.parentElement.clientWidth  || 440;
  const H = canvas.parentElement.clientHeight || 380;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const padL = 46, padR = 16, padT = 36, padB = 50;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const nHours  = 24;
  const nMonths = 12;
  const cellW   = plotW / nHours;
  const cellH   = plotH / nMonths;

  ctx.clearRect(0, 0, W, H);

  // Cells
  for (let m = 1; m <= nMonths; m++) {
    const monthData = grid[m] || {};
    for (let h = 0; h < nHours; h++) {
      const cell = monthData[h];
      let trs, n;
      if (cell === null || cell === undefined) {
        trs = null; n = 0;
      } else if (typeof cell === 'object') {
        trs = cell.v; n = cell.n;
      } else {
        trs = cell; n = 0;
      }

      if (trs === null || trs === undefined) {
        // Light green — no incidents
        ctx.fillStyle = '#1a3d1a';
      } else {
        const band = riskBand(trs);
        ctx.fillStyle = bandColour(band) + 'cc';
      }
      ctx.fillRect(
        padL + h * cellW,
        padT + (m - 1) * cellH,
        cellW - 0.5,
        cellH - 0.5
      );

      // Incident count annotation
      if (n > 0) {
        ctx.fillStyle = '#ffffff';
        ctx.font = `${Math.max(8, Math.min(10, cellH * 0.45))}px JetBrains Mono, monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(n),
          padL + (h + 0.5) * cellW,
          padT + (m - 0.5) * cellH
        );
      }
    }
  }

  // X-axis
  ctx.fillStyle = '#6a8aaa';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'center';
  for (let h = 0; h < nHours; h += 3) {
    ctx.fillText(`${String(h).padStart(2,'0')}`, padL + (h + 0.5) * cellW, padT + plotH + 14);
  }
  ctx.fillText('Hour of Day (SGT)', padL + plotW / 2, H - 4);

  // Y-axis (months)
  const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  ctx.textAlign = 'right';
  MONTHS.forEach((mo, i) => {
    ctx.fillText(mo, padL - 4, padT + (i + 0.5) * cellH + 4);
  });

  // Title
  ctx.fillStyle = '#b8cfe8';
  ctx.font = '600 11px Space Grotesk, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`${ship === 'All' ? 'All ships' : ship} · 2022–2025 incidents`, padL, padT - 10);
}

/* ── Colourbar ticks ──────────────────────────────────────── */
function updateColourbar() {
  if (!CFG) return;
  const T_LOW  = CFG.T_LOW;
  const T_MOD  = CFG.T_MOD;
  const T_HIGH = CFG.T_HIGH;

  // Colour bar proportional widths
  const total = 1.0;
  document.getElementById('cb-low').style.flex  = T_LOW;
  document.getElementById('cb-mod').style.flex  = T_MOD - T_LOW;
  document.getElementById('cb-high').style.flex = T_HIGH - T_MOD;
  document.getElementById('cb-crit').style.flex = 1.0 - T_HIGH;

  document.getElementById('cb-t-low').textContent  = T_LOW.toFixed(2);
  document.getElementById('cb-t-mod').textContent  = T_MOD.toFixed(2);
  document.getElementById('cb-t-high').textContent = T_HIGH.toFixed(2);
}

/* ── Welcome state thresholds ─────────────────────────────── */
function buildThresholdRow() {
  if (!CFG) return;
  const row = document.getElementById('threshold-row');
  row.innerHTML = [
    { label: 'LOW', color: '#2ca02c', val: `< ${CFG.T_LOW.toFixed(3)}` },
    { label: 'MODERATE', color: '#ff7f0e', val: `${CFG.T_LOW.toFixed(3)} – ${CFG.T_MOD.toFixed(3)}` },
    { label: 'HIGH', color: '#d62728', val: `${CFG.T_MOD.toFixed(3)} – ${CFG.T_HIGH.toFixed(3)}` },
    { label: 'CRITICAL', color: '#7b0000', val: `≥ ${CFG.T_HIGH.toFixed(3)}` },
  ].map(({ label, color, val }) =>
    `<div class="threshold-chip" style="border-color:${color}44;color:${color}">
      <span class="tc-label">${label}</span>
      <span class="tc-val">${val}</span>
    </div>`
  ).join('');
}

/* ── Env slider reset ─────────────────────────────────────── */
function resetEnvSliders(month) {
  if (!CFG?.MONTHLY_ENV) return;
  const env = CFG.MONTHLY_ENV[month] || CFG.MONTHLY_ENV[String(month)];
  if (!env) return;

  const fields = [
    { id: 't-wind',   key: 'Wind_Speed_10m_kmh',        valId: 't-wind-val',   suffix: ' km/h' },
    { id: 't-cloud',  key: 'Cloud_Cover_%',             valId: 't-cloud-val',  suffix: '%'     },
    { id: 't-lunar',  key: 'Effective_Lunar_Illum_%',   valId: 't-lunar-val',  suffix: '%'     },
    { id: 't-anomaly',key: 'Traffic_Anomaly_%',         valId: 't-anomaly-val', suffix: '%'    },
  ];
  fields.forEach(({ id, key, valId, suffix }) => {
    const slider = document.getElementById(id);
    const valEl  = document.getElementById(valId);
    if (slider && env[key] !== undefined) {
      slider.value    = env[key];
      valEl.textContent = parseFloat(env[key]).toFixed(1) + suffix;
      updateSliderFill(slider);
    }
  });

  const MONTH_NAMES = ['','January','February','March','April','May','June',
                       'July','August','September','October','November','December'];
  document.getElementById('env-note').textContent = `defaults: ${MONTH_NAMES[month]} medians`;
}

/* ── Slider fill gradient ─────────────────────────────────── */
function updateSliderFill(el) {
  const min = parseFloat(el.min);
  const max = parseFloat(el.max);
  const val = parseFloat(el.value);
  const pct = ((val - min) / (max - min) * 100).toFixed(1);
  el.style.setProperty('--pct', pct + '%');
}

/* ── Radio group utility ──────────────────────────────────── */
function getRadioVal(groupId) {
  const active = document.querySelector(`#${groupId} .radio-opt.active`);
  return active ? active.dataset.val : null;
}

function wireRadioGroup(groupId, onChange) {
  const group = document.getElementById(groupId);
  if (!group) return;
  group.querySelectorAll('.radio-opt').forEach(opt => {
    opt.addEventListener('click', () => {
      group.querySelectorAll('.radio-opt').forEach(o => o.classList.remove('active'));
      opt.classList.add('active');
      if (onChange) onChange(opt.dataset.val);
    });
  });
}

/* ── Populate dropdowns ───────────────────────────────────── */
function populateDropdowns() {
  const ships  = Object.keys(CFG.SHIP_TYPE_RISK);
  const months = [
    [1,'January'],[2,'February'],[3,'March'],[4,'April'],
    [5,'May'],[6,'June'],[7,'July'],[8,'August'],
    [9,'September'],[10,'October'],[11,'November'],[12,'December']
  ];

  ['t-ship','hm-ship'].forEach(id => {
    const sel = document.getElementById(id);
    sel.innerHTML = ships.map(s => `<option value="${s}">${s}</option>`).join('');
  });
  ['t-month','hm-month'].forEach(id => {
    const sel = document.getElementById(id);
    sel.innerHTML = months.map(([v,l]) => `<option value="${v}">${l}</option>`).join('');
    sel.value = 6;  // default June
  });
}

/* ── Tab switching ────────────────────────────────────────── */
function wireTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(t => { t.classList.remove('active'); t.classList.add('hidden'); });
      btn.classList.add('active');
      const tab = document.getElementById(`tab-${btn.dataset.tab}`);
      tab.classList.remove('hidden');
      tab.classList.add('active');
      if (btn.dataset.tab === 'heatmap') {
        // Small delay to ensure layout is done before drawing
        setTimeout(() => { drawStraitHeatmap(); drawMonthHourHeatmap(); }, 50);
      }
    });
  });
}

/* ── Heatmap controls wiring ──────────────────────────────── */
function wireHeatmapControls() {
  ['hm-ship','hm-month'].forEach(id => {
    document.getElementById(id)?.addEventListener('change', () => {
      drawStraitHeatmap();
      drawMonthHourHeatmap();
    });
  });
  wireRadioGroup('hm-dir-group', () => { drawStraitHeatmap(); });

  const hmHour  = document.getElementById('hm-hour');
  const hmSpeed = document.getElementById('hm-speed');
  hmHour.addEventListener('input', () => {
    document.getElementById('hm-hour-val').textContent =
      `${String(parseInt(hmHour.value)).padStart(2,'0')}:00`;
    updateSliderFill(hmHour);
    drawStraitHeatmap();
  });
  hmSpeed.addEventListener('input', () => {
    document.getElementById('hm-speed-val').textContent = hmSpeed.value;
    updateSliderFill(hmSpeed);
    drawStraitHeatmap();
  });
}

/* ── Transit panel wiring ─────────────────────────────────── */
function wireTransitControls() {
  wireRadioGroup('t-dir-group');

  const tHour  = document.getElementById('t-hour');
  const tSpeed = document.getElementById('t-speed');

  tHour.addEventListener('input', () => {
    document.getElementById('t-hour-val').textContent =
      `${String(parseInt(tHour.value)).padStart(2,'0')}:00`;
    updateSliderFill(tHour);
  });
  tSpeed.addEventListener('input', () => {
    document.getElementById('t-speed-val').textContent = tSpeed.value + ' kn';
    updateSliderFill(tSpeed);
  });

  // Env sliders
  const envSliders = [
    { id: 't-wind',   valId: 't-wind-val',   suffix: ' km/h' },
    { id: 't-cloud',  valId: 't-cloud-val',  suffix: '%'     },
    { id: 't-lunar',  valId: 't-lunar-val',  suffix: '%'     },
    { id: 't-anomaly',valId: 't-anomaly-val', suffix: '%'    },
  ];
  envSliders.forEach(({ id, valId, suffix }) => {
    const el  = document.getElementById(id);
    const val = document.getElementById(valId);
    el.addEventListener('input', () => {
      val.textContent = parseFloat(el.value).toFixed(1) + suffix;
      updateSliderFill(el);
    });
  });

  // Reset env sliders when month changes
  document.getElementById('t-month').addEventListener('change', e => {
    resetEnvSliders(parseInt(e.target.value));
  });


  document.getElementById('btn-run').addEventListener('click', () => {
    if (!MODEL || !CFG) {
      alert('Model not loaded yet. Please wait.');
      return;
    }
    runTransit();
  });
}

/* ── Loading overlay ──────────────────────────────────────── */
function setLoading(visible, msg) {
  const overlay = document.getElementById('loading-overlay');
  const sub     = document.getElementById('loading-sub');
  if (visible) {
    overlay.style.display = 'flex';
    if (msg) sub.textContent = msg;
  } else {
    overlay.style.display = 'none';
  }
}

/* ── Fetch helpers ────────────────────────────────────────── */
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return res.json();
}

/* ── Init ─────────────────────────────────────────────────── */
async function init() {
  setLoading(true, 'Loading configuration…');
  try {
    [CFG, INCIDENTS, STRAIT, MONTHOUR] = await Promise.all([
      fetchJSON('data/config.json'),
      fetchJSON('data/incidents.json'),
      fetchJSON('data/trs_strait.json'),
      fetchJSON('data/trs_monthour.json'),
    ]);

    setLoading(true, 'Loading IF model trees…');
    MODEL = await fetchJSON('data/model.json');

    setLoading(false);

    populateDropdowns();
    wireTabs();
    wireTransitControls();
    wireHeatmapControls();
    buildThresholdRow();
    updateColourbar();

    // Initialise slider fills & values for Transit tab
    document.querySelectorAll('.field-slider').forEach(updateSliderFill);
    document.getElementById('t-hour-val').textContent  = '06:00';
    document.getElementById('t-speed-val').textContent = '12 kn';
    document.getElementById('hm-hour-val').textContent = '06:00';
    document.getElementById('hm-speed-val').textContent= '12';

    // Pre-fill env sliders with June defaults
    resetEnvSliders(6);

    // Build Leaflet map eagerly (background incidents)
    buildLeafletMap(INCIDENTS);

    console.log(`[TRS] Init complete. ${MODEL.n} trees · ${INCIDENTS.length} incidents loaded.`);

  } catch (err) {
    setLoading(false);
    console.error('[TRS] Init failed:', err);
    document.getElementById('loading-overlay').innerHTML =
      `<div style="color:#d62728;font-size:1rem;text-align:center;padding:2rem;">
         ⚠ Failed to load model data.<br>
         <small style="color:#6a8aaa">Run Cell 22 in the notebook first, then serve via HTTP.<br>${err.message}</small>
       </div>`;
    document.getElementById('loading-overlay').style.display = 'flex';
  }
}

window.addEventListener('DOMContentLoaded', init);
window.addEventListener('resize', () => {
  if (document.getElementById('tab-heatmap').classList.contains('active')) {
    drawStraitHeatmap();
    drawMonthHourHeatmap();
  }
});
