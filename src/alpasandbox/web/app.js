/* AlpaSandbox web client. No build step, no dependencies. */
"use strict";

const $ = (id) => document.getElementById(id);
const CAM_ORDER = [6, 1, 0, 2, 3, 5];               // two columns, front to back: teles/wide, cross, rear corners
const S = {
  info: null, state: "starting", frame: null, ws: null, connected: false,
  zoom: 1, latencies: [], drive: [], logs: [], unseenErrors: 0, pending: new Map(), focusedCam: null,
};

/* ---------------- utilities ---------------- */
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  kids.flat().forEach((k) => k != null && n.append(k));
  return n;
}
function toast(message, level = "info", ms = 5000) {
  const t = el("div", { class: `toast ${level}`, role: level === "error" ? "alert" : "status", text: message });
  $("toasts").append(t);
  setTimeout(() => t.remove(), ms);
}
function send(cmd, payload = {}) {
  if (S.ws && S.ws.readyState === WebSocket.OPEN) S.ws.send(JSON.stringify({ cmd, ...payload }));
  else toast("Not connected to the sandbox server.", "error");
}

/* ---------------- connection ---------------- */
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  S.ws = ws;
  ws.onopen = () => { S.connected = true; $("conn").textContent = "Connected"; $("conn").classList.add("on"); };
  ws.onclose = () => {
    S.connected = false; $("conn").textContent = "Reconnecting…"; $("conn").classList.remove("on");
    setTimeout(connect, 1500);
  };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    (HANDLERS[m.type] || (() => {}))(m);
  };
}

const HANDLERS = {
  hello(m) { S.info = m; applyInfo(m); },
  state(m) { setState(m.state, m.error); },
  frame(m) { S.frame = m; renderFrame(m); },
  episode(m) {
    S.latencies = []; S.drive = []; S.lastReason = null; RTF.length = 0; drawDrive(m.duration); lastTimeline = "";
    setNavCurrent(m.nav_text);
    const started = `Scenario ${m.scenario.replaceAll("_", " ")} started`;
    feedAppend("nav-feed", el("div", { class: "msg system", text: `${started}${m.nav_text ? ` with “${m.nav_text}”` : ""}` }));
    $("scenario-select").value = m.scenario;
    setReason("Press Drive to start the scenario.", "", true);
    $("incident").hidden = true;
  },
  nav(m) {
    setNavCurrent(m.nav_text);
    if (m.changed === false) { toast(`Already the current command: “${m.nav_text}”`); return; }
    if (!$("tab-drive").classList.contains("active")) toast(`Navigation command sent: “${m.nav_text}”`);
    feedAppend("nav-feed", el("div", { class: "msg q cmd" }, el("span", { class: "who", text: "Navigation command" }), m.nav_text,
      el("small", { text: S.frame ? `Sent at ${fmt(S.frame.t)} s · the model drives with it from the next plan` : "The model drives with it from the next plan" })));
  },
  thinking(m) { addChat({ id: m.id, question: m.question, pending: true }); },
  answer(m) { addChat(m); },
  summary(m) { showSummary(m.summary); },
  settings(m) { $("realtime").checked = !!m.realtime; },
  log(m) { addLog(m); },
  toast(m) { toast(m.message, m.level); },
};

/* ---------------- info / state ---------------- */
function applyInfo(m) {
  const model = m.model;
  if (m.repo_url) $("repo-link").href = m.repo_url; else $("repo-link").removeAttribute("href");
  $("model-name").textContent = model ? model.display_name : "Model not loaded";
  $("model-runtime").textContent = m.runtime || "";
  $("model-chip").title = model ? `${model.description}\n${model.homepage || ""}` : "";
  $("backend-label").textContent = m.backend ? `Simulator: ${m.backend}` : "";
  $("realtime").checked = !!m.realtime;

  const sel = $("scenario-select");
  sel.replaceChildren(...m.scenarios.map((s) => el("option", { value: s.name, text: s.name.replaceAll("_", " "), title: s.description })));
  sel.value = m.scenario;

  $("nav-presets").replaceChildren(...m.nav_presets.map((p) => el("button", {
    type: "button", class: "chip", text: p, onclick: () => send("nav", { text: p }),
  })));
  setNavCurrent(m.nav_text);

  const vqa = !!(model && model.supports_vqa);
  $("question-presets").replaceChildren(...m.question_presets.map((q) => el("button", {
    type: "button", class: "chip", text: q, onclick: () => ask(q),
  })));
  $("ask-input").disabled = $("ask-send").disabled = !vqa;
  $("ask-input").placeholder = vqa ? "Ask about the current scene…" : "This model does not answer questions.";
  if (!vqa) $("chat-empty").textContent = "This model only drives. Load Alpamayo 2 Edge or Super to ask about the scene.";

  // rebuild both feeds: questions and answers in Conversation, this episode's reasoning in Drive instructions
  for (const id of ["chat", "nav-feed"]) { $(id).querySelectorAll(":scope > :not(.empty)").forEach((n) => n.remove()); $(id).querySelector(".empty").hidden = false; }
  lastTimeline = "";
  (m.chat || []).forEach(addChat);
  (m.reasoning_log || []).forEach(addTimeline);
  if (m.logs) { $("log").replaceChildren(); S.logs = []; m.logs.forEach((l) => addLog(l, true)); }
  S.drive = (m.drive || []).slice();
  S.latencies = [];
  drawDrive(m.duration);
  buildCameraWall(m);
  setState(m.state, m.error);
  if (m.summary && m.state === "finished") renderMetrics($("metrics"), m.summary);
}

function setState(state, error) {
  S.state = state;
  const dot = $("state-dot"); dot.className = `dot ${state}`;
  const play = $("btn-play");
  const running = state === "running";
  play.classList.toggle("playing", running);
  $("play-label").textContent = running ? "Pause" : state === "finished" ? "Again" : "Drive";
  play.setAttribute("aria-label", running ? "Pause" : "Drive");
  const busy = state === "loading" || state === "starting" || state === "error";
  ["btn-play", "btn-step", "btn-reset", "scenario-select"].forEach((id) => ($(id).disabled = busy));
  $("nav-send").disabled = busy;
  $("btn-step").disabled = busy || running || state === "finished";

  const ov = $("loading"), card = ov.querySelector(".loading-card");
  if (state === "loading" || state === "starting") {
    ov.hidden = false; card.classList.remove("error");
    $("loading-title").textContent = "Loading the model";
    $("loading-sub").textContent = "The first load downloads the weights and can take several minutes.";
  } else if (state === "error") {
    ov.hidden = false; card.classList.add("error");
    $("loading-title").textContent = "The sandbox could not start";
    $("loading-sub").replaceChildren(el("pre", { text: error || "Unknown error" }),
      el("span", { text: "Fix the problem shown above, then restart `alpasandbox app`. Run `alpasandbox doctor` to check your setup." }));
  } else {
    ov.hidden = true;
    if (error) toast(error, "error", 8000);
  }
}

/* ---------------- cameras ---------------- */
function buildCameraWall(m) {
  const wall = $("camera-wall");
  if (wall.childElementCount) return;
  CAM_ORDER.forEach((cid, i) => {
    const tile = el("button", { class: "cam", type: "button", "data-cam": cid, "aria-label": `${m.camera_names[cid]} camera, enlarge` },
      el("img", { alt: `${m.camera_names[cid]} camera`, id: `cam-${cid}` }),
      el("span", { class: "cam-label", text: m.camera_names[cid] }),
      el("span", { class: "cam-key", text: String(i + 1) }));
    tile.addEventListener("click", () => focusCam(cid));
    wall.append(tile);
  });
}
function focusCam(cid) {
  document.querySelectorAll(".cam.focused").forEach((n) => n.classList.remove("focused"));
  S.focusedCam = S.focusedCam === cid ? null : cid;
  if (S.focusedCam !== null) document.querySelector(`.cam[data-cam="${cid}"]`).classList.add("focused");
}

/* ---------------- HUD over the bird's-eye view ---------------- */
function renderHud(f, last) {
  const st = Math.max(-1, Math.min(1, (f.control || {}).steer || 0));     // left positive, 1 = full lock
  $("hud-wheel").style.transform = `rotate(${-st * 450}deg)`;           // a road car turns about 1.25 turns to lock
  const pct = Math.round(Math.abs(st) * 100);
  $("hud-steer").textContent = pct === 0 ? "0%" : `${st > 0 ? "L" : "R"} ${pct}%`;
  $("hud-speed").textContent = Math.round(f.speed * 3.6);
  const light = $("hud-light");
  if (f.signal) { light.className = `hud-light ${f.signal}`; light.hidden = false; light.title = `Traffic light: ${f.signal.toLowerCase()} (simulator ground truth)`; }
  else light.hidden = true;
  $("hud-reach").textContent = last ? `${fmt(Math.hypot(last[0], last[1]), 0)} m` : "–";
  $("hud-time").textContent = `${fmt(f.t)} s`;
}
function setBevMode(mode) {
  S.bevMode = mode === "top" ? "top" : "3d";
  $("bev-panel").dataset.mode = S.bevMode;
  const b = $("bev-view");
  b.textContent = S.bevMode === "3d" ? "2D" : "3D";
  b.setAttribute("aria-label", S.bevMode === "3d" ? "Switch to top-down view" : "Switch to tilted 3D view");
  try { localStorage.setItem("alpasandbox-bev", S.bevMode); } catch (e) { /* storage unavailable */ }
  if (S.frame) drawBev(S.frame);
}

/* ---------------- frame ---------------- */
function renderFrame(f) {
  for (const [cid, uri] of Object.entries(f.cams || {})) { const img = $(`cam-${cid}`); if (img) img.src = uri; }
  const c = f.control || {};
  $("throttle-bar").style.width = `${Math.round((c.throttle || 0) * 100)}%`;
  $("brake-bar").style.width = `${Math.round((c.brake || 0) * 100)}%`;
  const st = Math.max(-1, Math.min(1, c.steer || 0));   // left positive: bar grows to the left
  const sb = $("steer-bar");
  sb.style.width = `${Math.abs(st) * 50}%`; sb.style.left = st > 0 ? `${50 - st * 50}%` : "50%";
  if (f.latency_ms == null) { $("latency").textContent = "–"; $("latency-unit").textContent = "per plan"; }
  else if (f.latency_ms >= 1000) { $("latency").textContent = fmt(f.latency_ms / 1000, 1); $("latency-unit").textContent = "s per plan"; }
  else { $("latency").textContent = Math.round(f.latency_ms); $("latency-unit").textContent = "ms per plan"; }
  const lo = Math.abs(f.lane_offset) < 0.005 ? 0 : f.lane_offset;
  $("stat-lane").textContent = lo === 0 ? "0.00 m" : `${fmt(Math.abs(lo), 2)} m ${lo > 0 ? "left" : "right"}`;
  const last = f.plan && f.plan.length ? f.plan[f.plan.length - 1] : null;
  $("stat-reach").textContent = last ? `${fmt(Math.hypot(last[0], last[1]), 0)} m in 6.4 s` : "–";
  updateRtf(f);
  $("clock").textContent = `${fmt(f.t)} s / ${fmt(f.duration, 0)} s`;
  $("stepcount").textContent = `Step ${f.step}`;
  $("progress").style.width = `${Math.min(100, (100 * f.t) / (f.duration || 1))}%`;

  if (f.step > 0) {
    if (f.reasoning) { setReason(f.reasoning, `At ${fmt(f.t)} s${f.meta && f.meta.meta_action ? `, ${f.meta.meta_action}` : ""}`, false); S.lastReason = { t: f.t, text: f.reasoning }; }
    else setReason("Routine driving, no explanation this step.", `At ${fmt(f.t)} s`, true,
      S.lastReason ? `Last explanation, at ${fmt(S.lastReason.t)} s: ${S.lastReason.text}` : "");
    if (f.reasoning) maybeTimeline({ t: f.t, step: f.step, text: f.reasoning });
    if (f.latency_ms != null) { S.latencies.push(f.latency_ms); if (S.latencies.length > 600) S.latencies.shift(); }
    if (f.metrics) renderMetrics($("metrics"), f.metrics);
    if (!S.drive.length || f.t > S.drive[S.drive.length - 1].t) S.drive.push({ t: f.t, v: f.speed * 3.6, brake: (f.control || {}).brake || 0, reason: !!f.reasoning, inc: f.collision || f.off_road });
    drawSpark();
  }
  renderHud(f, last);
  const sig = $("signal");
  if (f.signal) { sig.className = `signal ${f.signal}`; sig.querySelector("span").textContent = `Traffic light: ${f.signal.toLowerCase()}`; sig.hidden = false; }
  else sig.hidden = true;
  const inc = $("incident");
  if (f.collision || f.off_road) { inc.textContent = f.collision ? "Collision" : "Off the road"; inc.hidden = false; }
  drawBev(f);
  drawDrive(f.duration);
}

/* ---------------- drive history: speed, braking, reasons ---------------- */
function drawDrive(duration) {
  const cv = $("drive"); if (!cv || !cv.clientWidth) return;
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const padL = 34, padB = 18, padT = 12, cw = w - padL - 6, ch = h - padB - padT;
  const T = Math.max(duration || 1, 1);
  const vmax = Math.max(40, ...S.drive.map((d) => d.v)) * 1.1;
  const X = (t) => padL + (t / T) * cw, Y = (v) => padT + ch - (v / vmax) * ch;
  g.font = `11px ${css("--font")}`; g.fillStyle = css("--muted"); g.strokeStyle = css("--grid"); g.lineWidth = 1;
  for (let v = 0; v <= vmax; v += 10) { if (v % 20) continue; g.beginPath(); g.moveTo(padL, Y(v)); g.lineTo(w - 6, Y(v)); g.stroke(); g.fillText(`${v}`, 4, Y(v) + 4); }
  for (let t = 0; t <= T; t += 5) g.fillText(`${t}s`, X(t) - 6, h - 4);
  const step = cw / Math.max(T / 0.1, 1);
  g.fillStyle = css("--red");
  S.drive.forEach((d) => { if (d.brake > 0.02) { g.globalAlpha = 0.18 + 0.5 * d.brake; g.fillRect(X(d.t) - step / 2, padT + ch - 8, Math.max(step, 1.5), 8); } });
  g.globalAlpha = 1; g.fillStyle = css("--amber");
  S.drive.forEach((d) => { if (d.reason) g.fillRect(X(d.t) - 0.75, padT - 8, 1.5, 7); });
  g.strokeStyle = css("--text"); g.lineWidth = 2; g.lineJoin = "round"; g.beginPath();
  S.drive.forEach((d, i) => (i ? g.lineTo(X(d.t), Y(d.v)) : g.moveTo(X(d.t), Y(d.v)))); g.stroke();
  const inc = S.drive.find((d) => d.inc);
  if (inc) { g.strokeStyle = css("--red"); g.setLineDash([4, 4]); g.beginPath(); g.moveTo(X(inc.t), padT); g.lineTo(X(inc.t), padT + ch); g.stroke(); g.setLineDash([]); }
  g.fillStyle = css("--muted"); g.fillText("km/h", 4, padT - 1);
}

const RTF = [];
function updateRtf(f) {
  const now = performance.now() / 1000;
  if (f.step === 0) { RTF.length = 0; $("stat-rtf").textContent = "–"; return; }
  RTF.push([now, f.t]); while (RTF.length > 30) RTF.shift();
  if (RTF.length < 3) return;
  const [w0, t0] = RTF[0], [w1, t1] = RTF[RTF.length - 1];
  if (w1 - w0 < 0.5 || t1 <= t0) return;
  const r = (t1 - t0) / (w1 - w0);
  $("stat-rtf").textContent = r >= 0.95 ? `${fmt(r, 1)}× real time` : `${fmt(r, r < 0.1 ? 2 : 1)}× real time`;
}
function setReason(text, meta, nominal, more = "") {   // status line at the right of the tab bar
  $("reason-text").textContent = text;
  $("reason-now").title = more ? `${text}\n\n${more}` : text;
  $("reason-meta").textContent = meta;
  $("reason-now").classList.toggle("nominal", nominal);
}
let lastTimeline = "";
function maybeTimeline(e) { if (e.text !== lastTimeline) addTimeline(e); }
// Each change in the model's reasoning is a bullet in the Drive instructions feed. Consecutive bullets
// share one list, so a command you send reads as a prompt and the reasoning after it as the response.
function addTimeline(e) {
  lastTimeline = e.text;
  const li = el("li", {}, el("time", { text: `${fmt(e.t)} s` }), el("span", { text: e.text }));
  const last = $("nav-feed").lastElementChild;
  if (last && last.classList.contains("reason-list")) {
    feedAppend("nav-feed", li, last);
    while (last.childElementCount > 200) last.firstElementChild.remove();
  } else feedAppend("nav-feed", el("ul", { class: "reason-list", "aria-label": "Model reasoning" }, li));
}
function chatAppend(node) { feedAppend("chat", node); }
function feedAppend(id, node, into = null) {   // keeps a feed pinned to the newest entry unless you scrolled up
  const c = $(id), pinned = c.scrollHeight - c.scrollTop - c.clientHeight < 80;
  c.querySelector(".empty").hidden = true;
  (into || c).append(node);
  while (c.childElementCount > 301) c.querySelector(":scope > :not(.empty)").remove();
  if (pinned) c.scrollTop = c.scrollHeight;
}

/* ---------------- bird's-eye view ---------------- */
function drawBev(f) {
  if (S.bevMode === "3d") drawBev3D(f); else drawBevTop(f);
}

/* Tilted chase-camera view. Ego frame: x forward, y left, z up (metres); a pinhole camera sits behind
   and above the car, pitched down at a point ahead. Everything is projected by hand onto the 2D canvas,
   with geometry behind the camera clipped at a near plane. */
const NEAR = 0.8;
function chaseCam(w, h, zoom) {
  const dist = 10 / zoom, height = 6.5 / zoom, ahead = 22;
  const pitch = Math.atan2(height, ahead + dist), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const f = h * 0.95, cx = w / 2, cy = h * 0.5;
  const depth = (p) => (p[0] + dist) * cp - ((p[2] || 0) - height) * sp;
  const proj = (p) => {
    const dx = p[0] + dist, dz = (p[2] || 0) - height, d = dx * cp - dz * sp, up = dx * sp + dz * cp;
    return [cx - (f * p[1]) / d, cy - (f * up) / d];
  };
  return { depth, proj, horizon: cy - f * Math.tan(pitch) };
}
const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, (a[2] || 0) + ((b[2] || 0) - (a[2] || 0)) * t];
function clipPoly(pts, cam) {                 // Sutherland-Hodgman against the near plane
  const out = [];
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length], da = cam.depth(a), db = cam.depth(b);
    if (da >= NEAR) out.push(a);
    if ((da >= NEAR) !== (db >= NEAR)) out.push(lerp3(a, b, (NEAR - da) / (db - da)));
  }
  return out;
}

function drawBev3D(f) {
  const cv = $("bev"), dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cam = chaseCam(w, h, S.zoom);
  g.fillStyle = css("--surface"); g.fillRect(0, 0, w, h);
  const sky = g.createLinearGradient(0, 0, 0, Math.max(1, cam.horizon));
  sky.addColorStop(0, css("--bg")); sky.addColorStop(1, css("--surface"));
  g.fillStyle = sky; g.fillRect(0, 0, w, Math.max(0, cam.horizon));

  const fillPoly = (pts) => {
    const c = clipPoly(pts, cam); if (c.length < 3) return;
    g.beginPath(); c.forEach((p, i) => { const [u, v] = cam.proj(p); i ? g.lineTo(u, v) : g.moveTo(u, v); }); g.closePath(); g.fill();
  };
  const stroke = (pts, color, width, dash = [], alpha = 1) => {
    if (!pts || pts.length < 2) return;
    g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash); g.lineJoin = "round"; g.lineCap = "round"; g.globalAlpha = alpha;
    g.beginPath(); let pen = false;
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], da = cam.depth(a);
      if (i > 0) {
        const b = pts[i - 1], db = cam.depth(b);
        if ((da >= NEAR) !== (db >= NEAR)) {
          const q = cam.proj(lerp3(b, a, (NEAR - db) / (da - db)));
          if (pen) g.lineTo(q[0], q[1]); else { g.moveTo(q[0], q[1]); pen = true; }
          if (da < NEAR) pen = false;
        }
      }
      if (da >= NEAR) { const [u, v] = cam.proj(a); if (pen) g.lineTo(u, v); else { g.moveTo(u, v); pen = true; } }
    }
    g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
  };

  // road: lane surfaces (CARLA) or the area between the two edges (builtin), then crosswalks and paint
  const lines = f.lines || [];
  g.fillStyle = css("--road");
  const edges = lines.filter((l) => l.kind === "edge");
  if (edges.length === 2 && !lines.some((l) => l.kind === "surface")) fillPoly([...edges[0].pts, ...[...edges[1].pts].reverse()]);
  for (const l of lines) if (l.kind === "surface") fillPoly(l.pts);
  g.fillStyle = css("--crosswalk");
  for (const l of lines) if (l.kind === "crosswalk") fillPoly(l.pts);
  for (const l of lines) {
    if (l.kind === "edge") stroke(l.pts, css("--road-edge"), 2.2);
    else if (l.kind === "center") stroke(l.pts, css("--road-edge"), 1.6, [10, 10]);
    else if (l.kind === "lane") stroke(l.pts, css("--road-edge"), 1.4, [8, 12], 0.75);
    else if (l.kind === "junction") stroke(l.pts, css("--muted"), 1, [], 0.3);
  }
  // range rings on the ground, labelled on the right
  for (let r = 10; r <= 100; r += 10) {
    const ring = []; for (let k = 0; k <= 72; k++) { const a = (k / 72) * 2 * Math.PI; ring.push([r * Math.cos(a), r * Math.sin(a)]); }
    stroke(ring, css("--road-edge"), 1, [], r % 20 ? 0.08 : 0.14);
  }
  g.fillStyle = css("--muted"); g.font = `12px ${css("--font")}`;
  for (const r of [20, 40, 60]) { const p = [r * 0.94, -r * 0.34]; if (cam.depth(p) > NEAR) { const [u, v] = cam.proj(p); if (v > cam.horizon + 8 && u < w - 60) g.fillText(`${r} m`, u + 4, v - 4); } }

  // alternatives, then the plan as a ribbon one car wide with a corridor on each side
  (f.candidates || []).forEach((c) => stroke([[0, 0], ...c], css("--teal"), 1.6, [], 0.5));
  const plan = f.plan && f.plan.length ? [[0, 0], ...f.plan] : null;
  if (plan) {
    const side = (k) => plan.map((p, i) => {
      const a = plan[Math.max(0, i - 1)], b = plan[Math.min(plan.length - 1, i + 1)];
      const tx = b[0] - a[0], ty = b[1] - a[1], n = Math.hypot(tx, ty) || 1;
      return [p[0] - (ty / n) * k, p[1] + (tx / n) * k];
    });
    const L = side(0.95), R = side(-0.95);
    g.fillStyle = css("--ribbon"); fillPoly([...L, ...[...R].reverse()]);
    for (let i = 10; i < plan.length; i += 10) stroke([L[i], R[i]], css("--ribbon-tick"), 2);      // 1 s marks
    stroke(side(1.35), css("--corridor"), 3); stroke(side(-1.35), css("--corridor"), 3);
  }
  // past path: cyan line from the trail into the car
  if (f.trail && f.trail.length) stroke([...f.trail, [0, 0]], css("--teal"), 2.5, [], 0.9);
  if (plan) stroke(plan, css("--teal"), 1.2, [], 0.6);

  // boxes: other actors and the ego, drawn far to near
  const kindColor = { vehicle: css("--red"), pedestrian: css("--amber"), cone: "#F07A2A" };
  const boxes = (f.actors || []).map((a) => {
    const ped = a.kind === "pedestrian";
    return { x: a.pos[0], y: a.pos[1], yaw: a.yaw, L: ped ? 0.6 : a.size[0], W: ped ? 0.6 : a.size[1],
             H: ped ? 1.8 : a.kind === "cone" ? 0.7 : 1.5, color: kindColor[a.kind] || css("--muted"), alpha: 0.7 };
  });
  boxes.push({ x: 1.35, y: 0, yaw: 0, L: 4.7, W: 1.9, H: 1.45, color: css("--text"), alpha: 0.22, ego: true });
  boxes.sort((a, b) => cam.depth([b.x, b.y]) - cam.depth([a.x, a.y]));
  for (const b of boxes) drawBox(g, cam, b, fillPoly);

  // fade the far distance into the horizon
  const fade = g.createLinearGradient(0, cam.horizon, 0, cam.horizon + h * 0.12);
  fade.addColorStop(0, css("--surface")); fade.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = fade; g.fillRect(0, cam.horizon - 1, w, h * 0.12 + 1);
}

function drawBox(g, cam, b, fillPoly) {
  const c = Math.cos(b.yaw), s = Math.sin(b.yaw), hl = b.L / 2, hw = b.W / 2;
  const base = [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]].map(([dx, dy]) => [b.x + dx * c - dy * s, b.y + dx * s + dy * c]);
  const bottom = base.map((p) => [p[0], p[1], 0]), top = base.map((p) => [p[0], p[1], b.H]);
  const faces = [0, 1, 2, 3].map((i) => [bottom[i], bottom[(i + 1) % 4], top[(i + 1) % 4], top[i]]);
  faces.sort((p, q) => cam.depth(q[0]) + cam.depth(q[2]) - cam.depth(p[0]) - cam.depth(p[2]));
  g.fillStyle = b.color; g.strokeStyle = b.color; g.lineWidth = 1.2;
  for (const face of [...faces, top]) {
    g.globalAlpha = b.alpha; fillPoly(face);
    g.globalAlpha = Math.min(1, b.alpha + 0.45); g.stroke();
  }
  g.globalAlpha = 1;
}

function drawBevTop(f) {
  const cv = $("bev"), dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = css("--surface"); g.fillRect(0, 0, w, h);
  const scale = (h / 52) * S.zoom;                    // px per metre
  const ox = w / 2, oy = h * 0.8;                       // ego position on screen
  const P = (x, y) => [ox - y * scale, oy - x * scale]; // ego frame: x forward (up), y left

  // range rings every 10 m
  g.strokeStyle = css("--grid"); g.lineWidth = 1;
  for (let r = 10; r <= 100; r += 10) { g.beginPath(); g.arc(ox, oy, r * scale, Math.PI, 2 * Math.PI); g.stroke(); }
  g.fillStyle = css("--muted"); g.font = `12px ${css("--font")}`;
  for (let r = 20; r <= 80; r += 20) { const u = ox + r * scale * 0.5, v = oy - r * scale * 0.866; if (v > 30 && u < w - 50) g.fillText(`${r} m`, u + 4, v - 4); }

  const line = (pts, color, width, dash = []) => {
    if (!pts || pts.length < 2) return;
    g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash); g.lineJoin = "round"; g.lineCap = "round";
    g.beginPath(); pts.forEach(([x, y], i) => { const [u, v] = P(x, y); i ? g.lineTo(u, v) : g.moveTo(u, v); }); g.stroke();
    g.setLineDash([]);
  };
  // road surface between edges (builtin backend gives exactly two edges)
  const edges = (f.lines || []).filter((l) => l.kind === "edge");
  if (edges.length === 2) {
    g.fillStyle = css("--road"); g.beginPath();
    edges[0].pts.forEach(([x, y], i) => { const [u, v] = P(x, y); i ? g.lineTo(u, v) : g.moveTo(u, v); });
    [...edges[1].pts].reverse().forEach(([x, y]) => { const [u, v] = P(x, y); g.lineTo(u, v); });
    g.closePath(); g.fill();
  }
  const poly = (pts) => {
    g.beginPath(); pts.forEach(([x, y], i) => { const [u, v] = P(x, y); i ? g.lineTo(u, v) : g.moveTo(u, v); });
    g.closePath(); g.fill();
  };
  // CARLA towns: one filled polygon per lane, junctions included, then crosswalks on top
  g.fillStyle = css("--road");
  for (const l of f.lines || []) if (l.kind === "surface") poly(l.pts);
  g.fillStyle = css("--crosswalk");
  for (const l of f.lines || []) if (l.kind === "crosswalk") poly(l.pts);
  for (const l of f.lines || []) {
    if (l.kind === "surface" || l.kind === "crosswalk") continue;
    if (l.kind === "edge") line(l.pts, css("--road-edge"), 2);
    else if (l.kind === "center") line(l.pts, css("--road-edge"), 1.5, [8, 10]);
    else if (l.kind === "junction") { g.globalAlpha = 0.35; line(l.pts, css("--muted"), 1); g.globalAlpha = 1; }
    else line(l.pts, css("--line"), 1, [3, 6]);
  }
  // actors
  const kindColor = { vehicle: css("--red"), pedestrian: css("--amber"), cone: "#F07A2A" };
  for (const a of f.actors || []) {
    const [u, v] = P(a.pos[0], a.pos[1]);
    const L = Math.max(a.size[0] * scale, 4), W = Math.max(a.size[1] * scale, 4);
    g.save(); g.translate(u, v); g.rotate(-a.yaw);
    g.fillStyle = kindColor[a.kind] || css("--muted");
    if (a.kind === "pedestrian") { g.beginPath(); g.arc(0, 0, Math.max(W, 6) / 2 + 1, 0, 2 * Math.PI); g.fill(); }
    else { g.beginPath(); g.roundRect(-W / 2, -L / 2, W, L, 3); g.fill(); }
    g.restore();
  }
  // history trail
  g.fillStyle = css("--teal");
  (f.trail || []).forEach(([x, y], i) => { if (i % 2) return; const [u, v] = P(x, y); g.globalAlpha = 0.3 + 0.6 * (i / 20); g.beginPath(); g.arc(u, v, 2.2, 0, 2 * Math.PI); g.fill(); });
  g.globalAlpha = 1;
  // alternative candidates, then the plan as lane paint with 1 s ticks
  (f.candidates || []).forEach((c) => { g.globalAlpha = 0.55; line([[0, 0], ...c], css("--teal"), 2); g.globalAlpha = 1; });
  if (f.plan && f.plan.length) {
    line([[0, 0], ...f.plan], css("--amber"), 5);
    g.fillStyle = css("--surface");
    f.plan.forEach(([x, y], i) => { if ((i + 1) % 10) return; const [u, v] = P(x, y); g.beginPath(); g.arc(u, v, 3.4, 0, 2 * Math.PI); g.fill(); g.strokeStyle = css("--amber"); g.lineWidth = 2; g.stroke(); });
  }
  // ego (rear axle at origin, ~4.7 m long)
  const [eu, ev] = P(0, 0);
  g.fillStyle = css("--text"); g.beginPath(); g.roundRect(eu - 0.95 * scale, ev - 3.7 * scale, 1.9 * scale, 4.7 * scale, 4); g.fill();
  g.fillStyle = css("--surface"); g.beginPath(); g.roundRect(eu - 0.7 * scale, ev - 2.6 * scale, 1.4 * scale, 1.0 * scale, 2); g.fill();
}

/* ---------------- metrics ---------------- */
const METRIC_LABELS = [
  ["distance_m", "Distance", (v) => `${fmt(v, 0)} m`],
  ["mean_speed_mps", "Mean speed", (v) => `${fmt(v * 3.6, 0)} km/h`],
  ["incidents", "Incidents", (v) => v],
  ["collisions", "Collisions", (v) => v],
  ["off_road_events", "Road departures", (v) => v],
  ["lane_offset_rms_m", "Lane offset (RMS)", (v) => `${fmt(v, 2)} m`],
  ["max_decel_mps2", "Hardest braking", (v) => `${fmt(v, 1)} m/s²`],
  ["mean_policy_latency_ms", "Mean latency", (v) => `${fmt(v, 0)} ms`],
  ["p95_policy_latency_ms", "95th pct. latency", (v) => `${fmt(v, 0)} ms`],
  ["reasoning_fraction", "Steps with reasoning", (v) => `${fmt(v * 100, 0)}%`],
];
function renderMetrics(target, m) {
  target.replaceChildren(...METRIC_LABELS.filter(([k]) => k in m).map(([k, label, f]) =>
    el("div", {}, el("dt", { text: label }), el("dd", { class: (["incidents", "collisions", "off_road_events"].includes(k) && m[k] > 0) ? "bad" : "", text: String(f(m[k])) }))));
}
function drawSpark() {
  const cv = $("spark"); if (!cv.clientWidth) return;
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const d = S.latencies; if (d.length < 2) return;
  const max = Math.max(...d) * 1.15 || 1;
  g.strokeStyle = css("--teal"); g.lineWidth = 1.5; g.beginPath();
  d.forEach((v, i) => { const x = (i / (d.length - 1)) * w, y = h - (v / max) * (h - 4) - 2; i ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.stroke();
  g.fillStyle = css("--muted"); g.font = `11px ${css("--font")}`; g.fillText(`${Math.round(max / 1.15)} ms max`, 4, 11);
}

/* ---------------- ask the model ---------------- */
function ask(q) {
  q = (q || "").trim(); if (!q) return;
  if (!(S.info && S.info.model && S.info.model.supports_vqa)) { toast("This model does not answer questions.", "warn"); return; }
  switchTab("chat");
  send("ask", { question: q, id: `q${Date.now()}` });
  $("ask-input").value = "";
}
function addChat(m) {
  let pair = document.querySelector(`[data-qid="${CSS.escape(m.id)}"]`);
  if (!pair) {
    pair = el("div", { "data-qid": m.id, class: "pair" },
      el("div", { class: "msg q" }, el("span", { class: "who", text: "Your question" }), m.question));
    chatAppend(pair);
  }
  pair.querySelector(".msg.a")?.remove();
  let a;
  if (m.pending) a = el("div", { class: "msg a typing", "aria-label": "Model is answering" }, el("span"), el("span"), el("span"));
  else if (m.error) a = el("div", { class: "msg a err", text: m.error });
  else a = el("div", { class: "msg a", text: m.answer || "(empty answer)" },
    el("small", { text: `${m.t != null ? `Scene at ${fmt(m.t)} s · ` : ""}${m.latency_ms != null ? `${m.latency_ms} ms` : ""}` }));
  pair.append(a);
  $("chat").scrollTop = $("chat").scrollHeight;
}

/* ---------------- log ---------------- */
function addLog(m, silent = false) {
  const t = new Date((m.time || Date.now() / 1000) * 1000);
  const li = el("li", { class: m.level || "info" }, el("time", { text: t.toLocaleTimeString([], { hour12: false }) }), el("span", { text: m.message }));
  $("log").append(li);
  while ($("log").childElementCount > 400) $("log").firstChild.remove();
  $("log").parentElement.scrollTop = $("log").parentElement.scrollHeight;
  const mini = $("loading-log");
  mini.append(li.cloneNode(true)); while (mini.childElementCount > 40) mini.firstChild.remove(); mini.scrollTop = mini.scrollHeight;
  if (!silent && m.level === "error" && !$("tab-log").classList.contains("active")) {
    S.unseenErrors++; $("log-badge").textContent = S.unseenErrors; $("log-badge").hidden = false;
  }
}

/* ---------------- summary ---------------- */
function showSummary(s) {
  const ok = s.status === "completed";
  $("summary-title").textContent = ok ? `Completed ${s.scenario.replaceAll("_", " ")}` :
    `Stopped: ${s.status === "collision" ? "collision" : "left the road"} at ${fmt(s.duration_s)} s`;
  renderMetrics($("summary-metrics"), s);
  const dlg = $("summary-modal"); if (!dlg.open) dlg.showModal();
}

/* ---------------- drive instructions ---------------- */
function setNavCurrent(text) {
  $("nav-current").textContent = text || "none";
  markNavPreset(text);
}
function sendNav() {
  const t = $("nav-input").value.trim();
  if (!t || $("nav-send").disabled) return;
  send("nav", { text: t });
  $("nav-input").value = "";
}

/* ---------------- tabs, controls, keys ---------------- */
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => { const on = t.dataset.tab === name; t.classList.toggle("active", on); t.setAttribute("aria-selected", on); });
  document.querySelectorAll(".tab-body").forEach((b) => b.classList.toggle("active", b.id === `tab-${name}`));
  if (name === "log") { S.unseenErrors = 0; $("log-badge").hidden = true; }
  if (name === "metrics") drawSpark();
}
function markNavPreset(text) {
  document.querySelectorAll("#nav-presets .chip").forEach((c) => c.classList.toggle("active", c.textContent === text));
}
function togglePlay() {
  if (S.state === "running") send("pause");
  else send("play");
}

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
$("btn-play").addEventListener("click", togglePlay);
$("btn-step").addEventListener("click", () => send("step"));
document.querySelectorAll(".signal-set button").forEach((b) => b.addEventListener("click", () => send("signal", { state: b.dataset.state })));
$("btn-reset").addEventListener("click", () => send("reset", { scenario: $("scenario-select").value }));
$("scenario-select").addEventListener("change", (e) => send("reset", { scenario: e.target.value }));
$("realtime").addEventListener("change", (e) => send("realtime", { value: e.target.checked }));
$("nav-form").addEventListener("submit", (e) => { e.preventDefault(); sendNav(); });
$("nav-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendNav(); } });
$("ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask($("ask-input").value); });
$("ask-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask($("ask-input").value); } });
$("zoom-in").addEventListener("click", () => { S.zoom = Math.min(4, S.zoom * 1.25); S.frame && drawBev(S.frame); });
$("zoom-out").addEventListener("click", () => { S.zoom = Math.max(0.4, S.zoom / 1.25); S.frame && drawBev(S.frame); });
$("bev-view").addEventListener("click", () => setBevMode(S.bevMode === "3d" ? "top" : "3d"));
{ let m = "3d"; try { m = localStorage.getItem("alpasandbox-bev") || "3d"; } catch (e) { /* ignore */ } setBevMode(m); }
$("summary-close").addEventListener("click", () => $("summary-modal").close());
$("summary-again").addEventListener("click", () => { $("summary-modal").close(); send("reset", { scenario: $("scenario-select").value, autoplay: true }); });
$("btn-help").addEventListener("click", () => $("help-modal").showModal());
$("help-close").addEventListener("click", () => $("help-modal").close());
$("btn-theme").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("alpasandbox-theme", next); } catch (e) { /* storage unavailable */ }
  if (S.frame) { drawBev(S.frame); drawDrive(S.frame.duration); } drawSpark();
});
window.addEventListener("resize", () => { if (S.frame) { drawBev(S.frame); drawDrive(S.frame.duration); } });
document.addEventListener("keydown", (e) => {
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName);
  if (e.key === "Escape") { if (S.focusedCam !== null) focusCam(S.focusedCam); if (typing) document.activeElement.blur(); return; }
  if (typing || e.ctrlKey || e.metaKey || e.altKey || document.querySelector("dialog[open]")) return;
  if (e.key === " ") { e.preventDefault(); togglePlay(); }
  else if (e.key === "n" || e.key === "N") send("step");
  else if (e.key === "r" || e.key === "R") send("reset", { scenario: $("scenario-select").value });
  else if (e.key === "g" || e.key === "G") { e.preventDefault(); switchTab("drive"); $("nav-input").focus(); }
  else if (e.key === "/") { e.preventDefault(); switchTab("chat"); $("ask-input").focus(); }
  else if (e.key === "?") $("help-modal").showModal();
  else if (/^[1-6]$/.test(e.key)) focusCam(CAM_ORDER[Number(e.key) - 1]);
});

try { const t = localStorage.getItem("alpasandbox-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }
connect();
