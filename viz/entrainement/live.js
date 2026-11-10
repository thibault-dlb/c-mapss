// Centre de contrôle de la recherche, EN DIRECT (un seul écran, 1920×1080, filmable).
// La page ne calcule rien d'autre que des mises en forme : elle relit le journal global (<base>/search.jsonl) et les journaux des
// entraînements en cours (<base>/slotK.json -> journal de chaque entraînement), écrits ligne par ligne par scripts/train.py --live.
//   ?base=demo     dossier à lire (défaut : runs). `demo` contient une SIMULATION (bandeau affiché : données fictives)
//   ?until=N       n'utilise que les N premiers événements du journal global (revoir un instant précis)
//   ?render=1      cache la barre de contrôle
(function () {
  "use strict";
  const W = 1920, H = 1080;
  // Charte « Radar doré » (mode sombre)
  const C = { bg: "#1C110B", panel: "#2A1B12", raised: "#38251A", border: "#4A3425", fg: "#F1E6CF", muted: "#B0A08A",
    cyan: "#3DD6C4", onCyan: "#0A1F1E", gold: "#E3B23C", danger: "#E5604D", warn: "#F28C28",
    map: "#130E04", land: "#221808", grid: "#4A3A12", mapText: "#C2A560", ring: "#A07A25", bronze: "#B07A1C", pale: "#F5D77E" };
  const F = { title: "Cinzel, 'Trajan Pro', serif", ui: "Inter, system-ui, sans-serif", mono: "'JetBrains Mono', ui-monospace, Consolas, monospace" };
  const params = new URLSearchParams(location.search);
  const BASE = params.get("base") || "runs";
  const UNTIL = params.has("until") ? +params.get("until") : Infinity;
  const RANK = d3.interpolateRgbBasis([C.bronze, C.gold, C.pale]);      // rampe d'or : bronze (moins bon) -> or pâle (meilleur)

  const num = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : String(x.toFixed(d)).replace(".", ",").replace("-", "−");
  const pct = (x) => `${Math.round(x * 100)} %`;
  const dur = (s) => { s = Math.max(0, Math.round(s)); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h ? `${h} h ${String(m).padStart(2, "0")}` : m ? `${m} min ${String(s % 60).padStart(2, "0")} s` : `${s} s`; };
  const hhmm = (d) => `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  const clip = (s, px, size, k = 0.54) => { const n = Math.max(4, Math.floor(px / (size * k))); return s.length > n ? s.slice(0, n - 1) + "…" : s; };

  const stageEl = document.getElementById("stage"), sourceEl = document.getElementById("source");
  sourceEl.textContent = `dossier : ${BASE}/`;
  const svg = d3.select("#stage").append("svg").attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
  const root = svg.append("g");

  // ------------------------------------------------------------------------------------------------ outils de dessin
  function T(g, x, y, s, o = {}) {
    const t = g.append("text").attr("x", x).attr("y", y).attr("fill", o.fill || C.fg).attr("font-size", o.size || 18)
      .attr("font-family", o.font || F.ui).attr("font-weight", o.weight || 400).attr("text-anchor", o.anchor || "start").text(s);
    if (o.font === F.mono) t.style("font-variant-numeric", "tabular-nums");
    if (o.ls) t.attr("letter-spacing", o.ls);
    return t;
  }
  const mono = (o) => ({ ...o, font: F.mono });
  function panel(g, x, y, w, h, title, sub) {
    g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h).attr("rx", 14).attr("fill", C.panel).attr("stroke", C.border);
    T(g, x + 22, y + 36, title, { font: F.title, weight: 600, size: 22, ls: "0.06em" });
    if (sub) T(g, x + 22, y + 58, sub, { size: 15, fill: C.muted });
  }
  function warnTriangle(g, x, y, s = 9) {                     // avertissement : toujours un triangle + un libellé
    g.append("path").attr("d", `M${x},${y - s} L${x + s * 1.1},${y + s * 0.8} L${x - s * 1.1},${y + s * 0.8} Z`).attr("fill", C.warn);
    g.append("rect").attr("x", x - 1).attr("y", y - s * 0.4).attr("width", 2).attr("height", s * 0.7).attr("fill", "#1C110B");
  }
  function arrow(g, x, y, down, col, s = 7) {
    g.append("path").attr("d", down ? `M${x - s},${y - s * 0.6} L${x + s},${y - s * 0.6} L${x},${y + s * 0.8} Z` : `M${x - s},${y + s * 0.6} L${x + s},${y + s * 0.6} L${x},${y - s * 0.8} Z`).attr("fill", col);
  }
  function legendItem(g, x, y, kind, label, col) {
    if (kind === "dot") g.append("circle").attr("cx", x + 6).attr("cy", y - 5).attr("r", 5).attr("fill", col);
    else if (kind === "ring") g.append("circle").attr("cx", x + 6).attr("cy", y - 5).attr("r", 7).attr("fill", "none").attr("stroke", col).attr("stroke-width", 2);
    else g.append("line").attr("x1", x).attr("x2", x + 22).attr("y1", y - 5).attr("y2", y - 5).attr("stroke", col).attr("stroke-width", 3).attr("stroke-dasharray", kind === "dash" ? "6 4" : null);
    T(g, x + (kind === "dot" || kind === "ring" ? 18 : 30), y, label, { size: 15, fill: C.muted });
    return x + (kind === "dot" || kind === "ring" ? 18 : 30) + label.length * 8.2 + 22;
  }

  // ------------------------------------------------------------------------------------------------ état construit depuis les journaux
  function buildGlobal(events) {
    const st = { n: events.length, plan: null, phases: [], total: 0, progress: null, baseline: null, phase: null, gens: [], starts: {}, runs: [],
      confirm: null, cv: null, final: null, ended: false, members: {}, t: events.length ? events[events.length - 1].t : 0 };
    for (const e of events) {
      if (e.progress) st.progress = e.progress;
      if (e.type === "plan") st.plan = e;
      else if (e.type === "plan_update") { st.phases = e.phases; st.total = e.total; st.progress = e.progress || st.progress; }
      else if (e.type === "baseline") st.baseline = e;
      else if (e.type === "phase") st.phase = e;
      else if (e.type === "generation_start") st.starts[e.gen] = e;
      else if (e.type === "run_done") st.runs.push(e);
      else if (e.type === "generation_end") st.gens.push(e);
      else if (e.type === "confirm_end") st.confirm = e;
      else if (e.type === "cv_end") st.cv = e;
      else if (e.type === "final_end") st.final = e;
      else if (e.type === "end") st.ended = true;
    }
    for (const g of st.gens) for (const m of g.ranking) if (!st.members[m.cid]) st.members[m.cid] = { ...m, gen: g.gen };
    return st;
  }
  function parseRun(text) {
    if (!text.endsWith("\n")) text = text.slice(0, text.lastIndexOf("\n") + 1);
    const r = { meta: null, probe: null, epochs: [], best: null, batch: null, preds: {}, ended: false, early: null };
    for (const l of text.split("\n")) {
      if (!l) continue;
      const e = JSON.parse(l);
      if (e.type === "meta") r.meta = e; else if (e.type === "probe") r.probe = e; else if (e.type === "epoch") r.epochs.push(e);
      else if (e.type === "best") r.best = e; else if (e.type === "batch") r.batch = e; else if (e.type === "prediction") r.preds[e.epoch] = e.pred;
      else if (e.type === "early_stop") r.early = e; else if (e.type === "end") r.ended = true;
    }
    return r;
  }

  // ------------------------------------------------------------------------------------------------ en-tête, KPI, phases
  function drawHeader(g, st, now) {
    for (const r of [60, 110, 160, 210, 260]) g.append("circle").attr("cx", 1830).attr("cy", 20).attr("r", r).attr("fill", "none").attr("stroke", C.ring).attr("stroke-opacity", 0.16);
    T(g, 42, 58, "Recherche de la meilleure IA", { font: F.title, weight: 600, size: 38, ls: "0.06em" });
    const demo = st.plan && st.plan.demo, over = st.ended;
    const label = !st.plan ? "en attente" : over ? "terminé" : (st.phase ? st.phase.name : "démarrage");
    const x0 = 730;
    g.append("rect").attr("x", x0).attr("y", 30).attr("width", 18 + label.length * 10.5 + 30).attr("height", 36).attr("rx", 18).attr("fill", over ? C.gold : C.cyan);
    T(g, x0 + 18, 54, label, { size: 17, weight: 600, fill: over ? "#1C110B" : C.onCyan });
    if (demo) {
      const bx = 930;
      g.append("rect").attr("x", bx).attr("y", 24).attr("width", 520).attr("height", 48).attr("rx", 10).attr("fill", "#3A1F0A").attr("stroke", C.warn).attr("stroke-width", 2);
      warnTriangle(g, bx + 28, 48, 11);
      T(g, bx + 52, 44, "SIMULATION — données fictives", { size: 19, weight: 600, fill: C.fg });
      T(g, bx + 52, 63, "aucun entraînement réel n'a été lancé", { size: 14, fill: C.muted });
    }
    T(g, 1878, 46, hhmm(now), mono({ size: 34, weight: 500, anchor: "end" }));
    T(g, 1878, 68, now.toLocaleDateString("fr-FR", { weekday: "long", day: "numeric", month: "long" }), { size: 14, fill: C.muted, anchor: "end" });
  }

  function kpiCard(g, x, y, w, h, title) {
    g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h).attr("rx", 14).attr("fill", C.panel).attr("stroke", C.border);
    T(g, x + 22, y + 30, title, { size: 15, fill: C.muted, weight: 500, ls: "0.04em" });
  }

  function best(st) {                         // « meilleure IA » : la mesure la plus avancée dont on dispose, et la baseline comparable
    const b = st.baseline;
    if (st.final) return { v: st.final.table.total.rmse_mean, ref: b && b.baseline_test_rmse, what: "RMSE sur le jeu de test, 5 graines", refLabel: "baseline test" };
    if (st.cv) return { v: st.cv.rmse_mean, ref: st.cv.baseline, what: "RMSE en validation croisée (5 plis × 5 graines)", refLabel: "baseline" };
    if (st.confirm && st.confirm.ranking.length) return { v: st.confirm.ranking[0].rmse_mean, ref: b && b.baseline_val_rmse_mean, what: "RMSE sur 5 plis de validation", refLabel: "baseline" };
    if (st.gens.length) { const nf = st.plan ? st.plan.screen_folds.length : 2;
      return { v: st.gens[st.gens.length - 1].best_so_far, ref: b && b.screen_rmse, what: `RMSE sur ${nf} plis de validation (tri)`, refLabel: "baseline, mêmes plis" }; }
    return null;
  }

  function drawKpis(g, st, now, fetchedAt) {
    const x0 = 42, y = 100, w = 441, h = 150, gap = 24, p = st.progress || { done: 0, total: st.total || 0, elapsed: 0, eta_seconds: 0 };
    const tick = st.ended || !st.plan ? 0 : (now - fetchedAt) / 1000;
    const elapsed = p.elapsed + tick, eta = Math.max(0, p.eta_seconds - tick), frac = p.total ? p.done / p.total : 0;
    // 1. avancement (anneau « radar »)
    kpiCard(g, x0, y, w, h, "AVANCEMENT");
    const cx = x0 + 82, cy = y + 88;
    for (const r of [56, 42, 28]) g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", r).attr("fill", "none").attr("stroke", C.ring).attr("stroke-opacity", 0.35);
    const ang = frac * 2 * Math.PI;
    g.append("path").attr("transform", `translate(${cx},${cy})`).attr("d", d3.arc()({ innerRadius: 0, outerRadius: 56, startAngle: Math.max(0, ang - 0.9), endAngle: ang })).attr("fill", C.cyan).attr("fill-opacity", 0.22);
    g.append("path").attr("transform", `translate(${cx},${cy})`).attr("d", d3.arc().cornerRadius(3)({ innerRadius: 47, outerRadius: 56, startAngle: 0, endAngle: Math.max(0.001, ang) })).attr("fill", C.gold);
    T(g, cx, cy + 8, pct(frac), mono({ size: 22, weight: 500, anchor: "middle" }));
    T(g, x0 + 168, y + 80, `${p.done}`, mono({ size: 44, weight: 500 }));
    T(g, x0 + 168 + String(p.done).length * 26 + 8, y + 80, `/ ${p.total}`, mono({ size: 24, fill: C.muted }));
    T(g, x0 + 168, y + 108, "entraînements terminés", { size: 16, fill: C.muted });
    T(g, x0 + 168, y + 132, st.plan ? `${st.plan.workers} en parallèle · ${st.plan.threads} threads chacun` : "", { size: 14, fill: C.muted });
    // 2. temps
    let x = x0 + w + gap; kpiCard(g, x, y, w, h, "TEMPS");
    T(g, x + 22, y + 78, st.ended ? "terminé" : dur(eta), mono({ size: 40, weight: 500 }));
    T(g, x + 22, y + 104, st.ended ? "durée totale" : "restant (estimation)", { size: 16, fill: C.muted });
    T(g, x + 22, y + 134, `écoulé ${dur(elapsed)}`, mono({ size: 17 }));
    if (!st.ended && p.total && p.done < p.total) T(g, x + w - 22, y + 134, `fin vers ${hhmm(new Date(now + eta * 1000))}`, mono({ size: 17, anchor: "end" }));
    // 3. meilleure IA vs baseline
    x += w + gap; kpiCard(g, x, y, w, h, "MEILLEURE IA JUSQU'ICI");
    const b = best(st);
    if (!b) T(g, x + 22, y + 90, "—", mono({ size: 40 }));
    else {
      T(g, x + 22, y + 78, num(b.v, 2), mono({ size: 44, weight: 500 }));
      T(g, x + 22 + num(b.v, 2).length * 26 + 10, y + 78, "cycles", { size: 17, fill: C.muted });
      T(g, x + 22, y + 100, b.what, { size: 14, fill: C.muted });
      if (b.ref) {
        const d = b.v - b.ref, ok = d < 0;
        if (ok) { arrow(g, x + 30, y + 130, true, C.cyan); T(g, x + 48, y + 135, `${num(Math.abs(d), 2)} de moins que la ${b.refLabel} (${num(b.ref, 2)})`, { size: 15, weight: 600 }); }
        else { warnTriangle(g, x + 30, y + 130, 8); T(g, x + 48, y + 135, `${num(Math.abs(d), 2)} de plus que la ${b.refLabel} (${num(b.ref, 2)})`, { size: 15, weight: 600, fill: C.warn }); }
      }
    }
    // 4. génération
    x += w + gap; kpiCard(g, x, y, w, h, "GÉNÉRATION");
    const G = st.plan ? st.plan.generations : 0, started = Object.keys(st.starts).length, cur = Math.max(0, started - 1), done = st.gens.length;
    const inSearch = started && !st.phase || (st.phase && st.phase.name === "tri");
    if (!G) T(g, x + 22, y + 90, "—", mono({ size: 40 }));
    else {
      T(g, x + 22, y + 78, inSearch ? `${cur + 1}` : `${G}`, mono({ size: 44, weight: 500 }));
      T(g, x + 22 + (inSearch ? String(cur + 1).length : String(G).length) * 26 + 8, y + 78, `/ ${G}`, mono({ size: 24, fill: C.muted }));
      T(g, x + w - 22, y + 78, inSearch ? "en cours" : "recherche terminée", { size: 15, fill: C.muted, anchor: "end" });
      const last = st.gens[done - 1];
      if (last && last.delta !== null && last.delta !== undefined) {
        if (last.delta < -1e-9) { arrow(g, x + 30, y + 112, true, C.cyan); T(g, x + 48, y + 117, `gén ${last.gen} : ${num(last.delta, 2)} sur le meilleur`, { size: 15, weight: 600 }); }
        else T(g, x + 22, y + 117, `gén ${last.gen} : pas de progrès sur le meilleur`, { size: 15, fill: C.muted });
      } else if (last) T(g, x + 22, y + 117, `gén 0 : point de départ ${num(last.best.fitness, 2)}`, { size: 15, fill: C.muted });
      const gain = st.gens.length ? st.gens[0].best.fitness - st.gens[done - 1].best_so_far : 0;
      if (done) T(g, x + 22, y + 140, `depuis la gén 0 : ${gain > 1e-9 ? "−" + num(gain, 2) : "aucun gain"}`, { size: 14, fill: C.muted });
    }
  }

  function drawPhases(g, st) {
    const x0 = 42, w = 1836, y = 270, h = 14, ph = st.phases, tot = st.total || 1;
    if (!ph.length) { T(g, x0, y + 12, "plan des étapes : en attente…", { size: 15, fill: C.muted }); return; }
    let cum = 0; const done = st.progress ? st.progress.done : 0;
    const names = { "validation croisée": "validation croisée", confirmation: "confirmation", final: "final" };
    ph.forEach((p, i) => {
      const sx = x0 + (cum / tot) * w, sw = (p.runs / tot) * w - 4;
      g.append("rect").attr("x", sx).attr("y", y).attr("width", sw).attr("height", h).attr("rx", 7).attr("fill", C.raised);
      const f = Math.max(0, Math.min(1, (done - cum) / p.runs));
      if (f > 0) g.append("rect").attr("x", sx).attr("y", y).attr("width", sw * f).attr("height", h).attr("rx", 7).attr("fill", f >= 1 ? C.gold : C.cyan);
      T(g, sx + sw / 2, y + 36, names[p.name] || p.name, { size: 15, fill: f > 0 ? C.fg : C.muted, anchor: "middle" });
      cum += p.runs;
    });
  }

  // ------------------------------------------------------------------------------------------------ graphe des générations
  function drawGenerations(g, st) {
    const px = 42, py = 336, pw = 1098, ph = 358;
    panel(g, px, py, pw, ph, "Les générations", "chaque point est un réglage de l'IA ; plus bas = erreur plus faible");
    let lx = px + 520; const ly = py + 36;
    lx = legendItem(g, lx, ly, "line", "meilleur", C.cyan); lx = legendItem(g, lx, ly, "dash", "moyenne", C.muted);
    legendItem(g, lx, ly, "dash", "baseline", C.pale);
    legendItem(g, px + 520, py + 58, "dot", "membre (or pâle = mieux classé)", C.gold); legendItem(g, px + 820, py + 58, "ring", "élite gardée", C.gold);
    const G = st.plan ? st.plan.generations : 8, ref = st.baseline && st.baseline.screen_rmse;
    const x0 = px + 78, x1 = px + pw - 170, y0 = py + 112, y1 = py + ph - 64;
    g.append("rect").attr("x", x0 - 14).attr("y", y0 - 14).attr("width", x1 - x0 + 28).attr("height", y1 - y0 + 28).attr("rx", 8).attr("fill", C.map);
    if (!st.gens.length) { T(g, (x0 + x1) / 2, (y0 + y1) / 2, st.plan ? "la génération 0 est en cours d'évaluation…" : "en attente du journal…", { size: 20, fill: C.mapText, anchor: "middle" }); return; }
    const vals = st.gens.flatMap((e) => e.ranking.map((m) => m.fitness)).concat(ref ? [ref] : []);
    const sx = d3.scaleLinear().domain([-0.4, Math.max(G - 1, 1) + 0.4]).range([x0, x1]);
    const sy = d3.scaleLinear().domain([d3.min(vals) - 0.4, d3.max(vals) + 0.4]).range([y1, y0]).nice(5);
    for (const v of sy.ticks(5)) { g.append("line").attr("x1", x0 - 14).attr("x2", x1 + 14).attr("y1", sy(v)).attr("y2", sy(v)).attr("stroke", C.grid);
      T(g, x0 - 22, sy(v) + 5, num(v, 0), mono({ size: 15, fill: C.mapText, anchor: "end" })); }
    for (let k = 0; k < G; k++) T(g, sx(k), y1 + 34, `gén ${k}`, { size: 15, fill: k < st.gens.length ? C.fg : C.mapText, anchor: "middle" });
    T(g, x0 - 14, y0 - 22, "erreur (RMSE, cycles)", { size: 14, fill: C.mapText });
    if (ref) { g.append("line").attr("x1", x0 - 14).attr("x2", x1 + 14).attr("y1", sy(ref)).attr("y2", sy(ref)).attr("stroke", C.pale).attr("stroke-width", 2).attr("stroke-dasharray", "8 6");
      T(g, x1 + 26, sy(ref) - 6, "baseline", { size: 15, fill: C.fg }); T(g, x1 + 26, sy(ref) + 13, num(ref, 2), mono({ size: 15, fill: C.muted })); }
    const line = d3.line().x((d) => sx(d[0])).y((d) => sy(d[1]));
    g.append("path").attr("d", line(st.gens.map((e) => [e.gen, e.mean]))).attr("fill", "none").attr("stroke", C.muted).attr("stroke-width", 2).attr("stroke-dasharray", "7 5");
    g.append("path").attr("d", line(st.gens.map((e) => [e.gen, e.best_so_far]))).attr("fill", "none").attr("stroke", C.cyan).attr("stroke-width", 3.5).attr("stroke-linejoin", "round");
    for (const e of st.gens) {
      const n = e.ranking.length;
      e.ranking.forEach((m, i) => {
        const jx = (i % 2 ? 1 : -1) * (6 + (i % 3) * 3) * (e.ranking.length > 1 ? 1 : 0);          // léger décalage horizontal : lisibilité des points proches
        const c = g.append("circle").attr("cx", sx(e.gen) + jx).attr("cy", sy(m.fitness)).attr("r", 6).attr("fill", RANK(1 - i / Math.max(1, n - 1))).attr("stroke", C.map).attr("stroke-width", 2);
        c.append("title").text(`${m.cid} · RMSE ${num(m.fitness, 2)}${m.elite ? " · élite (réglage gardé de la génération précédente)" : ""}\n${m.label}${m.parents && m.parents.length ? `\nparents : ${m.parents.join(", ")}` : ""}`);
        if (m.elite) g.append("circle").attr("cx", sx(e.gen) + jx).attr("cy", sy(m.fitness)).attr("r", 10).attr("fill", "none").attr("stroke", C.gold).attr("stroke-width", 2);
      });
      g.append("circle").attr("cx", sx(e.gen)).attr("cy", sy(e.best_so_far)).attr("r", 5.5).attr("fill", C.cyan).attr("stroke", C.map).attr("stroke-width", 2);
      if (e.delta !== null && e.delta < -1e-9) { arrow(g, sx(e.gen) - 26, y1 + 52, true, C.cyan, 6); T(g, sx(e.gen) - 14, y1 + 58, num(e.delta, 2), mono({ size: 15, weight: 500 })); }
    }
    const cur = Object.keys(st.starts).length - 1;                      // génération en cours : résultats partiels (un pli sur deux, par exemple)
    if (cur >= st.gens.length && st.starts[cur]) {
      const part = {};
      for (const r of st.runs) if (r.stage === "search" && r.gen === cur) (part[r.cid] = part[r.cid] || []).push(r.rmse);
      for (const [cid, a] of Object.entries(part)) g.append("circle").attr("cx", sx(cur)).attr("cy", sy(d3.mean(a))).attr("r", 5).attr("fill", "none").attr("stroke", C.cyan).attr("stroke-width", 2)
        .append("title").text(`${cid} · en cours : ${a.length} pli(s) terminé(s), RMSE ${num(d3.mean(a), 2)} jusqu'ici`);
      T(g, sx(cur), y0 - 2, "en cours", { size: 14, fill: C.cyan, anchor: "middle" });
    }
    const lastE = st.gens[st.gens.length - 1];
    T(g, x1 + 26, sy(lastE.best_so_far) + 5, `meilleur ${num(lastE.best_so_far, 2)}`, mono({ size: 14, weight: 500 }));
  }

  // ------------------------------------------------------------------------------------------------ entraînements en parallèle
  function spark(g, x, y, w, h, vals, ref) {
    g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h).attr("rx", 6).attr("fill", C.map);
    if (vals.length < 1) return;
    const all = vals.concat(ref ? [ref] : []), sx = d3.scaleLinear().domain([0, Math.max(6, vals.length - 1)]).range([x + 8, x + w - 8]),
      sy = d3.scaleLinear().domain([d3.min(all) - 0.5, d3.max(all) + 0.5]).range([y + h - 6, y + 6]);
    if (ref) g.append("line").attr("x1", x + 4).attr("x2", x + w - 4).attr("y1", sy(ref)).attr("y2", sy(ref)).attr("stroke", C.pale).attr("stroke-dasharray", "5 4").attr("stroke-opacity", 0.8);
    g.append("path").attr("d", d3.line().x((d, i) => sx(i)).y((d) => sy(d))(vals)).attr("fill", "none").attr("stroke", C.cyan).attr("stroke-width", 2.5);
    const bi = vals.indexOf(d3.min(vals));
    g.append("circle").attr("cx", sx(bi)).attr("cy", sy(vals[bi])).attr("r", 4.5).attr("fill", C.gold).attr("stroke", C.map).attr("stroke-width", 1.5);
  }

  function drawSlots(g, st, slots, o = {}) {
    const px = o.px ?? 1164, py = o.py ?? 336, pw = 714, ph = 358, n = o.n ?? ((st.plan && st.plan.workers) || 4), rh = 66;
    panel(g, px, py, pw, ph, o.title || "Entraînements en parallèle", o.sub || `${n} en même temps · erreur de validation à chaque époque`);
    const ref = st.baseline && st.baseline.screen_rmse;
    for (let k = 0; k < n; k++) {
      const y = py + 70 + k * rh, r = slots[k];
      g.append("line").attr("x1", px + 20).attr("x2", px + pw - 20).attr("y1", y - 6).attr("y2", y - 6).attr("stroke", C.border);
      if (!r || !r.meta) { T(g, px + 22, y + 34, `machine ${k + 1} : en attente d'un entraînement`, { size: 16, fill: C.muted }); continue; }
      const run = r.meta.run || {}, running = !r.ended, ep = r.epochs.length, last = r.epochs[ep - 1];
      const bestRmse = r.epochs.length ? d3.min(r.epochs, (e) => e.rmse) : null, bestEp = bestRmse === null ? 0 : r.epochs.find((e) => e.rmse === bestRmse).epoch;
      const stage = { search: "tri", cv: "validation croisée", final: "final" }[run.stage] || run.stage;
      g.append("circle").attr("cx", px + 30).attr("cy", y + 14).attr("r", 6).attr("fill", running ? C.cyan : C.gold);
      T(g, px + 44, y + 19, running ? "en cours" : "terminé", { size: 14, weight: 600, fill: C.muted });
      T(g, px + 44, y + 40, clip(run.cid || "", 190, 17, 0.62), mono({ size: 17, weight: 500 }));
      T(g, px + 44, y + 58, `${stage} · ${run.fold === null || run.fold === undefined ? "tous les plis" : "pli " + run.fold} · graine ${r.meta.config.seed}`, { size: 14, fill: C.muted });
      spark(g, px + 262, y - 2, 170, 54, r.epochs.map((e) => e.rmse), ref);
      T(g, px + 450, y + 18, `époque ${running ? ep + 1 : ep}`, mono({ size: 16 }));
      if (bestRmse !== null) T(g, px + 450, y + 40, `meilleur ${num(bestRmse, 2)}`, mono({ size: 16, weight: 500 }));
      if (bestRmse !== null) T(g, px + 450, y + 58, `à l'époque ${bestEp}`, { size: 14, fill: C.muted });
      if (running && r.batch && r.batch.epoch === ep + 1) {
        g.append("rect").attr("x", px + 590).attr("y", y + 8).attr("width", 100).attr("height", 8).attr("rx", 4).attr("fill", C.raised);
        g.append("rect").attr("x", px + 590).attr("y", y + 8).attr("width", 100 * r.batch.batch / r.batch.n_batches).attr("height", 8).attr("rx", 4).attr("fill", C.cyan);
        T(g, px + 590, y + 34, "lots de l'époque", { size: 13, fill: C.muted });
      } else if (r.early) T(g, px + 590, y + 20, "arrêt anticipé", { size: 14, fill: C.muted });
    }
  }

  // ------------------------------------------------------------------------------------------------ classement et évolution
  function drawRanking(g, st) {
    const px = 42, py = 712, pw = 578, ph = 334;
    panel(g, px, py, pw, ph, "Classement", "les 8 meilleurs réglages (erreur de tri)");
    const ref = st.baseline && st.baseline.screen_rmse;
    const list = Object.values(st.members).sort((a, b) => a.fitness - b.fitness).slice(0, 8);
    if (!list.length) { T(g, px + 22, py + 110, "aucun réglage évalué pour l'instant", { size: 17, fill: C.muted }); return; }
    ["#", "réglage", "RMSE", "vs baseline", "gén"].forEach((t, i) => T(g, px + [22, 56, 318, 456, 548][i], py + 84, t, { size: 14, fill: C.muted, anchor: i >= 2 ? "end" : "start" }));
    list.forEach((m, i) => {
      const y = py + 112 + i * 26.5, ok = ref && m.fitness < ref;
      if (i === 0) g.append("rect").attr("x", px + 12).attr("y", y - 19).attr("width", pw - 24).attr("height", 26).attr("rx", 6).attr("fill", C.raised);
      T(g, px + 22, y, String(i + 1), mono({ size: 15, fill: C.muted }));
      T(g, px + 56, y, clip(m.cid, 120, 15, 0.62), mono({ size: 15, weight: 500 })).append("title").text(m.label);
      T(g, px + 318, y, num(m.fitness, 2), mono({ size: 15, anchor: "end" }));
      if (ref) { if (ok) arrow(g, px + 384, y - 5, true, C.cyan, 5); else warnTriangle(g, px + 384, y - 5, 6);
        T(g, px + 456, y, `${num(m.fitness - ref, 2)}`, mono({ size: 15, anchor: "end", fill: ok ? C.fg : C.warn })); }
      T(g, px + 548, y, `${m.gen}`, mono({ size: 15, fill: C.muted, anchor: "end" }));
    });
    if (ref) T(g, px + 22, py + ph - 14, `écart négatif = meilleur que la baseline (${num(ref, 2)}, mêmes plis)`, { size: 14, fill: C.muted });
  }

  function drawEvolution(g, st, evo) {
    const px = 644, py = 712, pw = 1234, ph = 334;
    panel(g, px, py, pw, ph, "L'IA évolue", "un même moteur : cycles restants avant la panne (vertical) au fil de sa vie (horizontal), prédits par le meilleur réglage de chaque génération");
    legendItem(g, px + 800, py + 36, "dash", "réalité", C.muted); legendItem(g, px + 900, py + 36, "line", "prédiction de l'IA", C.cyan);
    const G = st.plan ? st.plan.generations : 8, cw = (pw - 44 - 3 * 14) / 4, chh = 100;
    for (let k = 0; k < G; k++) {
      const col = k % 4, row = Math.floor(k / 4), x = px + 22 + col * (cw + 14), y = py + 92 + row * (chh + 40);
      const e = st.gens[k], d = e && evo[e.best.cid];
      g.append("rect").attr("x", x).attr("y", y).attr("width", cw).attr("height", chh).attr("rx", 6).attr("fill", C.map);
      T(g, x, y - 10, `gén ${k}`, { size: 16, weight: 600 });
      if (e) T(g, x + cw, y - 10, `${e.best.cid} · ${num(e.best.fitness, 2)}`, mono({ size: 14, fill: C.muted, anchor: "end" }));
      if (!e) { T(g, x + cw / 2, y + chh / 2 + 5, "pas encore", { size: 15, fill: C.mapText, anchor: "middle" }); continue; }
      if (!d) { T(g, x + cw / 2, y + chh / 2 + 5, "courbe non disponible", { size: 14, fill: C.mapText, anchor: "middle" }); continue; }
      const xs = d3.scaleLinear().domain(d3.extent(d.probe.end_cycle)).range([x + 8, x + cw - 8]), ymax = d3.max(d.probe.true) * 1.05,
        ys = d3.scaleLinear().domain([0, ymax]).range([y + chh - 8, y + 8]);
      const ln = d3.line().x((v, i) => xs(d.probe.end_cycle[i])).y((v) => ys(v));
      g.append("path").attr("d", ln(d.probe.true)).attr("fill", "none").attr("stroke", C.muted).attr("stroke-width", 2).attr("stroke-dasharray", "5 4");
      g.append("path").attr("d", ln(d.pred)).attr("fill", "none").attr("stroke", C.cyan).attr("stroke-width", 2.2).attr("stroke-linejoin", "round")
        .append("title").text(`gén ${k} · ${e.best.cid} · époque ${d.epoch}`);
    }
  }

  // Essai isolé (scripts/train.py --smoke --live) : pas de journal global de recherche, on montre les entraînements réels en cours.
  function drawSolo() {
    const ks = Object.keys(S.slots).map(Number).sort((a, b) => a - b), now = new Date();
    T(root, 42, 58, "Essai d'entraînement", { font: F.title, weight: 600, size: 38, ls: "0.06em" });
    T(root, 1878, 46, hhmm(now), mono({ size: 34, weight: 500, anchor: "end" }));
    warnTriangle(root, 62, 112, 11);
    T(root, 84, 118, "Pas de recherche en cours : essai isolé (journaux lus dans " + BASE + "/)", { size: 19, fill: C.muted });
    ks.forEach((k, i) => drawSlots(root, { plan: null, baseline: null }, { 0: S.slots[k] }, { px: 42 + (i % 2) * 740, py: 150 + Math.floor(i / 2) * 380, n: 1,
      title: `Machine ${k + 1}`, sub: "erreur de validation à chaque époque" }));
    const r = S.slots[ks[0]], run = r && r.meta && r.meta.run;
    if (r && r.meta) {
      T(root, 42, 560, "Réseau", { font: F.title, weight: 600, size: 22, ls: "0.06em" });
      T(root, 42, 592, `${r.meta.n_params.toLocaleString("fr-FR")} paramètres · ${r.meta.device} · ${r.meta.threads} threads · ${r.epochs.length} époque(s) terminée(s)`, { size: 18, fill: C.muted });
      const last = r.epochs[r.epochs.length - 1];
      if (last) T(root, 42, 630, `dernière époque : erreur de validation ${num(last.rmse, 2)} cycles · exactitude du régime ${pct(last.regime_acc)} · ${num(last.seconds, 1)} s`, mono({ size: 18 }));
      if (r.meta.reference && r.meta.reference.baseline_val_rmse_fold) T(root, 42, 664, `baseline sur ce pli : ${num(r.meta.reference.baseline_val_rmse_fold, 2)} cycles`, { size: 18, fill: C.muted });
    }
    document.getElementById("state").textContent = "essai isolé · en direct";
  }

  // ------------------------------------------------------------------------------------------------ dessin d'ensemble
  const S = { g: null, slots: {}, evo: {}, fetchedAt: performance.now(), err: null };
  function draw() {
    root.selectAll("*").remove();
    root.append("rect").attr("width", W).attr("height", H).attr("fill", C.bg);
    const now = Date.now(), st = S.g;
    if (!st && S.solo && Object.keys(S.slots).length) { drawSolo(); return; }
    if (!st) {
      T(root, W / 2, 500, S.err || "En attente du journal de l'entraînement…", { font: F.title, weight: 600, size: 34, anchor: "middle", ls: "0.06em" });
      warnTriangle(root, W / 2, 560, 14); T(root, W / 2 + 28, 568, `rien à lire dans ${BASE}/ : lancer scripts/train.py --live`, { size: 20, fill: C.muted });
      return;
    }
    try {
      drawHeader(root, st, new Date(now)); drawKpis(root, st, performance.now(), S.fetchedAt); drawPhases(root, st);
      drawGenerations(root, st); drawSlots(root, st, S.slots); drawRanking(root, st); drawEvolution(root, st, S.evo);
    } catch (err) { console.error(err); window.ERROR = String(err.stack || err); }
    document.getElementById("state").textContent = st.ended ? `terminé · ${st.n} événements` : `en direct · ${st.n} événements`;
  }

  // ------------------------------------------------------------------------------------------------ lecture des journaux
  const cache = {};                                      // journal d'entraînement : relu tant qu'il n'est pas terminé
  async function runJournal(path) {
    const c = cache[path];
    if (c && c.ended) return c;
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) return c || null;
    const r = parseRun(await res.text());
    cache[path] = r; return r;
  }
  async function loadAux(st) {
    const n = (st && st.plan && st.plan.workers) || 4;
    await Promise.all(Array.from({ length: n }, async (_, k) => {
      try {
        const p = await fetch(`${BASE}/slot${k}.json`, { cache: "no-store" });
        if (!p.ok) return;
        const r = await runJournal((await p.json()).journal);
        if (r) S.slots[k] = r;
      } catch (e) { /* pointeur en cours d'écriture : on réessaie au prochain tour */ }
    }));
    if (!st) return;
    await Promise.all(st.gens.map(async (e) => {
      if (S.evo[e.best.cid]) return;
      const run = st.runs.find((r) => r.stage === "search" && r.cid === e.best.cid && r.fold === 0);
      if (!run || !run.journal) return;
      try {
        const r = await runJournal(run.journal);
        if (r && r.ended && r.probe) { const ep = run.best_epoch, pred = r.preds[ep] || r.preds[Math.max(...Object.keys(r.preds).map(Number))];
          if (pred) S.evo[e.best.cid] = { probe: r.probe, pred, epoch: r.preds[ep] ? ep : null }; }
      } catch (err) { /* journal introuvable (réglage repris d'un ancien résultat) : pas de courbe */ }
    }));
  }
  let timer = null;
  async function poll() {
    try {
      const res = await fetch(`${BASE}/search.jsonl`, { cache: "no-store" });
      if (!res.ok) { S.err = null; S.g = null; S.solo = true; await loadAux(null); }
      else {
        let text = await res.text();
        if (!text.endsWith("\n")) text = text.slice(0, text.lastIndexOf("\n") + 1);          // dernière ligne en cours d'écriture
        const events = text.split("\n").filter(Boolean).slice(0, UNTIL).map((l) => JSON.parse(l));
        events.forEach((e, i) => { if (e.seq !== i) throw new Error(`journal non contigu à l'événement n°${i}`); });
        S.g = events.length ? buildGlobal(events) : null; S.fetchedAt = performance.now();
        window.STATE = S.g;
        if (S.g) await loadAux(S.g);
        if (S.g && (S.g.ended || events.length >= UNTIL)) { draw(); window.READY = true; return; }
      }
      S.err = null;
    } catch (err) { S.err = String(err.message); window.ERROR = S.err; }
    draw(); window.READY = true;
    timer = setTimeout(poll, 1500);
  }

  function fit() {
    const rec = document.body.classList.contains("recording");
    const reserve = rec || params.get("render") === "1" ? 0 : 64;
    const s = Math.min(innerWidth / W, (innerHeight - reserve) / H);
    stageEl.style.transform = `translate(${(innerWidth - W * s) / 2}px,${(innerHeight - reserve - H * s) / 2}px) scale(${s})`;
  }
  if (params.get("render") === "1") document.body.classList.add("render");
  addEventListener("resize", fit); fit();
  VizRecord.attach({ button: document.getElementById("rec"), countdownEl: document.getElementById("countdown"),
    msgEl: document.getElementById("msg"), name: "recherche", onLayout: fit });
  setInterval(() => { if (S.g && !S.g.ended) draw(); }, 1000);                           // horloge et temps restant vivent entre deux lectures
  draw();
  (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(() => { draw(); poll(); });
})();
