// Tableau de bord d'entraînement EN DIRECT (un seul écran, 1920×1080, filmable).
// La page ne calcule rien : elle relit le journal d'événements que scripts/train.py écrit ligne par ligne
// (viz/entrainement/journal.jsonl) et dessine ce qu'il contient : poids et activations mesurés par couche, pertes,
// erreur de validation, prédictions d'un moteur de démonstration après chaque époque.
// Marche aussi sur un journal terminé (rejeu). ?journal=autre.jsonl change de fichier ; ?until=N n'utilise que les
// N premiers événements (pour revoir un instant précis).
(function () {
  "use strict";
  const W = 1920, H = 1080;
  const C = { fg: "#e8eef5", muted: "#8a98a8", bg: "#0f1419", grid: "#26303b", panel: "#1a222c",
    blue: "#4cc9f0", pink: "#f72585", yellow: "#ffd60a", green: "#80ed99", violet: "#b388ff", amber: "#ffb347", warn: "#ff6b81" };
  const FONT = "'Segoe UI', system-ui, sans-serif";
  const params = new URLSearchParams(location.search);
  const JOURNAL = params.get("journal") || "journal.jsonl";
  const UNTIL = params.has("until") ? +params.get("until") : Infinity;
  const num = (x, d = 2) => String(+x.toFixed(d)).replace(".", ",").replace("-", "−");
  const pct = (x) => `${Math.round(x * 100)} %`;
  const fmtT = (s) => `${Math.floor(s / 60)} min ${String(Math.floor(s % 60)).padStart(2, "0")} s`;

  const stageEl = document.getElementById("stage");
  document.getElementById("source").textContent = `journal : ${JOURNAL}`;
  const svg = d3.select("#stage").append("svg").attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`).style("font-family", FONT);
  svg.append("rect").attr("width", W).attr("height", H).attr("fill", C.bg);
  const gTitle = svg.append("g"), gCards = svg.append("g"), gNet = svg.append("g"), gLoss = svg.append("g"),
    gRmse = svg.append("g"), gPred = svg.append("g"), gMsg = svg.append("g");

  // ------------------------------------------------------------------------------------------------ état construit depuis le journal
  function build(events) {
    const st = { meta: null, probe: null, epochs: [], batches: [], layers: {}, kernels: {}, preds: {}, best: null, early: null,
      final: null, ended: false, checks: [], t: events.length ? events[events.length - 1].t : 0, n: events.length };
    for (const e of events) {
      if (e.type === "meta") st.meta = e; else if (e.type === "probe") st.probe = e; else if (e.type === "epoch") st.epochs.push(e);
      else if (e.type === "batch") st.batches.push(e); else if (e.type === "layers") { st.layers[e.epoch] = e.layers; st.kernels[e.epoch] = e.kernels; }
      else if (e.type === "prediction") st.preds[e.epoch] = e.pred; else if (e.type === "best") st.best = e;
      else if (e.type === "early_stop") st.early = e; else if (e.type === "final") st.final = e;
      else if (e.type === "check") st.checks.push(e); else if (e.type === "end") st.ended = true;
    }
    return st;
  }

  // ------------------------------------------------------------------------------------------------ outils de dessin
  const txt = (g, x, y, s, o = {}) => g.append("text").attr("x", x).attr("y", y).attr("fill", o.fill || C.fg)
    .attr("font-size", o.size || 26).attr("font-weight", o.bold ? 700 : 400).attr("text-anchor", o.anchor || "start").text(s);

  function chart(g, { x0, y0, w, h, ph = h + 130, xd, yd, title, series, hlines = [], yfmt = (v) => num(v, 2), xlab = "époque", marks = [] }) {
    const sx = d3.scaleLinear().domain(xd).range([x0, x0 + w]), sy = d3.scaleLinear().domain(yd).range([y0 + h, y0]).nice();
    g.append("rect").attr("x", x0 - 70).attr("y", y0 - 60).attr("width", w + 90).attr("height", ph).attr("rx", 12).attr("fill", C.panel);
    txt(g, x0, y0 - 22, title, { size: 28, bold: true });
    for (const v of sy.ticks(4)) {
      g.append("line").attr("x1", x0).attr("x2", x0 + w).attr("y1", sy(v)).attr("y2", sy(v)).attr("stroke", C.grid);
      txt(g, x0 - 10, sy(v) + 8, yfmt(v), { size: 22, fill: C.muted, anchor: "end" });
    }
    for (const v of sx.ticks(5)) txt(g, sx(v), y0 + h + 30, String(v), { size: 22, fill: C.muted, anchor: "middle" });
    txt(g, x0 + w / 2, y0 + h + 62, xlab, { size: 22, fill: C.muted, anchor: "middle" });
    for (const hl of hlines) {
      g.append("line").attr("x1", x0).attr("x2", x0 + w).attr("y1", sy(hl.y)).attr("y2", sy(hl.y)).attr("stroke", hl.color)
        .attr("stroke-width", 3).attr("stroke-dasharray", "10 8");
      txt(g, x0 + w - 8, sy(hl.y) - 10, hl.label, { size: 22, fill: hl.color, anchor: "end", bold: true });
    }
    for (const s of series) {
      if (!s.pts.length) continue;
      g.append("path").attr("fill", "none").attr("stroke", s.color).attr("stroke-width", s.width || 4).attr("stroke-opacity", s.opacity || 1)
        .attr("stroke-dasharray", s.dash || null).attr("d", d3.line().x((d) => sx(d[0])).y((d) => sy(d[1]))(s.pts));
      if (s.dots) g.selectAll(null).data(s.pts).join("circle").attr("cx", (d) => sx(d[0])).attr("cy", (d) => sy(d[1])).attr("r", 6).attr("fill", s.color);
    }
    for (const m of marks) g.append("circle").attr("cx", sx(m.x)).attr("cy", sy(m.y)).attr("r", 13).attr("fill", "none").attr("stroke", m.color).attr("stroke-width", 4);
    return { sx, sy };
  }

  // ------------------------------------------------------------------------------------------------ panneaux
  function drawTitle(st, live) {
    gTitle.selectAll("*").remove();
    txt(gTitle, 60, 80, "Le réseau apprend en direct", { size: 52, bold: true });
    const m = st.meta, ep = st.epochs.length ? st.epochs[st.epochs.length - 1].epoch : 0;
    const total = m ? (m.fixed_epochs || m.config.max_epochs) : "?";
    let status = !m ? "en attente du premier événement…" : st.ended ? "terminé" : `époque ${ep + (ep < total ? 1 : 0)} / ${total}`;
    const col = st.ended ? C.green : C.amber;
    gTitle.append("circle").attr("cx", 1470).attr("cy", 66).attr("r", 11).attr("fill", col);
    txt(gTitle, 1495, 78, status, { size: 36, bold: true, fill: col });
    if (m) txt(gTitle, 1860, 126, `${m.device} · ${m.threads} threads · graine ${m.config.seed} · réglage ${m.config.preset || "défaut"} · ${fmtT(st.t)}`, { size: 24, fill: C.muted, anchor: "end" });
    if (st.batches.length && !st.ended) {                                  // avancement dans l'époque, lu dans les événements « batch »
      const b = st.batches[st.batches.length - 1], inEpoch = b.epoch === ep + 1;
      if (inEpoch) {
        gTitle.append("rect").attr("x", 1495).attr("y", 96).attr("width", 365 * 0.6).attr("height", 10).attr("rx", 5).attr("fill", C.grid);
        gTitle.append("rect").attr("x", 1495).attr("y", 96).attr("width", 365 * 0.6 * (b.batch / b.n_batches)).attr("height", 10).attr("rx", 5).attr("fill", col);
      }
    }
  }

  function drawCards(st) {
    gCards.selectAll("*").remove();
    const last = st.epochs[st.epochs.length - 1], ref = st.meta && st.meta.reference;
    const refv = ref && (ref.baseline_val_rmse_fold ?? ref.baseline_val_rmse_mean);
    const hasVal = last && last.rmse !== undefined;
    const cards = [
      { t: "Erreur typique (RMSE)", v: hasVal ? `${num(last.rmse, 1)} cycles` : "—", s: refv && hasVal ? `baseline : ${num(refv, 1)}` : (refv ? `baseline : ${num(refv, 1)}` : ""), c: hasVal && refv && last.rmse < refv ? C.green : C.fg },
      { t: "Score PHM08 (validation)", v: hasVal ? (last.score >= 1e6 ? num(last.score / 1e6, 1) + " M" : num(last.score / 1e3, 0) + " k") : "—", s: "retard puni plus fort", c: C.fg },
      { t: "Régime reconnu", v: hasVal ? pct(last.regime_acc) : "—", s: "proxy par jeu", c: C.fg },
      { t: "Durée d'une époque", v: last ? `${num(last.seconds, 0)} s` : "—", s: st.best ? `meilleure époque : ${st.best.epoch}` : "", c: C.fg },
    ];
    cards.forEach((c, i) => {
      const x = 60 + i * 465;
      gCards.append("rect").attr("x", x).attr("y", 140).attr("width", 440).attr("height", 112).attr("rx", 12).attr("fill", C.panel);
      txt(gCards, x + 20, 174, c.t, { size: 24, fill: C.muted });
      txt(gCards, x + 20, 224, c.v, { size: 44, bold: true, fill: c.c });
      txt(gCards, x + 420, 224, c.s, { size: 22, fill: C.muted, anchor: "end" });
    });
  }

  function drawNet(st) {
    gNet.selectAll("*").remove();
    const m = st.meta;
    gNet.append("rect").attr("x", 20).attr("y", 270).attr("width", 770).attr("height", 770).attr("rx", 12).attr("fill", C.panel);
    txt(gNet, 40, 308, "Le réseau (structure fixe, poids qui évoluent)", { size: 28, bold: true });
    if (!m) return;
    const eps = Object.keys(st.layers).map(Number).sort((a, b) => a - b), cur = st.layers[eps[eps.length - 1]], l0 = st.layers[0];
    const byName = Object.fromEntries(m.layers.map((l) => [l.name, l]));
    const boxes = [
      { id: "input", x: 40, y: 340, w: 400, h: 70, t: "Entrée", s: `${m.model.window} cycles × ${m.model.n_features} mesures` },
      { id: "conv1", x: 40, y: 440, w: 400, h: 130, t: "Convolution 1", s: "repère des motifs courts" },
      { id: "conv2", x: 40, y: 600, w: 400, h: 130, t: "Convolution 2", s: "combine les motifs" },
      { id: "gru", x: 40, y: 760, w: 400, h: 135, t: "Mémoire (GRU)", s: "suit l'évolution dans le temps" },
      { id: "head_rul", x: 40, y: 925, w: 195, h: 105, t: "Durée restante", s: "" },
      { id: "head_regime", x: 245, y: 925, w: 195, h: 105, t: "Régime", s: "" },
    ];
    const link = (a, b) => gNet.append("line").attr("x1", a.x + a.w / 2).attr("x2", b.x + b.w / 2).attr("y1", a.y + a.h).attr("y2", b.y).attr("stroke", C.muted).attr("stroke-width", 4);
    link(boxes[0], boxes[1]); link(boxes[1], boxes[2]); link(boxes[2], boxes[3]);
    gNet.append("line").attr("x1", 240).attr("x2", 140).attr("y1", 895).attr("y2", 925).attr("stroke", C.muted).attr("stroke-width", 4);
    gNet.append("line").attr("x1", 240).attr("x2", 345).attr("y1", 895).attr("y2", 925).attr("stroke", C.muted).attr("stroke-width", 4);
    for (const b of boxes) {
      const ls = cur && cur[b.id], l0b = l0 && l0[b.id];
      let move = 0;
      if (ls && l0b && l0b.weight_norm > 0) move = Math.min(1, Math.abs(ls.weight_norm - l0b.weight_norm) / l0b.weight_norm);
      gNet.append("rect").attr("x", b.x).attr("y", b.y).attr("width", b.w).attr("height", b.h).attr("rx", 10)
        .attr("fill", C.bg).attr("stroke", ls ? d3.interpolateRgb("#34465a", C.blue)(move) : "#34465a").attr("stroke-width", 4 + 6 * move);
      txt(gNet, b.x + 16, b.y + 34, b.t, { size: b.w < 300 ? 24 : 28, bold: true });
      if (b.s) txt(gNet, b.x + 16, b.y + 62, b.s, { size: 22, fill: C.muted });
      if (b.id !== "input") {
        const p = byName[b.id].params;
        if (b.w > 300 && b.id !== "input") {
          txt(gNet, b.x + 16, b.y + b.h - 38, `${p.toLocaleString("fr-FR")} paramètres`, { size: 22, fill: C.muted });
          if (ls) txt(gNet, b.x + 16, b.y + b.h - 12, `poids ${num(ls.weight_norm, 2)} (départ ${num(l0b ? l0b.weight_norm : ls.weight_norm, 2)})` +
            (ls.dead_fraction !== undefined ? ` · éteints ${pct(ls.dead_fraction)}` : ""), { size: 21, fill: C.fg });
        } else if (ls) {
          txt(gNet, b.x + 16, b.y + 62, `poids ${num(ls.weight_norm, 2)}`, { size: 22, fill: C.fg });
          txt(gNet, b.x + 16, b.y + 88, `(départ ${num(l0b ? l0b.weight_norm : ls.weight_norm, 2)})`, { size: 20, fill: C.muted });
        }
      }
    }
    // noyaux de la 1ʳᵉ convolution (poids réels), à droite de la couche
    const k = cur && st.kernels[eps[eps.length - 1]];
    if (k) {
      const x0 = 470, y0 = 360, cw = 3.1, rh = 6;
      const flat = k.map((f) => f.flatMap((row) => row));                       // filtre × (mesure × position)
      const m0 = d3.max(Object.values(st.kernels), (kk) => d3.max(kk, (f) => d3.max(f, (r) => d3.max(r, Math.abs)))) || 1;
      const col = d3.scaleSequential(d3.interpolateRdBu).domain([m0, -m0]);
      gNet.selectAll(null).data(flat.flatMap((f, i) => f.map((v, j) => [i, j, v]))).join("rect")
        .attr("x", (d) => x0 + d[1] * cw).attr("y", (d) => y0 + d[0] * rh).attr("width", cw + 0.3).attr("height", rh + 0.3).attr("fill", (d) => col(d[2]));
      txt(gNet, x0, y0 - 14, "poids de la convolution 1", { size: 22, fill: C.muted });
      txt(gNet, x0, y0 + flat.length * rh + 28, `${flat.length} filtres × ${m.model.n_features} mesures × 5 cycles`, { size: 20, fill: C.muted });
    }
    txt(gNet, 470, 700, "Contour d'une couche :", { size: 22, fill: C.muted });
    txt(gNet, 470, 728, "plus il est vif, plus ses poids", { size: 22, fill: C.muted });
    txt(gNet, 470, 756, "ont bougé depuis le départ.", { size: 22, fill: C.muted });
  }

  function drawCharts(st) {
    gLoss.selectAll("*").remove(); gRmse.selectAll("*").remove();
    const m = st.meta, total = m ? (m.fixed_epochs || m.config.max_epochs) : 10, E = st.epochs;
    const xd = [0, Math.max(5, Math.min(total, Math.ceil((E.length + 1) / 5) * 5))];
    const hasVal = E.length && E[0].val_mse !== undefined;
    const batchPts = st.batches.map((b) => [b.epoch - 1 + b.batch / b.n_batches, b.mse]);
    const trPts = E.map((e) => [e.epoch, e.train_mse]), vaPts = hasVal ? E.map((e) => [e.epoch, e.val_mse]) : [];
    const ymax = d3.max([...batchPts, ...trPts, ...vaPts], (d) => d[1]) || 0.05;
    chart(gLoss, { x0: 880, y0: 340, w: 410, h: 210, ph: 400, xd, yd: [0, ymax], title: "Perte sur la durée restante", yfmt: (v) => num(v, 3),
      series: [{ pts: batchPts, color: C.blue, width: 2, opacity: 0.35 }, { pts: trPts, color: C.blue, dots: true }, { pts: vaPts, color: C.pink, dots: true }] });
    txt(gLoss, 855, 660, "entraînement", { size: 22, fill: C.blue, bold: true });
    txt(gLoss, 1015, 660, "validation", { size: 22, fill: C.pink, bold: true });
    txt(gLoss, 1145, 660, "(carré de l'erreur)", { size: 19, fill: C.muted });

    const rm = hasVal ? E.map((e) => [e.epoch, e.rmse]) : [];
    const ref = m && m.reference, refv = ref && (ref.baseline_val_rmse_fold ?? ref.baseline_val_rmse_mean);
    const yv = [...rm.map((d) => d[1]), ...(refv ? [refv] : [])];
    const marks = st.best && rm.length ? [{ x: st.best.epoch, y: st.best.rmse, color: C.green }] : [];
    chart(gRmse, { x0: 1440, y0: 340, w: 380, h: 210, ph: 400, xd, yd: [Math.max(0, (d3.min(yv) || 10) - 6), (d3.max(yv) || 50) + 4], title: "Erreur de validation (cycles)",
      yfmt: (v) => num(v, 0), series: [{ pts: rm, color: C.green, dots: true }], marks,
      hlines: refv ? [{ y: refv, label: `baseline ${num(refv, 1)}`, color: C.yellow }] : [] });
    txt(gRmse, 1440, 660, st.early ? `arrêt à l'époque ${st.early.epoch} (meilleure : ${st.early.best_epoch})` : "plus bas = mieux", { size: 22, fill: C.muted });
  }

  function drawPred(st) {
    gPred.selectAll("*").remove();
    const p = st.probe, eps = Object.keys(st.preds).map(Number).sort((a, b) => a - b);
    if (!p || !eps.length) return;
    const cur = eps[eps.length - 1], x0 = 880, y0 = 790, w = 940, h = 170;
    const all = [...p.true, ...p.true_capped, ...eps.flatMap((e) => st.preds[e])];
    const pts = (arr) => arr.map((v, i) => [p.end_cycle[i], v]);
    const series = [{ pts: pts(p.true), color: "#c8d0da", width: 5, dash: "3 10" }, { pts: pts(p.true_capped), color: "#c8d0da", width: 4, opacity: 0.45 }];
    const ghosts = eps.filter((e) => e !== cur && e !== 0).slice(-3);
    ghosts.forEach((e, i) => series.push({ pts: pts(st.preds[e]), color: C.green, width: 3, opacity: 0.18 + 0.12 * i }));
    if (cur !== 0) series.push({ pts: pts(st.preds[0]), color: C.amber, width: 3, opacity: 0.8, dash: "8 8" });
    series.push({ pts: pts(st.preds[cur]), color: C.green, width: 5 });
    chart(gPred, { x0, y0, w, h, xd: d3.extent(p.end_cycle), yd: [0, d3.max(all) * 1.05], ph: 340, title: `Moteur ${p.engine} : durée restante réelle et prédite — époque ${cur}`,
      xlab: "vie du moteur (cycles)", yfmt: (v) => num(v, 0), series });
    const lg = [["réelle", "#c8d0da"], ["réelle plafonnée", "#9aa6b4"], ["prédite", C.green], ["au départ", C.amber]];
    lg.forEach(([t, c], i) => { gPred.append("rect").attr("x", 960 + i * 220).attr("y", 1052).attr("width", 26).attr("height", 8).attr("fill", c); txt(gPred, 994 + i * 220, 1062, t, { size: 21, fill: C.muted }); });
  }

  function message(s) { gMsg.selectAll("*").remove(); if (s) txt(gMsg, W / 2, 620, s, { size: 44, fill: C.warn, anchor: "middle" }); }

  function draw(st, live) {
    drawTitle(st, live); drawCards(st); drawNet(st); drawCharts(st); drawPred(st);
    const bad = st.checks.filter((c) => !c.ok);
    message(bad.length ? `Contrôle du journal en échec : ${bad.map((c) => c.name).join(", ")}` : (!st.meta ? "En attente du journal d'entraînement…" : ""));
    document.getElementById("state").textContent = st.ended ? `terminé · ${st.n} événements` : `en direct · ${st.n} événements`;
  }

  // ------------------------------------------------------------------------------------------------ lecture du journal
  let ended = false, recorder = null;
  async function poll() {
    try {
      const res = await fetch(JOURNAL, { cache: "no-store" });
      if (res.ok) {
        let text = await res.text();
        if (!text.endsWith("\n")) text = text.slice(0, text.lastIndexOf("\n") + 1);      // dernière ligne encore en cours d'écriture
        const events = text.split("\n").filter(Boolean).slice(0, UNTIL).map((l) => JSON.parse(l));
        events.forEach((e, i) => { if (e.seq !== i) throw new Error(`journal non contigu à l'événement n°${i}`); });
        const st = build(events);
        window.STATE = st;
        draw(st, true);
        if (st.ended && !ended) { ended = true; if (recorder) recorder.stopSoon(3000); }
        if (st.ended || events.length >= UNTIL) { window.READY = true; return; }       // journal fini : on cesse de relire
      } else {
        message(`Journal introuvable (HTTP ${res.status}) : lancer scripts/train.py`);
      }
    } catch (err) {
      message(String(err.message)); window.ERROR = String(err.message);
    }
    window.READY = true;
    setTimeout(poll, 1000);
  }

  function fit() {
    const rec = document.body.classList.contains("recording");
    const reserve = rec || params.get("render") === "1" ? 0 : 64;
    const s = Math.min(innerWidth / W, (innerHeight - reserve) / H);
    stageEl.style.transform = `translate(${(innerWidth - W * s) / 2}px,${(innerHeight - reserve - H * s) / 2}px) scale(${s})`;
  }
  if (params.get("render") === "1") document.body.classList.add("render");
  addEventListener("resize", fit);
  fit();
  recorder = VizRecord.attach({ button: document.getElementById("rec"), countdownEl: document.getElementById("countdown"),
    msgEl: document.getElementById("msg"), name: "entrainement", onLayout: fit });
  poll();
})();
