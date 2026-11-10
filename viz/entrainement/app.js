// Vidéo grand public de l'entraînement du réseau (4 scènes, une phrase par scène). Rejoue le journal écrit par
// scripts/train.py : architecture et nombres de paramètres réels, pertes réelles époque après époque, prédictions réelles du
// réseau sur un moteur de validation après chaque époque, erreur de validation réelle contre la baseline.
// Le cœur commun est dans viz/lib/core.js ; la version en direct est dans live.html.
VizCore.start({
  journal: "journal.jsonl",
  hint: "Lancer scripts/train.py --smoke (court essai) ou --cv --journal",
  noteFinale: "Illustration fidèle des vrais calculs du programme.",
  scenes(ctx) {
    const { d3, W, COLORS, FG, MUTED, GRID, clamp01, seg, num, int, by, one, rotLabel, baseline } = ctx;
    const meta = one("meta"), probe = one("probe");
    const epochs = by("epoch"), preds = by("prediction"), bests = by("best"), early = one("early_stop");
    const ref = meta.reference || {};
    const refv = ref.baseline_val_rmse_fold ?? ref.baseline_val_rmse_mean;
    const GREEN = COLORS[3], BLUE = COLORS[0], PINK = COLORS[1], YELLOW = COLORS[2], AMBER = COLORS[5];
    const L = Object.fromEntries(meta.layers.map((l) => [l.name, l]));
    // Les légendes suivent ce que le journal montre vraiment (sinon elles pourraient contredire les courbes).
    const valMse = epochs.filter((e) => e.val_mse !== undefined).map((e) => e.val_mse);
    const valRises = valMse.length > 2 && valMse[valMse.length - 1] > Math.min(...valMse) * 1.05;
    const lastBest = bests.length ? bests[bests.length - 1] : null;
    const beatsBaseline = lastBest && refv ? lastBest.rmse < refv : null;

    // ---------------------------------------------------------------- scène 1 : le réseau
    function sceneNet(g) {
      const boxes = [
        { id: "input", x: 90, y: 400, w: 300, h: 190, t: "Les mesures", s: `${meta.model.window} cycles` },
        { id: "conv1", x: 470, y: 400, w: 300, h: 190, t: "Repérer", s: "des motifs courts" },
        { id: "conv2", x: 850, y: 400, w: 300, h: 190, t: "Combiner", s: "les motifs" },
        { id: "gru", x: 1230, y: 400, w: 300, h: 190, t: "Se souvenir", s: "de l'évolution" },
        { id: "head_rul", x: 1590, y: 270, w: 310, h: 190, t: "Durée restante", s: "avant la panne" },
        { id: "head_regime", x: 1590, y: 530, w: 310, h: 190, t: "Régime de panne", s: "probable" },
      ];
      const links = [[0, 1], [1, 2], [2, 3], [3, 4], [3, 5]];
      const ln = g.selectAll("line.l").data(links).join("line").attr("stroke", MUTED).attr("stroke-width", 5)
        .attr("x1", (d) => boxes[d[0]].x + boxes[d[0]].w).attr("y1", (d) => boxes[d[0]].y + boxes[d[0]].h / 2)
        .attr("x2", (d) => boxes[d[1]].x).attr("y2", (d) => boxes[d[1]].y + boxes[d[1]].h / 2);
      const bx = g.selectAll("g.b").data(boxes).join("g");
      bx.append("rect").attr("x", (b) => b.x).attr("y", (b) => b.y).attr("width", (b) => b.w).attr("height", (b) => b.h).attr("rx", 16)
        .attr("fill", "#1a222c").attr("stroke", (b, i) => (i < 4 ? BLUE : GREEN)).attr("stroke-width", 6);
      bx.append("text").attr("x", (b) => b.x + b.w / 2).attr("y", (b) => b.y + 72).attr("text-anchor", "middle").attr("fill", FG)
        .attr("font-size", (b) => (b.w > 300 ? 36 : 40)).attr("font-weight", 700).text((b) => b.t);
      bx.append("text").attr("x", (b) => b.x + b.w / 2).attr("y", (b) => b.y + 118).attr("text-anchor", "middle").attr("fill", MUTED)
        .attr("font-size", 30).text((b) => b.s);
      bx.append("text").attr("x", (b) => b.x + b.w / 2).attr("y", (b) => b.y + b.h - 22).attr("text-anchor", "middle").attr("fill", MUTED)
        .attr("font-size", 26).text((b) => (b.id === "input" ? `${meta.model.n_features} mesures par cycle` : `${int(L[b.id].params)} réglages`));
      const total = g.append("text").attr("x", W / 2).attr("y", 840).attr("text-anchor", "middle").attr("fill", FG).attr("font-size", 44).attr("font-weight", 700)
        .text(`${int(meta.n_params)} réglages à ajuster pendant l'apprentissage`);
      return {
        dur: 11, simple: true, refs: [meta],
        title: "Le réseau de neurones", subtitle: "",
        captions: [[0.6, 10.2, "Le réseau lit 30 cycles de mesures et en tire une durée restante."]],
        update(u) {
          bx.attr("opacity", (b, i) => seg(u, 0.5 + i * 0.7, 1.2 + i * 0.7));
          ln.attr("opacity", (d) => seg(u, 0.9 + d[1] * 0.7, 1.5 + d[1] * 0.7));
          total.attr("opacity", seg(u, 5.2, 6.0));
        },
      };
    }

    // ---------------------------------------------------------------- scène 2 : il apprend (pertes réelles)
    function sceneLoss() {
      return (g) => {
        const P = { x0: 300, x1: 1620, y0: 260, y1: 790 };
        const hasVal = epochs[0].val_mse !== undefined;
        const nE = epochs.length;
        const sx = d3.scaleLinear().domain([0, nE]).range([P.x0, P.x1]);
        const ymax = d3.max(epochs, (e) => Math.max(e.train_mse, hasVal ? e.val_mse : 0));
        const sy = d3.scaleLinear().domain([0, ymax * 1.08]).range([P.y1, P.y0]);
        baseline(g, P.x0, P.x1, P.y1);
        g.append("line").attr("x1", P.x0).attr("x2", P.x0).attr("y1", P.y0).attr("y2", P.y1).attr("stroke", MUTED).attr("stroke-width", 3);
        g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", P.y1 + 66).attr("text-anchor", "middle").attr("fill", MUTED).attr("font-size", 32).text("passages sur toutes les données  →");
        rotLabel(g, P.x0 - 50, (P.y0 + P.y1) / 2, "erreur (plus bas = mieux)").attr("font-size", 32);
        const tr = g.append("path").attr("fill", "none").attr("stroke", BLUE).attr("stroke-width", 8).attr("stroke-linecap", "round");
        const va = g.append("path").attr("fill", "none").attr("stroke", PINK).attr("stroke-width", 8).attr("stroke-linecap", "round");
        const dtr = g.selectAll("circle.t").data(epochs).join("circle").attr("class", "t").attr("r", 9).attr("fill", BLUE).attr("cx", (e) => sx(e.epoch)).attr("cy", (e) => sy(e.train_mse));
        const dva = g.selectAll("circle.v").data(hasVal ? epochs : []).join("circle").attr("class", "v").attr("r", 9).attr("fill", PINK).attr("cx", (e) => sx(e.epoch)).attr("cy", (e) => sy(e.val_mse));
        const lg = g.append("g").attr("transform", `translate(${P.x1 - 560},${P.y0 + 10})`);
        lg.append("circle").attr("r", 10).attr("fill", BLUE); lg.append("text").attr("x", 24).attr("y", 11).attr("fill", FG).attr("font-size", 32).text("sur les données d'entraînement");
        lg.append("circle").attr("cy", 52).attr("r", 10).attr("fill", PINK); lg.append("text").attr("x", 24).attr("y", 63).attr("fill", FG).attr("font-size", 32).text("sur des données jamais vues");
        const lab = g.append("text").attr("x", P.x0 + 30).attr("y", P.y0 + 30).attr("fill", FG).attr("font-size", 40).attr("font-weight", 700);
        return {
          dur: 12, simple: true, refs: [...epochs],
          title: "Il apprend", subtitle: "",
          captions: [[0.6, 11.2, valRises
            ? "À l'entraînement l'erreur baisse ; sur des données jamais vues, elle remonte."
            : "À chaque passage sur les données, son erreur baisse."]],
          update(u) {
            const k = seg(u, 0.8, 9.0, d3.easeLinear) * nE;                     // nombre d'époques déjà montrées
            const shown = epochs.filter((e) => e.epoch <= Math.floor(k + 1e-9));
            const line = (key) => d3.line().x((e) => sx(e.epoch)).y((e) => sy(e[key]))(shown);
            tr.attr("d", shown.length ? line("train_mse") : null);
            va.attr("d", hasVal && shown.length ? line("val_mse") : null);
            dtr.attr("opacity", (e) => (e.epoch <= Math.floor(k + 1e-9) ? 1 : 0)); dva.attr("opacity", (e) => (e.epoch <= Math.floor(k + 1e-9) ? 1 : 0));
            lab.text(shown.length ? `passage n°${shown.length}` : "");
          },
        };
      };
    }

    // ---------------------------------------------------------------- scène 3 : les prédictions se précisent
    function scenePred(g) {
      const P = { x0: 300, x1: 1620, y0: 260, y1: 790 };
      const cyc = probe.end_cycle;
      const sx = d3.scaleLinear().domain(d3.extent(cyc)).range([P.x0 + 10, P.x1 - 10]);
      const sy = d3.scaleLinear().domain([0, d3.max([...probe.true, ...preds.flatMap((p) => p.pred)]) * 1.05]).range([P.y1, P.y0]);
      baseline(g, P.x0, P.x1, P.y1 + 16);
      g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", P.y1 + 76).attr("text-anchor", "middle").attr("fill", MUTED).attr("font-size", 32).text("vie du moteur  →");
      rotLabel(g, P.x0 - 50, (P.y0 + P.y1) / 2, "durée restante (cycles)").attr("font-size", 32);
      const line = (arr) => d3.line().x((d, i) => sx(cyc[i])).y((d) => sy(d))(arr);
      g.append("path").attr("d", line(probe.true)).attr("fill", "none").attr("stroke", "#c8d0da").attr("stroke-width", 6).attr("stroke-dasharray", "4 12").attr("stroke-linecap", "round");
      const cur = g.append("path").attr("fill", "none").attr("stroke", GREEN).attr("stroke-width", 8).attr("stroke-linejoin", "round");
      const prev = g.append("path").attr("fill", "none").attr("stroke", GREEN).attr("stroke-width", 6).attr("stroke-linejoin", "round");
      const lab = g.append("text").attr("x", P.x1).attr("y", P.y0 - 6).attr("text-anchor", "end").attr("fill", FG).attr("font-size", 40).attr("font-weight", 700);
      const lg = g.append("g").attr("transform", `translate(${P.x0},${P.y0 - 30})`);
      lg.append("line").attr("x2", 56).attr("stroke", "#c8d0da").attr("stroke-width", 6).attr("stroke-dasharray", "4 12");
      lg.append("text").attr("x", 72).attr("y", 10).attr("fill", FG).attr("font-size", 30).text("la vraie durée restante");
      lg.append("line").attr("x1", 440).attr("x2", 496).attr("stroke", GREEN).attr("stroke-width", 8);
      lg.append("text").attr("x", 512).attr("y", 10).attr("fill", FG).attr("font-size", 30).text("ce que prédit le réseau");
      const n = preds.length;
      return {
        dur: 12, simple: true, refs: [probe, ...preds],
        title: "Ses prédictions se précisent", subtitle: "",
        captions: [[0.6, 11.2, "Ses prédictions changent à chaque passage et se rapprochent de la réalité."]],
        update(u) {
          const k = Math.min(n - 1, Math.floor(seg(u, 0.8, 10.0, d3.easeLinear) * n));   // état réel affiché : celui de l'époque k
          const f = seg(u, 0.8, 10.0, d3.easeLinear) * n - Math.floor(seg(u, 0.8, 10.0, d3.easeLinear) * n);
          const fade = clamp01(f / 0.25);                                          // fondu entre deux états réels (aucune valeur interpolée)
          cur.attr("d", line(preds[k].pred)).attr("opacity", k === 0 ? 1 : fade);
          prev.attr("d", k > 0 ? line(preds[k - 1].pred) : null).attr("opacity", k > 0 ? 1 - fade : 0);
          lab.text(preds[k].epoch === 0 ? "avant d'apprendre" : `après ${preds[k].epoch} passage${preds[k].epoch > 1 ? "s" : ""}`);
        },
      };
    }

    // ---------------------------------------------------------------- scène 4 : il s'arrête au bon moment, contre la baseline
    function sceneEnd(g) {
      const P = { x0: 300, x1: 1620, y0: 260, y1: 790 };
      const rm = epochs.filter((e) => e.rmse !== undefined);
      const best = bests.length ? bests[bests.length - 1] : null;
      const sx = d3.scaleLinear().domain([0, rm.length]).range([P.x0, P.x1]);
      const yv = [...rm.map((e) => e.rmse), ...(refv ? [refv] : [])];
      const sy = d3.scaleLinear().domain([Math.max(0, d3.min(yv) - 6), d3.max(yv) + 3]).range([P.y1, P.y0]);
      baseline(g, P.x0, P.x1, P.y1);
      g.append("line").attr("x1", P.x0).attr("x2", P.x0).attr("y1", P.y0).attr("y2", P.y1).attr("stroke", MUTED).attr("stroke-width", 3);
      g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", P.y1 + 66).attr("text-anchor", "middle").attr("fill", MUTED).attr("font-size", 32).text("passages sur toutes les données  →");
      rotLabel(g, P.x0 - 100, (P.y0 + P.y1) / 2, "erreur typique (cycles)").attr("font-size", 32);
      for (const v of sy.ticks(4)) {
        g.append("line").attr("x1", P.x0).attr("x2", P.x1).attr("y1", sy(v)).attr("y2", sy(v)).attr("stroke", GRID);
        g.append("text").attr("x", P.x0 - 14).attr("y", sy(v) + 9).attr("text-anchor", "end").attr("fill", MUTED).attr("font-size", 26).text(num(v, 0));
      }
      const refG = g.append("g");
      if (refv) {
        refG.append("line").attr("x1", P.x0).attr("x2", P.x1).attr("y1", sy(refv)).attr("y2", sy(refv)).attr("stroke", YELLOW).attr("stroke-width", 5).attr("stroke-dasharray", "14 10");
        refG.append("text").attr("x", P.x1 - 30).attr("y", sy(refv) - 14).attr("text-anchor", "end").attr("fill", YELLOW).attr("font-size", 32).attr("font-weight", 700)
          .text(`score de la baseline : ${num(refv, 1)}`);
      }
      const path = g.append("path").attr("fill", "none").attr("stroke", GREEN).attr("stroke-width", 8).attr("stroke-linecap", "round");
      const dots = g.selectAll("circle.d").data(rm).join("circle").attr("class", "d").attr("r", 9).attr("fill", GREEN).attr("cx", (e, i) => sx(i + 1)).attr("cy", (e) => sy(e.rmse));
      const ring = g.append("circle").attr("r", 20).attr("fill", "none").attr("stroke", FG).attr("stroke-width", 6);
      const bestTxt = g.append("text").attr("text-anchor", "middle").attr("fill", FG).attr("font-size", 34).attr("font-weight", 700);
      const bi = best ? rm.findIndex((e) => e.epoch === best.epoch) : -1;
      const si = early ? rm.findIndex((e) => e.epoch === early.epoch) : -1;       // époque où l'arrêt précoce a eu lieu (événement réel)
      const stopG = g.append("g");
      if (si >= 0) {
        stopG.append("line").attr("x1", sx(si + 1)).attr("x2", sx(si + 1)).attr("y1", P.y0).attr("y2", P.y1).attr("stroke", COLORS[1]).attr("stroke-width", 4).attr("stroke-dasharray", "10 10");
        stopG.append("text").attr("x", sx(si + 1) - 14).attr("y", P.y0 + 34).attr("text-anchor", "end").attr("fill", COLORS[1]).attr("font-size", 32).attr("font-weight", 700).text("arrêt");
      }
      return {
        dur: 12, simple: true, refs: [...epochs, ...bests, ...(early ? [early] : [])],
        title: "Il s'arrête au bon moment", subtitle: "",
        captions: [[0.6, 11.2, (early ? "Il s'arrête à temps" : "On garde son meilleur moment")
          + (beatsBaseline === null ? "." : beatsBaseline ? " et fait mieux que la baseline." : ", mais n'est pas encore mieux que la baseline.")]],
        update(u) {
          const k = seg(u, 0.8, 8.0, d3.easeLinear) * rm.length;
          const shown = rm.filter((e, i) => i + 1 <= Math.floor(k + 1e-9));
          path.attr("d", shown.length ? d3.line().x((e, i) => sx(i + 1)).y((e) => sy(e.rmse))(shown) : null);
          dots.attr("opacity", (e, i) => (i + 1 <= Math.floor(k + 1e-9) ? 1 : 0));
          const a = seg(u, 8.4, 9.2);
          stopG.attr("opacity", a);
          if (bi >= 0) {
            ring.attr("cx", sx(bi + 1)).attr("cy", sy(rm[bi].rmse)).attr("opacity", a);
            bestTxt.attr("x", sx(bi + 1)).attr("y", sy(rm[bi].rmse) + 60).attr("opacity", a)
              .text(`meilleur : ${num(rm[bi].rmse, 1)}`);
          }
        },
      };
    }

    const all = [sceneNet, sceneLoss(), scenePred, sceneEnd];
    return { simple: all, tech: all };
  },
});
