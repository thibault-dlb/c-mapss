// Scènes de la vidéo « métriques » (RMSE et score de PHM08). Le journal est écrit par scripts/baseline.py --journal :
// prédictions réelles de la baseline sur des moteurs de validation, courbe de pénalité calculée par la vraie fonction
// `phm_penalty`, résultats réels du test. Le cœur commun est dans viz/lib/core.js.
VizCore.start({
  journal: "journal.jsonl",
  hint: "Lancer scripts/baseline.py --journal",
  noteFinale: "Illustration fidèle des vrais calculs du programme.",
  scenes(ctx) {
    const { d3, W, COLORS, FG, MUTED, GRID, clamp01, seg, lerp, num, int, one, rotLabel, baseline } = ctx;

    const meta = one("meta"), curve = one("penalty_curve"), eng = one("engines"), test = one("test");
    const E = eng.engines;
    const EARLY = COLORS[0], LATE = COLORS[1], TRUE = "#c8d0da";
    const colorOf = (x) => (x.error < 0 ? EARLY : LATE);
    const fmtErr = (e) => `${e > 0 ? "+" : e < 0 ? "−" : ""}${Math.abs(Math.round(e))}`;

    // ---------------------------------------------------------------- scène 1 : une erreur par moteur
    function sceneErrors(g) {
      const P = { x0: 300, x1: 1560, y0: 215, y1: 860 };
      const sx = d3.scaleLinear().domain([0, 200]).range([P.x0, P.x1]);
      const rowH = (P.y1 - P.y0) / E.length;
      baseline(g, P.x0, P.x1, P.y1 + 16);
      g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", P.y1 + 70).attr("fill", MUTED).attr("font-size", 32)
        .attr("text-anchor", "middle").text("durée restante avant la panne (cycles)  →");
      [0, 50, 100, 150, 200].forEach((v) => g.append("text").attr("x", sx(v)).attr("y", P.y1 + 44).attr("fill", MUTED)
        .attr("font-size", 24).attr("text-anchor", "middle").text(v));
      const rows = g.selectAll("g.row").data(E).join("g").attr("class", "row")
        .attr("transform", (x, i) => `translate(0,${P.y0 + i * rowH + rowH / 2})`);
      rows.append("line").attr("x1", (x) => sx(Math.min(x.true, x.pred))).attr("x2", (x) => sx(Math.max(x.true, x.pred)))
        .attr("stroke", colorOf).attr("stroke-width", 6).attr("stroke-linecap", "round");
      rows.append("circle").attr("cx", (x) => sx(x.true)).attr("r", 12).attr("fill", TRUE);
      rows.append("circle").attr("cx", (x) => sx(x.pred)).attr("r", 12).attr("fill", colorOf);
      rows.append("text").attr("x", (x) => sx(Math.max(x.true, x.pred)) + 24).attr("y", 9).attr("font-size", 28)
        .attr("font-weight", 700).attr("fill", colorOf).text((x) => fmtErr(x.error));
      const lg = g.append("g").attr("transform", `translate(${P.x0},150)`);
      [[TRUE, "vraie durée restante"], [EARLY, "prédiction trop tôt"], [LATE, "prédiction trop tard"]].forEach(([c, t], i) => {
        lg.append("circle").attr("cx", 12 + i * 400).attr("cy", -9).attr("r", 12).attr("fill", c);
        lg.append("text").attr("x", 36 + i * 400).attr("fill", FG).attr("font-size", 28).text(t);
      });
      return {
        dur: 11, simple: true, refs: [eng],
        title: "Mesurer l'erreur d'une prédiction", subtitle: "",
        captions: [[0.6, 10.4, "Pour chaque moteur : de combien de cycles se trompe-t-on ?"]],
        update(u) {
          rows.attr("opacity", (x, i) => seg(u, 1.0 + i * 0.4, 1.5 + i * 0.4));
        },
      };
    }

    // ---------------------------------------------------------------- scène 2 : trop tôt n'est pas trop tard
    function sceneCurve(g) {
      const P = { x0: 300, x1: 1620, y0: 230, y1: 810 };
      const d = curve.d, pen = curve.penalty;
      const sx = d3.scaleLinear().domain([d[0], d.at(-1)]).range([P.x0, P.x1]);
      const sy = d3.scaleLinear().domain([0, d3.max(pen)]).range([P.y1, P.y0]);
      baseline(g, P.x0, P.x1, P.y1);
      g.append("line").attr("x1", sx(0)).attr("x2", sx(0)).attr("y1", P.y0 - 10).attr("y2", P.y1).attr("stroke", GRID).attr("stroke-width", 3);
      g.append("text").attr("x", (P.x0 + sx(0)) / 2).attr("y", P.y1 + 62).attr("fill", EARLY).attr("font-size", 34)
        .attr("font-weight", 700).attr("text-anchor", "middle").text("← prédit trop tôt");
      g.append("text").attr("x", (sx(0) + P.x1) / 2).attr("y", P.y1 + 62).attr("fill", LATE).attr("font-size", 34)
        .attr("font-weight", 700).attr("text-anchor", "middle").text("prédit trop tard →");
      g.append("text").attr("x", sx(0)).attr("y", P.y1 + 62).attr("fill", MUTED).attr("font-size", 30)
        .attr("text-anchor", "middle").text("juste");
      rotLabel(g, P.x0 - 50, (P.y0 + P.y1) / 2, "pénalité").attr("font-size", 32);
      const pts = d.map((v, i) => [sx(v), sy(pen[i])]);
      const neg = pts.filter((p, i) => d[i] <= 0), pos = pts.filter((p, i) => d[i] >= 0);
      const line = d3.line();
      const pathE = g.append("path").attr("fill", "none").attr("stroke", EARLY).attr("stroke-width", 8).attr("stroke-linecap", "round");
      const pathL = g.append("path").attr("fill", "none").attr("stroke", LATE).attr("stroke-width", 8).attr("stroke-linecap", "round");
      // deux repères : la même erreur de 20 cycles, d'un côté et de l'autre (valeurs lues dans le journal)
      const at = (v) => pen[d.indexOf(v)];
      const mark = (v, c, text) => {
        const x = sx(v), y = sy(at(v));
        g.append("line").attr("x1", x).attr("x2", x).attr("y1", y).attr("y2", P.y1).attr("stroke", c).attr("stroke-width", 3).attr("stroke-dasharray", "8 8").attr("opacity", 0.8);
        const dot = g.append("circle").attr("cx", x).attr("cy", y).attr("r", 12).attr("fill", c);
        // Étiquettes centrées au-dessus de chaque point, assez haut pour ne pas chevaucher la courbe.
        const lbl = g.append("text").attr("x", x).attr("y", y - 130).attr("text-anchor", "middle")
          .attr("fill", c).attr("font-size", 34).attr("font-weight", 700).text(text);
        return [dot, lbl];
      };
      const mE = mark(-20, EARLY, `20 cycles trop tôt : ${num(at(-20), 1)}`);
      const mL = mark(20, LATE, `20 cycles trop tard : ${num(at(20), 1)}`);
      return {
        dur: 11.5, simple: true, refs: [curve],
        title: "Trop tôt n'est pas trop tard", subtitle: "",
        captions: [[0.6, 10.8, "Se tromper en retard coûte plus cher que se tromper en avance."]],
        update(u) {
          const a = seg(u, 0.8, 4.0, d3.easeLinear);            // la courbe se trace de gauche à droite
          const xs = P.x0 + a * (P.x1 - P.x0);
          pathE.attr("d", line(neg.filter((p) => p[0] <= xs)));
          pathL.attr("d", line(pos.filter((p) => p[0] <= xs)));
          mE.forEach((e) => e.attr("opacity", seg(u, 4.4, 5.2)));
          mL.forEach((e) => e.attr("opacity", seg(u, 6.4, 7.2)));
        },
      };
    }

    // ---------------------------------------------------------------- scène 3 : le score d'un jeu de moteurs
    function sceneScore(g) {
      const S = [...E].sort((a, b) => b.penalty - a.penalty);
      const P = { x0: 300, x1: 1560, y0: 215, y1: 850 };
      const rowH = (P.y1 - P.y0) / S.length;
      const sx = d3.scaleLinear().domain([0, d3.max(S, (x) => x.penalty)]).range([0, P.x1 - P.x0 - 200]);
      g.append("line").attr("x1", P.x0).attr("x2", P.x0).attr("y1", P.y0 - 6).attr("y2", P.y1).attr("stroke", MUTED).attr("stroke-width", 3);
      const rows = g.selectAll("g.row").data(S).join("g").attr("class", "row")
        .attr("transform", (x, i) => `translate(${P.x0},${P.y0 + i * rowH})`);
      const bars = rows.append("rect").attr("x", 4).attr("y", 4).attr("height", rowH - 10).attr("rx", 6).attr("fill", colorOf);
      const lbls = rows.append("text").attr("y", rowH / 2 + 8).attr("font-size", 28).attr("fill", FG);
      const total = g.append("text").attr("x", P.x1).attr("y", 150).attr("text-anchor", "end").attr("fill", FG)
        .attr("font-size", 46).attr("font-weight", 700);
      g.append("text").attr("x", P.x0).attr("y", 150).attr("fill", MUTED).attr("font-size", 30).text("pénalité de chaque moteur, de la plus lourde à la plus légère");
      return {
        dur: 11.5, simple: true, refs: [eng],
        title: "Le score additionne les pénalités", subtitle: "",
        captions: [[0.6, 10.8, "Le score total additionne les pénalités : les plus grosses erreurs pèsent le plus lourd."]],
        update(u) {
          const w = (i) => seg(u, 1.0 + i * 0.35, 1.8 + i * 0.35);
          bars.attr("width", (x, i) => Math.max(2, sx(x.penalty)) * w(i));
          lbls.attr("x", (x, i) => 4 + Math.max(2, sx(x.penalty)) * w(i) + 18)
            .attr("opacity", (x, i) => w(i)).text((x) => `${num(x.penalty, 1)}  (${fmtErr(x.error)} cycles)`);
          const k = clamp01((u - 1.0) / (0.35 * S.length));
          total.text(`score : ${num(d3.sum(S.slice(0, Math.floor(k * S.length)), (x) => x.penalty), 1)}`).attr("opacity", seg(u, 1.2, 2.0));
        },
      };
    }

    // ---------------------------------------------------------------- scène 4 : deux chiffres par jeu (test)
    function sceneResults(g) {
      const sets = ["FD001", "FD002", "FD003", "FD004"];
      const P = { x0: 300, x1: 1500, y0: 270, y1: 830 };
      const rowH = (P.y1 - P.y0) / sets.length;
      const maxR = d3.max(sets, (s) => Math.max(test.by_subset[s].rmse, test.constant[s].rmse));
      const sx = d3.scaleLinear().domain([0, maxR]).range([0, P.x1 - P.x0]);
      g.append("text").attr("x", P.x0).attr("y", 215).attr("fill", MUTED).attr("font-size", 30)
        .text("erreur typique (RMSE, en cycles) sur le jeu de test");
      const rows = g.selectAll("g.row").data(sets).join("g").attr("class", "row")
        .attr("transform", (s, i) => `translate(0,${P.y0 + i * rowH})`);
      rows.append("text").attr("x", P.x0 - 24).attr("y", rowH / 2 + 4).attr("text-anchor", "end").attr("fill", FG)
        .attr("font-size", 38).attr("font-weight", 700).text((s) => s);
      rows.append("rect").attr("x", P.x0).attr("y", 6).attr("height", 22).attr("rx", 6).attr("fill", "#3a4656")
        .attr("width", (s) => sx(test.constant[s].rmse));
      const bar = rows.append("rect").attr("x", P.x0).attr("y", 34).attr("height", 44).attr("rx", 8).attr("fill", COLORS[3]);
      const rm = rows.append("text").attr("y", 70).attr("font-size", 34).attr("font-weight", 700).attr("fill", FG);
      rows.append("text").attr("x", P.x1 + 60).attr("y", 62).attr("fill", COLORS[2]).attr("font-size", 34).attr("font-weight", 700)
        .text((s) => `score ${int(test.by_subset[s].score)}`);
      const lg = g.append("g").attr("transform", `translate(${P.x0},${P.y1 + 50})`);
      lg.append("rect").attr("width", 28).attr("height", 18).attr("rx", 4).attr("fill", "#3a4656");
      lg.append("text").attr("x", 40).attr("y", 16).attr("fill", MUTED).attr("font-size", 26).text("prédire toujours la moyenne");
      lg.append("rect").attr("x", 520).attr("width", 28).attr("height", 18).attr("rx", 4).attr("fill", COLORS[3]);
      lg.append("text").attr("x", 560).attr("y", 16).attr("fill", MUTED).attr("font-size", 26).text("la baseline");
      return {
        dur: 12, simple: true, refs: [test],
        title: "Deux chiffres pour juger", subtitle: "",
        captions: [[0.6, 11.2, "L'erreur typique, et le score qui punit surtout les retards."]],
        update(u) {
          const a = (i) => seg(u, 1.0 + i * 0.5, 2.2 + i * 0.5);
          bar.attr("width", (s, i) => sx(test.by_subset[s].rmse) * a(i));
          rm.attr("x", (s, i) => P.x0 + sx(test.by_subset[s].rmse) * a(i) + 18).attr("opacity", (s, i) => a(i))
            .text((s) => `${num(test.by_subset[s].rmse, 1)} cycles`);
          rows.attr("opacity", (s, i) => seg(u, 0.8 + i * 0.5, 1.4 + i * 0.5));
        },
      };
    }

    const all = [sceneErrors, sceneCurve, sceneScore, sceneResults];
    return { simple: all, tech: all };
  },
});
