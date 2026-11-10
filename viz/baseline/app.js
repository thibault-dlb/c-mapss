// Scènes de la vidéo « baseline » (statistiques de fenêtre + Gradient Boosting). Le journal est écrit par
// scripts/baseline.py --journal : une vraie fenêtre de validation et ses 6 statistiques, les prédictions réelles du modèle
// après 0, 1, 2... itérations (`staged_predict`), les résultats réels de la validation croisée et du test.
// Le cœur commun est dans viz/lib/core.js.
VizCore.start({
  journal: "journal.jsonl",
  hint: "Lancer scripts/baseline.py --journal",
  noteFinale: "Illustration fidèle des vrais calculs du programme.",
  scenes(ctx) {
    const { d3, W, COLORS, FG, MUTED, GRID, clamp01, seg, lerp, num, int, by, one, rotLabel, baseline } = ctx;

    const meta = one("meta"), win = one("window_stats"), lb = one("learning_begin"), stages = by("stage");
    const choice = one("choice"), fin = one("final");
    const SERIES = "#7fb2ff";
    const padExt = (arr) => { const [a, b] = d3.extent(arr), m = (b - a) * 0.12; return [a - m, b + m]; };

    // ---------------------------------------------------------------- scène 1 : résumer une fenêtre
    function sceneSummary(g) {
      const P = { x0: 260, x1: 1560, y0: 250, y1: 800 };
      const n = win.values.length;
      const sx = d3.scaleLinear().domain([0, n - 1]).range([P.x0 + 20, P.x1 - 20]);
      const sy = d3.scaleLinear().domain(padExt([...win.values, win.min, win.max])).range([P.y1, P.y0]);
      baseline(g, P.x0, P.x1, P.y1 + 20);
      g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", P.y1 + 74).attr("fill", MUTED).attr("font-size", 32)
        .attr("text-anchor", "middle").text("les 30 derniers cycles  →");
      rotLabel(g, P.x0 - 50, (P.y0 + P.y1) / 2, "mesure d'un capteur").attr("font-size", 32);
      g.append("path").attr("fill", "none").attr("stroke", SERIES).attr("stroke-opacity", 0.4).attr("stroke-width", 3)
        .attr("d", d3.line().x((d, i) => sx(i)).y((d) => sy(d))(win.values));
      const dots = g.selectAll("circle.dot").data(win.values).join("circle").attr("class", "dot").attr("r", 8)
        .attr("cx", (d, i) => sx(i)).attr("cy", (d) => sy(d)).attr("fill", SERIES);
      // les 6 statistiques, lues dans le journal (calculées par window_features)
      const hline = (v, c, label, dy = -14) => {
        const gg = g.append("g");
        gg.append("line").attr("x1", P.x0).attr("x2", P.x1).attr("y1", sy(v)).attr("y2", sy(v)).attr("stroke", c)
          .attr("stroke-width", 5).attr("stroke-dasharray", "14 10");
        gg.append("text").attr("x", P.x1 + 16).attr("y", sy(v) + 10).attr("fill", c).attr("font-size", 32).attr("font-weight", 700).text(label);
        return gg;
      };
      const band = g.append("rect").attr("x", P.x0).attr("width", P.x1 - P.x0).attr("y", sy(win.mean + win.std))
        .attr("height", sy(win.mean - win.std) - sy(win.mean + win.std)).attr("fill", COLORS[4]).attr("fill-opacity", 0.18);
      const bandLbl = g.append("text").attr("x", P.x1 + 16).attr("y", sy(win.mean + win.std) + 10).attr("fill", COLORS[4])
        .attr("font-size", 32).attr("font-weight", 700).text("dispersion");
      const sMean = hline(win.mean, COLORS[2], "moyenne");
      const sMin = hline(win.min, COLORS[0], "plus bas");
      const sMax = hline(win.max, COLORS[1], "plus haut");
      const lastDot = g.append("circle").attr("cx", sx(n - 1)).attr("cy", sy(win.last)).attr("r", 18).attr("fill", "none")
        .attr("stroke", COLORS[3]).attr("stroke-width", 7);
      // sous le cercle : la dernière valeur est souvent proche de la moyenne, une étiquette au-dessus chevaucherait « moyenne »
      const lastLbl = g.append("text").attr("x", sx(n - 1) + 40).attr("y", sy(win.last) + 84).attr("fill", COLORS[3]).attr("font-size", 32)
        .attr("font-weight", 700).attr("text-anchor", "end").text("dernière valeur");
      const slope = g.append("line").attr("x1", sx(0)).attr("x2", sx(n - 1)).attr("y1", sy(win.line[0])).attr("y2", sy(win.line[1]))
        .attr("stroke", "#ffffff").attr("stroke-width", 9).attr("stroke-linecap", "round");
      const slopeLbl = g.append("text").attr("x", P.x1 + 16).attr("y", sy(win.line[1]) - 2).attr("fill", "#ffffff").attr("font-size", 32)
        .attr("font-weight", 700).text("tendance");
      const t0 = (i) => 1.8 + i * 1.6;
      return {
        dur: 13.5, simple: true, refs: [win],
        title: "Résumer une fenêtre en quelques chiffres", subtitle: "",
        captions: [[0.6, 12.8, "Chaque fenêtre est résumée : moyenne, extrêmes, dispersion, dernière valeur, tendance."]],
        update(u) {
          dots.attr("opacity", seg(u, 0.1, 1.0, d3.easeLinear));
          sMean.attr("opacity", seg(u, t0(0), t0(0) + 0.6));
          sMin.attr("opacity", seg(u, t0(1), t0(1) + 0.6)); sMax.attr("opacity", seg(u, t0(1), t0(1) + 0.6));
          band.attr("opacity", seg(u, t0(2), t0(2) + 0.6)); bandLbl.attr("opacity", seg(u, t0(2), t0(2) + 0.6));
          lastDot.attr("opacity", seg(u, t0(3), t0(3) + 0.6)); lastLbl.attr("opacity", seg(u, t0(3), t0(3) + 0.6));
          slope.attr("opacity", seg(u, t0(4), t0(4) + 0.6)); slopeLbl.attr("opacity", seg(u, t0(4), t0(4) + 0.6));
        },
      };
    }

    // ---------------------------------------------------------------- scène 2 : apprendre étape par étape
    function sceneLearning(g) {
      const P = { x0: 260, x1: 1300, y0: 250, y1: 800 };
      const cyc = lb.end_cycle;
      const sx = d3.scaleLinear().domain(d3.extent(cyc)).range([P.x0 + 10, P.x1 - 10]);
      const sy = d3.scaleLinear().domain([0, d3.max([...lb.true, ...lb.true_capped, ...stages.flatMap((s) => s.pred)])]).range([P.y1, P.y0]);
      baseline(g, P.x0, P.x1, P.y1 + 16);
      g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", P.y1 + 72).attr("fill", MUTED).attr("font-size", 32)
        .attr("text-anchor", "middle").text("vie du moteur  →");
      rotLabel(g, P.x0 - 50, (P.y0 + P.y1) / 2, "durée restante (cycles)").attr("font-size", 32);
      const line = (arr) => d3.line().x((d, i) => sx(cyc[i])).y((d) => sy(d))(arr);
      g.append("path").attr("d", line(lb.true)).attr("fill", "none").attr("stroke", "#c8d0da").attr("stroke-width", 6).attr("stroke-dasharray", "4 12").attr("stroke-linecap", "round");
      g.append("path").attr("d", line(lb.true_capped)).attr("fill", "none").attr("stroke", "#c8d0da").attr("stroke-width", 6).attr("stroke-opacity", 0.5);
      const pred = g.append("path").attr("fill", "none").attr("stroke", COLORS[3]).attr("stroke-width", 8).attr("stroke-linejoin", "round");
      const lg = g.append("g").attr("transform", `translate(${P.x0},215)`);
      lg.append("line").attr("x2", 56).attr("stroke", "#c8d0da").attr("stroke-width", 6).attr("stroke-dasharray", "4 12");
      lg.append("text").attr("x", 72).attr("y", 10).attr("fill", FG).attr("font-size", 28).text("vraie durée restante");
      lg.append("line").attr("x1", 460).attr("x2", 516).attr("stroke", COLORS[3]).attr("stroke-width", 8);
      lg.append("text").attr("x", 532).attr("y", 10).attr("fill", FG).attr("font-size", 28).text("prédiction du modèle");
      lg.append("line").attr("x1", 810).attr("x2", 866).attr("stroke", "#c8d0da").attr("stroke-opacity", 0.5).attr("stroke-width", 6);
      lg.append("text").attr("x", 882).attr("y", 10).attr("fill", FG).attr("font-size", 28).text(`plafond ${lb.cap}`);
      const stepTxt = g.append("text").attr("x", P.x1 + 80).attr("y", 300).attr("fill", FG).attr("font-size", 44).attr("font-weight", 700);
      const errTxt = g.append("text").attr("x", P.x1 + 80).attr("y", 356).attr("fill", MUTED).attr("font-size", 32);
      // petite courbe d'apprentissage : erreur moyenne en fonction de l'étape, curseur sur l'étape affichée
      const Q = { x0: P.x1 + 90, x1: 1830, y0: 470, y1: 760 };
      const qx = d3.scaleLinear().domain([0, lb.n_iter]).range([Q.x0, Q.x1]);
      const qy = d3.scaleLinear().domain([0, d3.max([lb.start_rmse, ...lb.rmse_curve])]).range([Q.y1, Q.y0]);
      baseline(g, Q.x0, Q.x1, Q.y1);
      g.append("text").attr("x", Q.x0).attr("y", Q.y0 - 16).attr("fill", MUTED).attr("font-size", 26).text("erreur moyenne (cycles)");
      g.append("text").attr("x", (Q.x0 + Q.x1) / 2).attr("y", Q.y1 + 44).attr("fill", MUTED).attr("font-size", 26).attr("text-anchor", "middle").text("étapes d'apprentissage");
      g.append("path").attr("fill", "none").attr("stroke", MUTED).attr("stroke-width", 4)
        .attr("d", d3.line().x((d, i) => qx(i + 1)).y((d) => qy(d))(lb.rmse_curve));
      const cursor = g.append("circle").attr("r", 12).attr("fill", COLORS[3]);
      const tStage = (k) => 1.4 + k * 0.75;
      return {
        dur: 13, simple: true, refs: [lb, ...stages],
        title: "Apprendre étape par étape", subtitle: "",
        captions: [[0.6, 12.4, "À chaque étape, le modèle corrige ses erreurs de l'étape précédente."]],
        update(u) {
          const k = Math.max(0, Math.min(stages.length - 1, Math.floor((u - 1.4) / 0.75)));   // un instantané réel par palier
          const s = u < 1.4 ? stages[0] : stages[k];
          pred.attr("d", line(s.pred)).attr("opacity", seg(u, 0.4, 1.0, d3.easeLinear));
          stepTxt.text(s.iter === 0 ? "départ : la moyenne" : `étape ${s.iter}`);
          errTxt.text(`erreur moyenne : ${num(s.rmse_val, 1)} cycles`);
          cursor.attr("cx", qx(s.iter)).attr("cy", qy(s.rmse_val));
        },
      };
    }

    // ---------------------------------------------------------------- scène 3 : choisir les réglages
    function sceneChoice(g) {
      const S = choice.summary, best = choice.best;
      const P = { x0: 560, x1: 1540, y0: 230, y1: 800 };
      const rowH = (P.y1 - P.y0) / S.length;
      const sx = d3.scaleLinear().domain([0, d3.max(S, (s) => s.rmse_mean + s.rmse_std)]).range([0, P.x1 - P.x0]);
      g.append("text").attr("x", P.x0).attr("y", 190).attr("fill", MUTED).attr("font-size", 30).text("erreur moyenne sur des moteurs jamais vus (cycles)");
      const isBest = (s) => s.sensors === best.sensors && s.cap === best.cap;
      const rows = g.selectAll("g.row").data(S).join("g").attr("class", "row")
        .attr("transform", (s, i) => `translate(0,${P.y0 + i * rowH})`);
      rows.append("text").attr("x", P.x0 - 24).attr("y", rowH / 2 + 4).attr("text-anchor", "end").attr("font-size", 30)
        .attr("fill", (s) => (isBest(s) ? COLORS[3] : FG)).attr("font-weight", (s) => (isBest(s) ? 700 : 400))
        .text((s) => `plafond ${s.cap} · ${s.sensors} capteurs`);
      const bars = rows.append("rect").attr("x", P.x0).attr("y", 8).attr("height", rowH - 22).attr("rx", 8)
        .attr("fill", (s) => (isBest(s) ? COLORS[3] : "#4b5a6e"));
      const vals = rows.append("text").attr("y", rowH / 2 + 6).attr("font-size", 30).attr("font-weight", 700).attr("fill", FG);
      return {
        dur: 12, simple: true, refs: [choice],
        title: "Choisir les réglages", subtitle: "",
        captions: [[0.6, 11.2, "On garde la combinaison qui se trompe le moins sur des moteurs qu'elle n'a jamais vus."]],
        update(u) {
          const a = (i) => seg(u, 1.0 + i * 0.5, 2.0 + i * 0.5);
          bars.attr("width", (s, i) => sx(s.rmse_mean) * a(i));
          vals.attr("x", (s, i) => P.x0 + sx(s.rmse_mean) * a(i) + 16).attr("opacity", (s, i) => a(i)).text((s) => num(s.rmse_mean, 1));
        },
      };
    }

    // ---------------------------------------------------------------- scène 4 : le score à battre
    function sceneTarget(g) {
      const sets = ["FD001", "FD002", "FD003", "FD004"];
      const P = { x0: 300, x1: 1500, y0: 270, y1: 830 };
      const rowH = (P.y1 - P.y0) / sets.length;
      const maxR = d3.max(sets, (s) => Math.max(fin.test[s].rmse, fin.constant[s].rmse));
      const sx = d3.scaleLinear().domain([0, maxR]).range([0, P.x1 - P.x0]);
      g.append("text").attr("x", P.x0).attr("y", 215).attr("fill", MUTED).attr("font-size", 30).text("erreur typique (RMSE, en cycles) sur le jeu de test");
      const rows = g.selectAll("g.row").data(sets).join("g").attr("class", "row")
        .attr("transform", (s, i) => `translate(0,${P.y0 + i * rowH})`);
      rows.append("text").attr("x", P.x0 - 24).attr("y", rowH / 2 + 4).attr("text-anchor", "end").attr("fill", FG)
        .attr("font-size", 38).attr("font-weight", 700).text((s) => s);
      rows.append("rect").attr("x", P.x0).attr("y", 6).attr("height", 22).attr("rx", 6).attr("fill", "#3a4656").attr("width", (s) => sx(fin.constant[s].rmse));
      const bar = rows.append("rect").attr("x", P.x0).attr("y", 34).attr("height", 44).attr("rx", 8).attr("fill", COLORS[3]);
      const rm = rows.append("text").attr("y", 70).attr("font-size", 34).attr("font-weight", 700).attr("fill", FG);
      const lg = g.append("g").attr("transform", `translate(${P.x0},${P.y1 + 50})`);
      lg.append("rect").attr("width", 28).attr("height", 18).attr("rx", 4).attr("fill", "#3a4656");
      lg.append("text").attr("x", 40).attr("y", 16).attr("fill", MUTED).attr("font-size", 26).text("prédire toujours la moyenne");
      lg.append("rect").attr("x", 520).attr("width", 28).attr("height", 18).attr("rx", 4).attr("fill", COLORS[3]);
      lg.append("text").attr("x", 560).attr("y", 16).attr("fill", MUTED).attr("font-size", 26).text("la baseline");
      return {
        dur: 12, simple: true, refs: [fin],
        title: "Le score à battre", subtitle: "",
        captions: [[0.6, 11.2, "Voilà le résultat à battre pour le futur réseau de neurones."]],
        update(u) {
          const a = (i) => seg(u, 1.0 + i * 0.5, 2.2 + i * 0.5);
          rows.attr("opacity", (s, i) => seg(u, 0.8 + i * 0.5, 1.4 + i * 0.5));
          bar.attr("width", (s, i) => sx(fin.test[s].rmse) * a(i));
          rm.attr("x", (s, i) => P.x0 + sx(fin.test[s].rmse) * a(i) + 18).attr("opacity", (s, i) => a(i)).text((s) => `${num(fin.test[s].rmse, 1)} cycles`);
        },
      };
    }

    const all = [sceneSummary, sceneLearning, sceneChoice, sceneTarget];
    return { simple: all, tech: all };
  },
});
