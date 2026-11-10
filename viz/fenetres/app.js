// Scènes de la vidéo « fenêtres glissantes ». Le journal est écrit par scripts/fenetres.py --journal :
// chaque fenêtre du moteur de démonstration y est un événement créé par le vrai code (cmapss/windows.py).
// Le cœur commun (lecture, chronologie, contrôles, enregistrement) est dans viz/lib/core.js.
VizCore.start({
  journal: "journal.jsonl",
  hint: "Lancer scripts/fenetres.py --journal",
  noteFinale: "Illustration fidèle des vrais calculs du programme.",
  scenes(ctx) {
    const { d3, W, COLORS, FG, MUTED, GRID, clamp01, seg, lerp, num, int, by, one, rotLabel, baseline } = ctx;

    const meta = one("meta"), eng = one("engine"), wins = by("window"), N = wins.length;
    const shortEv = one("short_engine"), testEv = one("test_window"), split = one("split");
    const WIN = meta.window;
    const SERIES = "#7fb2ff", WINCOL = COLORS[2], LABEL = COLORS[3], COPY = COLORS[5], TRAIN = "#7fb2ff", VAL = COLORS[5];
    const padExt = (arr) => { const [a, b] = d3.extent(arr), m = (b - a) * 0.1; return [a - m, b + m]; };
    const winRect = (g) => g.append("rect").attr("rx", 12).attr("fill", WINCOL).attr("fill-opacity", 0.14)
      .attr("stroke", WINCOL).attr("stroke-width", 5);

    // Graphique commun aux scènes 1 et 2 : la série du moteur de démonstration
    function engineChart(g) {
      const P = { x0: 240, x1: 1680, y0: 270, y1: 800 };
      const cyc = eng.cycle, n = cyc.length;
      const sx = d3.scaleLinear().domain([cyc[0] - 0.5, cyc[n - 1] + 0.5]).range([P.x0, P.x1]);
      const sy = d3.scaleLinear().domain(padExt(eng.x)).range([P.y1, P.y0]);
      baseline(g, P.x0, P.x1, P.y1 + 20);
      g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", P.y1 + 72).attr("fill", MUTED).attr("font-size", 32)
        .attr("text-anchor", "middle").text("temps (de vol en vol)  →");
      rotLabel(g, P.x0 - 40, (P.y0 + P.y1) / 2, "mesure d'un capteur").attr("font-size", 32);
      g.append("path").attr("fill", "none").attr("stroke", SERIES).attr("stroke-opacity", 0.35).attr("stroke-width", 3)
        .attr("d", d3.line().x((d, i) => sx(cyc[i])).y((d) => sy(d))(eng.x));
      const dots = g.selectAll("circle.dot").data(eng.x).join("circle").attr("class", "dot").attr("r", 7)
        .attr("cx", (d, i) => sx(cyc[i])).attr("cy", (d) => sy(d)).attr("fill", SERIES);
      return { P, sx, sy, dots, cyc, n };
    }

    // ---------------------------------------------------------------- scène 1 : la fenêtre qui glisse
    function sceneSlide(g) {
      const { P, sx, dots, cyc } = engineChart(g);
      const box = winRect(g).attr("y", P.y0 - 40).attr("height", P.y1 - P.y0 + 80);
      const brace = g.append("text").attr("y", P.y0 - 56).attr("text-anchor", "middle").attr("fill", WINCOL)
        .attr("font-size", 36).attr("font-weight", 700).text(`${WIN} cycles`);
      const counter = g.append("text").attr("x", W - 80).attr("y", 118).attr("text-anchor", "end").attr("fill", FG)
        .attr("font-size", 44).attr("font-weight", 700);
      return {
        dur: 12, simple: true, refs: [eng, ...wins],
        title: "Regarder le passé récent du moteur", subtitle: "",
        captions: [[0.6, 11.6, `Le modèle ne regarde pas une mesure isolée : il voit les ${WIN} derniers cycles.`]],
        update(u) {
          const fade = seg(u, 0.1, 1.2, d3.easeLinear);
          const p = seg(u, 1.8, 10.2, d3.easeLinear);
          const k = Math.min(N - 1, Math.floor(p * N));          // chaque pas = une vraie fenêtre du journal
          const w = wins[k];
          const x0 = sx(w.start - 0.5), x1 = sx(w.end + 0.5);
          box.attr("x", x0).attr("width", x1 - x0).attr("opacity", seg(u, 1.2, 1.8));
          brace.attr("x", (x0 + x1) / 2).attr("opacity", seg(u, 1.2, 1.8));
          dots.attr("opacity", (d, i) => fade * (u >= 1.2 && cyc[i] >= w.start && cyc[i] <= w.end ? 1 : 0.3));
          counter.text(`fenêtre n° ${k + 1} sur ${N}`).attr("opacity", seg(u, 1.2, 1.8));
        },
      };
    }

    // ---------------------------------------------------------------- scène 2 : une étiquette par fenêtre
    function sceneLabel(g) {
      const { P, sx, dots, cyc } = engineChart(g);
      const ks = [0, Math.floor((N - 1) / 2), N - 1].map((k) => wins[k]);
      const t0 = (i) => 1.0 + i * 3.4;
      const boxes = ks.map((w) => winRect(g).attr("y", P.y0 - 40).attr("height", P.y1 - P.y0 + 80)
        .attr("x", sx(w.start - 0.5)).attr("width", sx(w.end + 0.5) - sx(w.start - 0.5)));
      const chips = ks.map((w) => g.append("text").attr("x", (sx(w.start - 0.5) + sx(w.end + 0.5)) / 2).attr("y", P.y0 - 62)
        .attr("text-anchor", "middle").attr("fill", LABEL).attr("font-size", 38).attr("font-weight", 700)
        .text(w.rul === 0 ? "panne" : `il reste ${int(w.rul)} cycles`));
      return {
        dur: 11.5, simple: true, refs: [eng, ...ks],
        title: "Une étiquette par fenêtre", subtitle: "",
        captions: [[0.6, 10.8, "À chaque fenêtre, on associe le temps qui reste avant la panne."]],
        update(u) {
          const fade = seg(u, 0.1, 0.9, d3.easeLinear);
          boxes.forEach((b, i) => b.attr("opacity", seg(u, t0(i), t0(i) + 0.6)));
          chips.forEach((c, i) => c.attr("opacity", seg(u, t0(i) + 0.9, t0(i) + 1.6)));
          dots.attr("opacity", (d, i) => fade * (ks.some((w, j) => u >= t0(j) && cyc[i] >= w.start && cyc[i] <= w.end) ? 1 : 0.3));
        },
      };
    }

    // ---------------------------------------------------------------- scène 3 : moteur trop court (remplissage)
    function scenePadding(g) {
      const P = { x0: 300, x1: 1620, y0: 300, y1: 780 };
      const xs = shortEv.x_window, np = shortEv.n_padded, nc = shortEv.n_cycles;
      const slotW = (P.x1 - P.x0) / WIN, sxSlot = (i) => P.x0 + (i + 0.5) * slotW;
      const sy = d3.scaleLinear().domain(padExt(xs)).range([P.y1, P.y0]);
      baseline(g, P.x0, P.x1, P.y1 + 20);
      g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", P.y1 + 72).attr("fill", MUTED).attr("font-size", 32)
        .attr("text-anchor", "middle").text("temps (de vol en vol)  →");
      const box = winRect(g).attr("x", P.x0).attr("width", P.x1 - P.x0).attr("y", P.y0 - 40).attr("height", P.y1 - P.y0 + 80);
      g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", P.y0 - 56).attr("text-anchor", "middle").attr("fill", WINCOL)
        .attr("font-size", 36).attr("font-weight", 700).text(`fenêtre de ${WIN} cycles`);
      // la zone vide, à gauche, avant les vrais cycles
      const gap = g.append("rect").attr("x", P.x0 + 6).attr("width", np * slotW - 12).attr("y", P.y0 - 30)
        .attr("height", P.y1 - P.y0 + 60).attr("rx", 10).attr("fill", "none").attr("stroke", MUTED)
        .attr("stroke-width", 3).attr("stroke-dasharray", "12 10");
      const gapLabel = g.append("text").attr("x", P.x0 + np * slotW / 2).attr("y", (P.y0 + P.y1) / 2)
        .attr("text-anchor", "middle").attr("fill", MUTED).attr("font-size", 36).attr("font-weight", 700)
        .text(`il manque ${np} cycles`);
      const real = d3.range(np, WIN).map((i) => g.append("circle").attr("cx", sxSlot(i)).attr("cy", sy(xs[i])).attr("r", 11).attr("fill", SERIES));
      const copies = d3.range(0, np).map((i) => g.append("circle").attr("cx", sxSlot(i)).attr("cy", sy(xs[i])).attr("r", 11)
        .attr("fill", "none").attr("stroke", COPY).attr("stroke-width", 5));
      const guide = g.append("line").attr("x1", sxSlot(0)).attr("x2", sxSlot(np)).attr("y1", sy(xs[np])).attr("y2", sy(xs[np]))
        .attr("stroke", COPY).attr("stroke-width", 3).attr("stroke-dasharray", "6 8");
      const lblCopy = g.append("text").attr("x", P.x0 + np * slotW / 2).attr("y", P.y1 + 130).attr("text-anchor", "middle")
        .attr("fill", COPY).attr("font-size", 34).attr("font-weight", 700).text(`${np} cycles recopiés`);
      const lblReal = g.append("text").attr("x", P.x0 + (np + (WIN - np) / 2) * slotW).attr("y", P.y1 + 130)
        .attr("text-anchor", "middle").attr("fill", SERIES).attr("font-size", 34).attr("font-weight", 700)
        .text(`${nc} vrais cycles`);
      return {
        dur: 11.5, simple: true, refs: [shortEv],
        title: "Et si le moteur est trop court ?", subtitle: "",
        captions: [
          [0.6, 4.0, `Ce moteur n'a que ${nc} cycles : il en manque ${np} pour remplir la fenêtre.`],
          [4.4, 10.8, "On répète sa première mesure pour compléter."],
        ],
        update(u) {
          const fade = seg(u, 0.1, 1.0, d3.easeLinear);
          box.attr("opacity", fade);
          real.forEach((c, i) => c.attr("opacity", seg(u, 0.4 + i * 0.04, 0.8 + i * 0.04)));
          lblReal.attr("opacity", seg(u, 1.4, 2.0));
          const gp = seg(u, 2.2, 3.0), gOut = seg(u, 4.4, 5.0);
          gap.attr("opacity", gp * (1 - gOut)); gapLabel.attr("opacity", gp * (1 - gOut));
          // les copies apparaissent une à une, de la plus proche à la plus éloignée du premier vrai cycle
          copies.forEach((c, i) => c.attr("opacity", seg(u, 5.0 + (np - 1 - i) * 0.3, 5.3 + (np - 1 - i) * 0.3)));
          guide.attr("opacity", seg(u, 5.0, 5.6));
          lblCopy.attr("opacity", seg(u, 8.6, 9.2));
        },
      };
    }

    // ---------------------------------------------------------------- scène 4 : entraînement, test, validation
    function sceneSplit(g) {
      const gA = g.append("g"), gB = g.append("g");
      // ---- partie A : plein de fenêtres à l'entraînement, une seule au test
      const colTitle = (x, t, c) => gA.append("text").attr("x", x).attr("y", 200).attr("text-anchor", "middle").attr("fill", c)
        .attr("font-size", 46).attr("font-weight", 700).text(t);
      colTitle(500, "Entraînement", TRAIN); colTitle(1400, "Test", COPY);
      const mini = (cx, cycles, vals, x0, x1, y0, y1) => {
        const sx = d3.scaleLinear().domain([cycles[0] - 0.5, cycles.at(-1) + 0.5]).range([x0, x1]);
        const sy = d3.scaleLinear().domain(padExt(vals)).range([y1, y0]);
        gA.append("path").attr("fill", "none").attr("stroke", cx).attr("stroke-width", 3).attr("stroke-opacity", 0.8)
          .attr("d", d3.line().x((d, i) => sx(cycles[i])).y((d) => sy(d))(vals));
        baseline(gA, x0, x1, y1 + 14);
        return sx;
      };
      const sxL = mini(TRAIN, eng.cycle, eng.x, 150, 850, 260, 430);
      const sxR = mini(COPY, testEv.cycle, testEv.x_engine, 1050, 1750, 260, 430);
      // escalier : une barre par vraie fenêtre du journal
      const barH = 3, stairY = 480;
      const bars = gA.selectAll("rect.bar").data(wins).join("rect").attr("class", "bar").attr("rx", 1)
        .attr("x", (w) => sxL(w.start - 0.5)).attr("width", (w) => sxL(w.end + 0.5) - sxL(w.start - 0.5))
        .attr("y", (w, k) => stairY + k * barH).attr("height", barH - 1).attr("fill", WINCOL).attr("fill-opacity", 0.8);
      const nTrain = gA.append("text").attr("x", 500).attr("y", stairY + N * barH + 56).attr("text-anchor", "middle")
        .attr("fill", WINCOL).attr("font-size", 40).attr("font-weight", 700);
      const tBox = winRect(gA).attr("x", sxR(testEv.start - 0.5)).attr("width", sxR(testEv.end + 0.5) - sxR(testEv.start - 0.5))
        .attr("y", 240).attr("height", 220);
      const tLbl = gA.append("text").attr("x", 1400).attr("y", 600).attr("text-anchor", "middle").attr("fill", WINCOL)
        .attr("font-size", 40).attr("font-weight", 700).text("1 seule fenêtre");
      const tLbl2 = gA.append("text").attr("x", 1400).attr("y", 652).attr("text-anchor", "middle").attr("fill", MUTED)
        .attr("font-size", 32).text(`les ${WIN} derniers cycles`);
      const tRul = gA.append("text").attr("x", 1400).attr("y", 730).attr("text-anchor", "middle").attr("fill", LABEL)
        .attr("font-size", 36).attr("font-weight", 700).text(`on connaît la vraie durée restante : ${int(testEv.rul)} cycles`);

      // ---- partie B : découpage par moteur
      const ex = split.examples, maxW = d3.max(ex, (e) => e.windows);
      const sw = d3.scaleLinear().domain([0, maxW]).range([0, 560]);
      const legend = (x, y, c, l1, l2) => {
        gB.append("rect").attr("x", x).attr("y", y - 30).attr("width", 36).attr("height", 36).attr("rx", 6).attr("fill", c);
        gB.append("text").attr("x", x + 56).attr("y", y).attr("fill", FG).attr("font-size", 36).attr("font-weight", 700).text(l1);
        gB.append("text").attr("x", x + 56).attr("y", y + 40).attr("fill", MUTED).attr("font-size", 28).text(l2);
      };
      legend(250, 230, TRAIN, "Entraînement", `${int(split.train_engines)} moteurs, ${int(split.train_windows)} fenêtres`);
      legend(1050, 230, VAL, "Validation", `${int(split.val_engines)} moteurs, ${int(split.val_windows)} fenêtres`);
      // chaque côté dans sa colonne : un moteur n'apparaît que d'un côté
      const cols = ["entrainement", "validation"];
      const placed = cols.flatMap((side, c) => ex.filter((e) => e.side === side).map((e, r) => ({ ...e, c, r })));
      const rowsG = gB.selectAll("g.row").data(placed).join("g").attr("class", "row")
        .attr("transform", (e) => `translate(${e.c === 0 ? 250 : 1050},${340 + e.r * 84})`);
      rowsG.append("rect").attr("width", (e) => sw(e.windows)).attr("height", 50).attr("rx", 8)
        .attr("fill", (e) => (e.side === "validation" ? VAL : TRAIN));
      rowsG.append("text").attr("x", (e) => sw(e.windows) + 16).attr("y", 36).attr("fill", MUTED).attr("font-size", 28)
        .text((e) => `${int(e.windows)} fenêtres`);

      return {
        dur: 15, simple: true, refs: [eng, testEv, one("counts"), split],
        title: "Entraînement, test et validation", subtitle: "",
        captions: [
          [0.6, 6.4, "Entraînement : plein de fenêtres par moteur. Test : une seule."],
          [7.8, 14.4, "Un moteur reste entier d'un seul côté : ses fenêtres se ressemblent trop."],
        ],
        update(u) {
          const a = seg(u, 0.1, 0.9, d3.easeLinear) * (1 - seg(u, 6.6, 7.2, d3.easeLinear));
          gA.attr("opacity", a);
          const k = Math.floor(seg(u, 1.2, 4.4, d3.easeLinear) * N);
          bars.attr("opacity", (w, i) => (i < k ? 1 : 0));
          nTrain.text(`${int(k)} fenêtres pour ce seul moteur`);
          tBox.attr("opacity", seg(u, 4.8, 5.6)); tLbl.attr("opacity", seg(u, 5.2, 5.9));
          tLbl2.attr("opacity", seg(u, 5.2, 5.9)); tRul.attr("opacity", seg(u, 5.8, 6.4));
          gB.attr("opacity", seg(u, 7.2, 8.0, d3.easeLinear));
          rowsG.attr("opacity", (e, i) => seg(u, 8.2 + i * 0.25, 8.7 + i * 0.25));
        },
      };
    }

    const all = [sceneSlide, sceneLabel, scenePadding, sceneSplit];
    return { simple: all, tech: all };
  },
});
