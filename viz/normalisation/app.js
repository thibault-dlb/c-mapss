// Scènes de la vidéo « normalisation par condition de vol ». Le journal est écrit par scripts/normalise.py --journal ;
// le cœur commun (lecture, chronologie, contrôles, enregistrement) est dans viz/lib/core.js.
VizCore.start({
  journal: "journal.jsonl",
  hint: "Lancer scripts/normalise.py --journal",
  noteFinale: "Illustration fidèle des vrais calculs du programme.",
  scenes(ctx) {
    const { d3, W, H, COLORS, FG, MUTED, BG, GRID, WARN, AMBER, clamp01, ease, seg, lerp, num, sci, int, events, by, one,
      head, meta, axisBottom, axisLeft, rotLabel, slideTag, tagOpacity, near, seqRange, slideOp, baseline } = ctx;

    // ---------------------------------------------------------------- scène 1 : lecture et conditions de vol
    function sceneConditions(g) {
      const reads = by("read"), condEv = one("conditions"), conds = condEv.conditions, samp = one("settings_sample");
      const pts = samp.points; // [altitude, Mach, TRA, condition, jeu]
      const P = { x0: 200, x1: 1240, y0: 200, y1: 850 };
      const sx = d3.scaleLinear().domain([-4, 46]).range([P.x0, P.x1]);
      const sy = d3.scaleLinear().domain([-0.06, 0.92]).range([P.y1, P.y0]);
      axisBottom(g, sx, P.y1, 6, (d) => num(d, 0));
      axisLeft(g, sy, P.x0, 6, (d) => num(d, 1));
      g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", P.y1 + 66).attr("fill", MUTED)
        .attr("font-size", 26).attr("text-anchor", "middle").text("réglage 1 : altitude");
      rotLabel(g, P.x0 - 100, (P.y0 + P.y1) / 2, "réglage 2 : Mach");

      const nMax = d3.max(conds, (c) => c.n);
      const ringR = (c) => 14 + 58 * Math.sqrt(c.n / nMax);
      const rings = g.selectAll("circle.ring").data(conds).join("circle").attr("class", "ring")
        .attr("cx", (c) => sx(c.altitude)).attr("cy", (c) => sy(c.mach))
        .attr("fill", "none").attr("stroke-width", 4).attr("stroke", (c, i) => COLORS[i]);
      const dots = g.selectAll("circle.pt").data(pts).join("circle").attr("class", "pt")
        .attr("cx", (d) => sx(d[0])).attr("cy", (d) => sy(d[1])).attr("r", 4.5);
      const tags = g.selectAll("text.tag").data(conds).join("text").attr("class", "tag")
        .attr("x", (c) => sx(c.altitude)).attr("y", (c) => sy(c.mach) - ringR(c) - 14)
        .attr("text-anchor", "middle").attr("font-size", 34).attr("font-weight", 700)
        .attr("fill", (c, i) => COLORS[i]).text((c, i) => i + 1);

      // Colonne de droite : lecture jeu par jeu, puis légende des conditions
      const LX = 1320;
      const tRead = (i) => 0.4 + i * 1.1;
      const readRows = g.selectAll("g.read").data(reads).join("g").attr("class", "read")
        .attr("transform", (d, i) => `translate(${LX},${215 + i * 42})`);
      readRows.append("text").attr("fill", FG).attr("font-size", 28).attr("font-weight", 700).text((d) => d.subset);
      readRows.append("text").attr("x", 120).attr("fill", MUTED).attr("font-size", 26)
        .text((d) => `${int(d.rows)} cycles · ${d.engines} moteurs`);
      const total = g.append("text").attr("x", LX).attr("y", 215 + 4 * 42 + 6).attr("fill", FG)
        .attr("font-size", 30).attr("font-weight", 700);

      const legend = g.selectAll("g.leg").data(conds).join("g").attr("class", "leg")
        .attr("transform", (c, i) => `translate(${LX},${470 + i * 66})`);
      legend.append("circle").attr("r", 12).attr("cx", 12).attr("cy", -8).attr("fill", (c, i) => COLORS[i]);
      legend.append("text").attr("x", 38).attr("fill", FG).attr("font-size", 26).text((c, i) => `Condition ${i + 1}`);
      const legN = legend.append("text").attr("x", 520).attr("fill", FG).attr("font-size", 26).attr("text-anchor", "end");
      legend.append("text").attr("x", 38).attr("y", 28).attr("fill", MUTED).attr("font-size", 22)
        .text((c) => `altitude ${num(c.altitude, 0)} · Mach ${num(c.mach)} · TRA ${num(c.tra, 0)} · jeux ${c.subsets.map((s) => s.replace("FD00", "")).join("·")}`);

      return {
        dur: 12.5,
        title: "Lecture des données et conditions de vol",
        subtitle: "Chaque cycle appartient à l'une des 6 conditions de vol",
        refs: [...reads, condEv, samp],
        captions: [
          [0.3, 4.7, "Lecture des 4 jeux d'entraînement, l'un après l'autre"],
          [5.0, 8.2, "Les 3 réglages ne prennent que 6 combinaisons : 6 conditions de vol"],
          [8.4, 12.2, "FD001 et FD003 n'utilisent que la condition 1 ; FD002 et FD004 utilisent les 6"],
        ],
        update(u) {
          readRows.attr("opacity", (d, i) => seg(u, tRead(i), tRead(i) + 0.4));
          let cum = 0, eng = 0;
          reads.forEach((d, i) => { if (u >= tRead(i)) { cum += d.rows; eng += d.engines; } });
          total.text(`Total : ${int(cum)} cycles · ${eng} moteurs`);
          const col = seg(u, 5.0, 6.2);
          dots.attr("opacity", (d) => (u >= tRead(d[4]) ? 0.6 : 0))
            .attr("fill", (d) => (col <= 0 ? "#9aa7b5" : d3.interpolateRgb("#9aa7b5", COLORS[d[3]])(col)));
          const ring = seg(u, 6.2, 7.6);
          rings.attr("r", (c) => ringR(c) * ring).attr("opacity", ring);
          tags.attr("opacity", ring);
          legend.attr("opacity", seg(u, 5.4, 6.4));
          legN.text((c) => `${int(c.n)} cycles`);
        },
      };
    }

    // ---------------------------------------------------------------- scène 2 : calcul de μ et σ, bloc par bloc
    function sceneStats(g) {
      const begin = one("stats_begin"), fin = one("stats_final"), chk = one("check");
      const ups = by("stats_update"), N = ups.length, conds = one("conditions").conditions;
      const sensor = begin.sensor;
      const P = { x0: 200, x1: 1440 };
      const sx = d3.scaleLinear().domain([0, begin.n_rows]).range([P.x0, P.x1]);
      const allSd = ups.flatMap((e) => e.cond.filter(Boolean).map((c) => c[2]));
      // μ : une bande par condition, chacune avec sa propre échelle (sinon l'écart entre conditions écrase la variation)
      const STRIP = { y0: 195, h: 62, gap: 6 };
      const stripTop = (k) => STRIP.y0 + k * (STRIP.h + STRIP.gap);
      const syMu = conds.map((c, k) => {
        const vals = ups.filter((e) => e.cond[k]).map((e) => e.cond[k][1]);
        const [a, b] = d3.extent(vals), m = Math.max((b - a) * 0.15, 0.05);
        return d3.scaleLinear().domain([a - m, b + m]).range([stripTop(k) + STRIP.h, stripTop(k)]);
      });
      const sySd = d3.scaleLinear().domain([0, d3.max(allSd) * 1.1]).range([840, 660]);

      conds.forEach((c, k) => {
        g.append("rect").attr("x", P.x0).attr("y", stripTop(k)).attr("width", P.x1 - P.x0).attr("height", STRIP.h)
          .attr("fill", "#141b23");
        const [lo, hi] = syMu[k].domain();
        g.append("text").attr("x", P.x0 - 10).attr("y", stripTop(k) + 14).attr("text-anchor", "end")
          .attr("fill", MUTED).attr("font-size", 17).text(num(hi, 1));
        g.append("text").attr("x", P.x0 - 10).attr("y", stripTop(k) + STRIP.h - 2).attr("text-anchor", "end")
          .attr("fill", MUTED).attr("font-size", 17).text(num(lo, 1));
        g.append("circle").attr("cx", P.x0 + 14).attr("cy", stripTop(k) + 14).attr("r", 7).attr("fill", COLORS[k]);
      });
      axisLeft(g, sySd, P.x0, 4, (d) => num(d, 1));
      axisBottom(g, sx, 840, 8, (d) => int(d));
      g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", 893).attr("fill", MUTED).attr("font-size", 26)
        .attr("text-anchor", "middle").text("lignes lues (dans l'ordre des données : FD001, FD002, FD003, FD004)");
      rotLabel(g, P.x0 - 130, 400, `μ de ${sensor} (échelle par bande)`);
      rotLabel(g, P.x0 - 110, 750, `σ de ${sensor}`);
      // Frontières entre jeux : ce sont les lignes réellement lues
      let acc = 0;
      by("read").forEach((d, i) => {
        if (i > 0) {
          g.append("line").attr("x1", sx(acc)).attr("x2", sx(acc)).attr("y1", 195).attr("y2", 840)
            .attr("stroke", GRID).attr("stroke-width", 2).attr("stroke-dasharray", "6 8");
        }
        g.append("text").attr("x", sx(acc + d.rows / 2)).attr("y", 182).attr("text-anchor", "middle")
          .attr("fill", MUTED).attr("font-size", 22).text(d.subset);
        acc += d.rows;
      });

      const lineMu = conds.map((c, k) => d3.line().defined((p) => p != null).x((p) => sx(p[0])).y((p) => syMu[k](p[1])));
      const lineSd = d3.line().defined((p) => p != null).x((p) => sx(p[0])).y((p) => sySd(p[1]));
      const pathsMu = conds.map((c, k) => g.append("path").attr("fill", "none").attr("stroke", COLORS[k]).attr("stroke-width", 4));
      const pathsSd = conds.map((c, k) => g.append("path").attr("fill", "none").attr("stroke", COLORS[k]).attr("stroke-width", 4));
      const headMu = conds.map((c, k) => g.append("circle").attr("r", 7).attr("fill", COLORS[k]));
      const headSd = conds.map((c, k) => g.append("circle").attr("r", 7).attr("fill", COLORS[k]));

      // Panneau de lecture : valeurs courantes, lues telles quelles dans l'événement en cours
      const RX = 1530;
      const panel = g.append("g").attr("transform", `translate(${RX},210)`);
      const pBlock = panel.append("text").attr("fill", FG).attr("font-size", 30).attr("font-weight", 700);
      const pRows = panel.append("text").attr("y", 40).attr("fill", MUTED).attr("font-size", 24);
      const pSub = panel.append("text").attr("y", 74).attr("fill", MUTED).attr("font-size", 24);
      const pEv = panel.append("text").attr("y", 108).attr("fill", MUTED).attr("font-size", 22);
      const rows = panel.selectAll("g.row").data(conds).join("g").attr("class", "row")
        .attr("transform", (c, i) => `translate(0,${170 + i * 100})`);
      rows.append("circle").attr("r", 10).attr("cx", 10).attr("cy", -8).attr("fill", (c, i) => COLORS[i]);
      rows.append("text").attr("x", 32).attr("fill", FG).attr("font-size", 24).text((c, i) => `Condition ${i + 1}`);
      const rowA = rows.append("text").attr("x", 32).attr("y", 30).attr("fill", MUTED).attr("font-size", 22);
      const rowB = rows.append("text").attr("x", 32).attr("y", 58).attr("fill", MUTED).attr("font-size", 22);

      const hist = (e, which) => conds.map((c, k) => ups.slice(0, e + 1)
        .map((x) => (x.cond[k] ? [x.rows_seen, x.cond[k][which]] : null)));

      const finalNote = g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", 932).attr("text-anchor", "middle")
        .attr("fill", COLORS[3]).attr("font-size", 28).attr("font-weight", 700)
        .text(`Contrôle : écart avec le calcul direct de pandas ≤ ${sci(Math.max(chk.max_abs_diff_mean, chk.max_abs_diff_std))}`);

      return {
        dur: 14,
        title: "Calcul de μ et σ, bloc par bloc",
        subtitle: `Statistiques de ${sensor} par condition de vol, mises à jour à chaque bloc de ${int(begin.block)} lignes`,
        refs: [begin, ...ups, fin, chk],
        captions: [
          [0.3, 2.3, `Le calcul lit les données par blocs de ${int(begin.block)} lignes`],
          [2.5, 5.0, "FD001 n'a que la condition 1 : les autres n'apparaissent qu'avec FD002"],
          [5.2, 10.6, "μ et σ de chaque condition se précisent à mesure que les lignes sont lues"],
          [11.2, 13.8, "Valeurs finales : celles qui serviront à normaliser (identiques au calcul direct)"],
        ],
        update(u) {
          const p = seg(u, 0.8, 10.6, d3.easeLinear);
          const e = Math.min(N - 1, Math.floor(p * N));
          const cur = ups[e];
          conds.forEach((c, k) => {
            const hm = hist(e, 1)[k], hs = hist(e, 2)[k];
            pathsMu[k].attr("d", lineMu[k](hm)); pathsSd[k].attr("d", lineSd(hs));
            const last = cur.cond[k];
            const lm = hm.filter(Boolean).at(-1), ls = hs.filter(Boolean).at(-1);
            headMu[k].attr("opacity", last ? 1 : 0).attr("cx", lm ? sx(lm[0]) : 0).attr("cy", lm ? syMu[k](lm[1]) : 0);
            headSd[k].attr("opacity", last ? 1 : 0).attr("cx", ls ? sx(ls[0]) : 0).attr("cy", ls ? sySd(ls[1]) : 0);
          });
          pBlock.text(`Bloc ${cur.block + 1} / ${cur.n_blocks}`);
          pRows.text(`Lignes lues : ${int(cur.rows_seen)}`);
          pSub.text(`Jeu lu : ${cur.subsets.join(", ")}`);
          pEv.text(`événement n°${cur.seq} du journal`);
          rowA.text((c, k) => (cur.cond[k] ? `μ = ${num(cur.cond[k][1], 2)} · σ = ${num(cur.cond[k][2], 3)}` : "pas encore vue"));
          rowB.text((c, k) => (cur.cond[k] ? `${int(cur.cond[k][0])} lignes` : ""));
          finalNote.attr("opacity", seg(u, 10.8, 11.6));
        },
      };
    }

    // ---------------------------------------------------------------- scène 3 : un moteur, trois états réels
    function sceneEngine(g) {
      const ev = one("engine_states"), S = ev.sensors[ev.demo_sensor], n = ev.cycle.length;
      const P = { x0: 200, x1: 1500, y0: 230, y1: 860 };
      const sx = d3.scaleLinear().domain(d3.extent(ev.cycle)).range([P.x0 + 20, P.x1 - 20]);
      const padded = (arr) => { const [a, b] = d3.extent(arr), m = (b - a) * 0.08; return [a - m, b + m]; };
      const sy = ["raw", "centered", "scaled"].map((k) => d3.scaleLinear().domain(padded(S[k])).range([P.y1, P.y0]));
      const fmts = [(d) => num(d, 0), (d) => num(d, 0), (d) => num(d, 1)];
      const axes = sy.map((s, k) => axisLeft(g, s, P.x0, 6, fmts[k]));
      const labels = [
        `${ev.demo_sensor} brut (unité du capteur)`,
        `${ev.demo_sensor} − μ (unité du capteur)`,
        `${ev.demo_sensor} normalisé (en écarts-types)`,
      ].map((t) => rotLabel(g, P.x0 - 110, (P.y0 + P.y1) / 2, t));
      axisBottom(g, sx, P.y1, 7, (d) => num(d, 0));
      g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", P.y1 + 66).attr("fill", MUTED)
        .attr("font-size", 26).attr("text-anchor", "middle").text(`cycle (moteur ${ev.engine}, couleur = condition de vol)`);

      const zeroLine = g.append("line").attr("x1", P.x0).attr("x2", P.x1).attr("stroke", FG).attr("stroke-opacity", 0.35)
        .attr("stroke-width", 2).attr("stroke-dasharray", "8 8");
      const dots = g.selectAll("circle.dot").data(d3.range(n)).join("circle").attr("class", "dot").attr("r", 6.5)
        .attr("cx", (i) => sx(ev.cycle[i])).attr("fill", (i) => COLORS[ev.cond[i]]);
      const trend = d3.line().x((i) => sx(ev.cycle[i])).y((i) => sy[2](ev.trend[i]));
      const trendPath = g.append("path").attr("fill", "none").attr("stroke", "#ffffff").attr("stroke-width", 6)
        .attr("stroke-linecap", "round").attr("stroke-linejoin", "round").attr("d", trend(d3.range(n)));

      const stateLabels = ["x  (valeur brute)", "x − μ  (on retire la moyenne de la condition)",
        "(x − μ) / σ  (on divise par l'écart-type de la condition)"];
      const stateText = stateLabels.map((t) => g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", 200)
        .attr("text-anchor", "middle").attr("fill", FG).attr("font-size", 34).attr("font-weight", 700).text(t));
      const tag = slideTag(g, (P.x0 + P.x1) / 2, 200);

      const RX = 1600;
      const panel = g.append("g").attr("transform", `translate(${RX},250)`);
      panel.append("text").attr("fill", FG).attr("font-size", 36).attr("font-weight", 700).text("z = (x − μ) / σ");
      panel.append("text").attr("y", 40).attr("fill", MUTED).attr("font-size", 22).text("μ et σ de chaque condition");
      panel.append("text").attr("y", 68).attr("fill", MUTED).attr("font-size", 22).text("(calcul de la scène précédente)");
      const rows = panel.selectAll("g.row").data(ev.mu).join("g").attr("class", "row")
        .attr("transform", (d, i) => `translate(0,${130 + i * 90})`);
      rows.append("circle").attr("r", 10).attr("cx", 10).attr("cy", -8).attr("fill", (d, i) => COLORS[i]);
      rows.append("text").attr("x", 32).attr("fill", FG).attr("font-size", 24).text((d, i) => `Condition ${i + 1}`);
      rows.append("text").attr("x", 32).attr("y", 30).attr("fill", MUTED).attr("font-size", 22)
        .text((d, i) => `μ = ${num(ev.mu[i], 1)} · σ = ${num(ev.sigma[i], 2)}`);

      return {
        dur: 18,
        title: "Un moteur : trois états réels",
        subtitle: `Capteur ${ev.demo_sensor} du moteur ${ev.engine} : brut, puis x − μ, puis (x − μ) / σ`,
        refs: [ev],
        captions: [
          [0.3, 3.7, "Valeur brute : elle change de niveau à chaque changement de condition de vol"],
          [4.0, 6.6, "On retire d'abord la moyenne μ de la condition de chaque point"],
          [6.9, 8.4, "Les niveaux sont alignés autour de 0, mais l'échelle reste celle du capteur"],
          [8.6, 11.2, "On divise ensuite par l'écart-type σ de la condition"],
          [11.5, 13.2, "Valeur normalisée : les conditions sont neutralisées"],
          [13.6, 17.7, "La tendance de dégradation (moyenne glissante, trait blanc) devient lisible"],
        ],
        update(u) {
          const s = seg(u, 4.4, 6.6) + seg(u, 9.0, 11.2); // 0 = brut, 1 = x − μ, 2 = normalisé
          const keys = ["raw", "centered", "scaled"];
          const y = (i) => {
            const a = Math.min(1, s), b = Math.max(0, s - 1);
            return s <= 1 ? lerp(sy[0](S.raw[i]), sy[1](S.centered[i]), a) : lerp(sy[1](S.centered[i]), sy[2](S.scaled[i]), b);
          };
          dots.attr("cy", (i) => y(i)).attr("opacity", seg(u, 0.1, 0.8, d3.easeLinear) * 0.9);
          axes.forEach((a, k) => { const o = clamp01(1 - Math.abs(s - k) * 2); a.attr("opacity", o); labels[k].attr("opacity", o); });
          stateText.forEach((t, k) => t.attr("opacity", near(s, k) * seg(u, 0.4, 1.0)));
          tag.attr("opacity", tagOpacity(s));
          // La ligne « 0 » n'a de sens que sur les états x − μ et normalisé : jamais pendant un glissement
          const k0 = s < 1.5 ? 1 : 2;
          zeroLine.attr("y1", sy[k0](0)).attr("y2", sy[k0](0)).attr("opacity", near(s, k0));
          panel.attr("opacity", seg(u, 3.4, 4.2));
          trendPath.attr("opacity", seg(u, 13.8, 14.6));
        },
      };
    }

    // ---------------------------------------------------------------- scène 4 : distributions
    function sceneDistributions(g) {
      const hr = by("histogram").find((e) => e.kind === "raw"), hn = by("histogram").find((e) => e.kind === "scaled");
      const conds = one("conditions").conditions, sensor = hr.sensor;
      const P = { x0: 200, x1: 1700, y0: 240, y1: 850 };
      const centers = (e) => e.slice(0, -1).map((a, i) => (a + e[i + 1]) / 2);
      const cRaw = centers(hr.edges), cNorm = centers(hn.edges);
      const sxRaw = d3.scaleLinear().domain([hr.edges[0], hr.edges.at(-1)]).range([P.x0, P.x1]);
      const sxNorm = d3.scaleLinear().domain([hn.edges[0], hn.edges.at(-1)]).range([P.x0, P.x1]);
      const syRaw = d3.scaleLinear().domain([0, d3.max(hr.density, (a) => d3.max(a))]).range([P.y1, P.y0]);
      const syNorm = d3.scaleLinear().domain([0, d3.max(hn.density, (a) => d3.max(a))]).range([P.y1, P.y0]);

      const axRaw = axisBottom(g, sxRaw, P.y1, 8, (d) => num(d, 0));
      const axNorm = axisBottom(g, sxNorm, P.y1, 9, (d) => num(d, 0));
      const labRaw = g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", P.y1 + 66).attr("fill", MUTED)
        .attr("font-size", 26).attr("text-anchor", "middle").text(`valeur brute de ${sensor}`);
      const labNorm = g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", P.y1 + 66).attr("fill", MUTED)
        .attr("font-size", 26).attr("text-anchor", "middle").text(`valeur normalisée de ${sensor} (écarts-types)`);
      rotLabel(g, P.x0 - 40, (P.y0 + P.y1) / 2, "densité (par condition)");

      const zeroMark = g.append("g");
      zeroMark.append("line").attr("x1", sxNorm(0)).attr("x2", sxNorm(0)).attr("y1", P.y0 + 40).attr("y2", P.y1)
        .attr("stroke", FG).attr("stroke-width", 3).attr("stroke-dasharray", "10 8");
      zeroMark.append("text").attr("x", sxNorm(0)).attr("y", P.y0 + 28).attr("fill", FG).attr("font-size", 28)
        .attr("text-anchor", "middle").text("moyenne = 0");
      const sigmaMark = g.append("g");
      [-1, 1].forEach((sg) => {
        sigmaMark.append("line").attr("x1", sxNorm(sg)).attr("x2", sxNorm(sg)).attr("y1", P.y0 + 40).attr("y2", P.y1)
          .attr("stroke", MUTED).attr("stroke-width", 2).attr("stroke-dasharray", "4 8");
        sigmaMark.append("text").attr("x", sxNorm(sg)).attr("y", P.y0 + 28).attr("fill", MUTED).attr("font-size", 26)
          .attr("text-anchor", "middle").text(sg < 0 ? "−1 σ" : "+1 σ");
      });

      const areas = conds.map((c, k) => g.append("path").attr("fill", COLORS[k]).attr("fill-opacity", 0.32)
        .attr("stroke", COLORS[k]).attr("stroke-width", 4).attr("stroke-linejoin", "round"));
      const area = d3.area().x((p) => p[0]).y0(P.y1).y1((p) => p[1]);
      const stateRaw = g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", 200).attr("text-anchor", "middle")
        .attr("fill", FG).attr("font-size", 34).attr("font-weight", 700).text("Histogrammes des valeurs brutes");
      const stateNorm = g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", 200).attr("text-anchor", "middle")
        .attr("fill", FG).attr("font-size", 34).attr("font-weight", 700).text("Histogrammes des valeurs normalisées");
      const tag = slideTag(g, (P.x0 + P.x1) / 2, 200);

      const leg = g.append("g").attr("transform", "translate(1330,215)");
      leg.selectAll("g").data(conds).join("g").attr("transform", (c, i) => `translate(0,${i * 36})`)
        .each(function (c, i) {
          d3.select(this).append("circle").attr("r", 10).attr("cx", 10).attr("cy", -8).attr("fill", COLORS[i]);
          d3.select(this).append("text").attr("x", 32).attr("fill", FG).attr("font-size", 23).text(`Condition ${i + 1}`);
        });

      return {
        dur: 14,
        title: "Les distributions se recentrent",
        subtitle: `Valeurs de ${sensor} sur les 4 jeux, une courbe par condition de vol`,
        refs: [hr, hn],
        captions: [
          [0.3, 4.2, "Avant : 6 distributions séparées, une par condition de vol"],
          [4.4, 5.6, "Un même état de moteur n'a pas la même valeur selon la condition"],
          [5.8, 9.2, "Chaque condition est ramenée à une moyenne de 0 et un écart-type de 1"],
          [9.8, 13.7, "Après : les 6 distributions se superposent, seule la dégradation les écarte"],
        ],
        update(u) {
          const fade = seg(u, 0.1, 1.0, d3.easeLinear);
          const m = seg(u, 5.8, 9.4); // 0 = brut, 1 = normalisé (réel, pas interpolé : glissement visuel)
          areas.forEach((a, k) => {
            const pts = d3.range(cRaw.length).map((i) => [
              lerp(sxRaw(cRaw[i]), sxNorm(cNorm[i]), m),
              lerp(syRaw(hr.density[k][i]), syNorm(hn.density[k][i]), m),
            ]);
            a.attr("d", area(pts)).attr("opacity", fade);
          });
          const oRaw = clamp01(1 - m * 2), oNorm = clamp01(m * 2 - 1);
          axRaw.attr("opacity", oRaw); labRaw.attr("opacity", oRaw);
          axNorm.attr("opacity", oNorm); labNorm.attr("opacity", oNorm);
          stateRaw.attr("opacity", near(2 * m, 0) * fade); stateNorm.attr("opacity", near(2 * m, 2));
          tag.attr("opacity", tagOpacity(2 * m));
          zeroMark.attr("opacity", seg(u, 9.6, 10.4));
          sigmaMark.attr("opacity", seg(u, 10.4, 11.2));
        },
      };
    }

    // ---------------------------------------------------------------- scène 5 : grille des capteurs
    function sceneGrid(g) {
      const ev = one("engine_states"), fin = one("stats_final");
      const names = meta.sensors_all, n = ev.cycle.length;
      const constant = new Set(ev.constant_sensors);
      const cells = names.map((s) => ({ sensor: s, constant: constant.has(s), ...ev.sensors[s] }));
      const kept = cells.filter((c) => !c.constant);
      const layout21 = (i) => ({ x: 85 + (i % 7) * 250, y: 175 + Math.floor(i / 7) * 245, w: 232, h: 215 });
      const layout17 = (j) => ({ x: 70 + (j % 6) * 290, y: 175 + Math.floor(j / 6) * 245, w: 272, h: 215 });
      const keptIndex = new Map(kept.map((c, j) => [c.sensor, j]));
      const ext = (arr) => { let [a, b] = d3.extent(arr); if (b - a < 1e-9) { a -= 1; b += 1; } const m = (b - a) * 0.1; return [a - m, b + m]; };
      const info = cells.map((c) => ({ rawExt: ext(c.raw), normExt: ext(c.scaled) }));

      // Capteurs conservés mais constants dans certaines conditions (lu dans stats_final, pas deviné)
      const partial = fin.sensors.map((s) => [s, fin.conditions.filter((c) => c.flat.includes(s)).length]).filter(([, k]) => k > 0);
      const partialNote = partial.map(([s, k]) => `${s} reste constant dans ${k} condition${k > 1 ? "s" : ""} sur ${fin.conditions.length}`).join(" ; ");

      const cellG = g.selectAll("g.cell").data(cells).join("g").attr("class", "cell");
      cellG.append("rect").attr("rx", 10).attr("fill", "#141b23").attr("stroke", GRID).attr("stroke-width", 2);
      cellG.append("text").attr("font-size", 26).attr("font-weight", 700).attr("fill", FG).text((c) => c.sensor);
      cellG.append("path").attr("fill", "none").attr("stroke-width", 2.6).attr("stroke-linejoin", "round");
      cellG.append("text").attr("class", "const").attr("fill", WARN).attr("font-size", 24).attr("font-weight", 700)
        .attr("text-anchor", "end").text("constant");
      cellG.append("line").attr("stroke", WARN).attr("stroke-width", 4).attr("stroke-linecap", "round");

      const status = g.append("text").attr("x", W - 80).attr("y", 935).attr("text-anchor", "end").attr("fill", FG)
        .attr("font-size", 34).attr("font-weight", 700);
      const stateL = g.append("text").attr("x", W - 80).attr("y", 100).attr("text-anchor", "end").attr("fill", FG)
        .attr("font-size", 34).attr("font-weight", 700);
      const tag = slideTag(g, W - 460, 150);
      const line = d3.line();

      return {
        dur: 15,
        title: "Tous les capteurs d'un coup",
        subtitle: `Les ${names.length} capteurs du moteur ${ev.engine} : brut, puis normalisé par condition`,
        refs: [ev, fin],
        captions: [
          [0.3, 3.0, "Avant : les courbes sont brouillées par les conditions de vol"],
          [3.2, 6.4, "Normalisation par condition de vol, capteur par capteur : (x − μ) / σ"],
          [6.8, 9.2, `${[...constant].join(", ")} deviennent plats : aucune information utile`],
          [9.5, 12.9, `Il reste ${kept.length} capteurs pour entraîner le modèle`],
          ...(partialNote ? [[13.1, 14.8, partialNote]] : []),
        ],
        update(u) {
          const fade = seg(u, 0.1, 0.9, d3.easeLinear);
          const m = seg(u, 3.4, 6.4);
          const hi = seg(u, 6.8, 7.6);
          const re = seg(u, 9.5, 12.5);
          cellG.each(function (c, i) {
            const L0 = layout21(i), j = keptIndex.get(c.sensor), L1 = j === undefined ? L0 : layout17(j);
            const L = { x: lerp(L0.x, L1.x, re), y: lerp(L0.y, L1.y, re), w: lerp(L0.w, L1.w, re), h: L0.h };
            const e = info[i], padX = 14, top = 48, bot = 14;
            const xs = (t) => L.x + padX + (t / (n - 1)) * (L.w - 2 * padX);
            const ys = (v, ex) => L.y + L.h - bot - ((v - ex[0]) / (ex[1] - ex[0])) * (L.h - top - bot);
            const pts = d3.range(n).map((t) => [xs(t), lerp(ys(c.raw[t], e.rawExt), ys(c.scaled[t], e.normExt), m)]);
            const col = c.constant ? d3.interpolateRgb("#7fb2ff", WARN)(hi) : "#7fb2ff";
            const cg = d3.select(this);
            cg.select("rect").attr("x", L.x).attr("y", L.y).attr("width", L.w).attr("height", L.h)
              .attr("stroke", c.constant ? d3.interpolateRgb(GRID, WARN)(hi) : GRID);
            cg.select("text").attr("x", L.x + 14).attr("y", L.y + 32);
            cg.select("path").attr("d", line(pts)).attr("stroke", col);
            cg.select("text.const").attr("x", L.x + L.w - 14).attr("y", L.y + 32).attr("opacity", c.constant ? hi : 0);
            cg.select("line").attr("x1", L.x + 18).attr("y1", L.y + L.h - 20).attr("x2", L.x + L.w - 18).attr("y2", L.y + 44)
              .attr("opacity", c.constant ? hi * 0.9 : 0);
            cg.attr("opacity", fade * (c.constant ? 1 - clamp01(re * 3) : 1)); // disparaît avant que les voisins arrivent
          });
          stateL.text(m < 0.02 ? "valeurs brutes" : m > 0.98 ? "valeurs normalisées" : "").attr("opacity", fade);
          tag.attr("opacity", tagOpacity(m * 1.0));
          status.text(re > 0.5 ? `${names.length} → ${kept.length} capteurs retenus` : `${names.length} capteurs`)
            .attr("opacity", seg(u, 0.4, 1.2));
        },
      };
    }

    // ================================================================ version simple (grand public)
    // Mêmes données du journal, mais : pas de formule, pas de chiffres, une seule phrase par scène.

    // Mots géants AVANT -> (nom du traitement, visible seulement pendant le glissement) -> APRÈS
    function avantApres(g, x, y, anchor = "middle", size = 60) {
      const mk = (t, fill, sz = size) => g.append("text").attr("x", x).attr("y", y).attr("text-anchor", anchor)
        .attr("fill", fill).attr("font-size", sz).attr("font-weight", 700).text(t);
      return [mk("AVANT", FG), mk("→ traitement : mise à la même échelle", FG, Math.round(size * 0.72)),
        mk("APRÈS", COLORS[3])];
    }
    // Légende : 6 pastilles numérotées = les 6 situations de vol (même couleur dans toutes les scènes)
    function legendSituations(g, x, y) {
      const lg = g.append("g").attr("transform", `translate(${x},${y})`);
      lg.append("text").attr("fill", MUTED).attr("font-size", 26).attr("text-anchor", "end").text("situation de vol :");
      COLORS.forEach((c, k) => {
        lg.append("circle").attr("cx", 28 + k * 62).attr("cy", -9).attr("r", 20).attr("fill", c);
        lg.append("text").attr("x", 28 + k * 62).attr("y", -1).attr("text-anchor", "middle").attr("fill", BG)
          .attr("font-size", 24).attr("font-weight", 700).text(k + 1);
      });
      return lg;
    }

    function simpleConditions(g) {
      const condEv = one("conditions"), conds = condEv.conditions, samp = one("settings_sample");
      const P = { x0: 340, x1: 1580, y0: 190, y1: 830 };
      const sx = d3.scaleLinear().domain([-4, 46]).range([P.x0, P.x1]);
      const sy = d3.scaleLinear().domain([-0.06, 0.92]).range([P.y1, P.y0]);
      baseline(g, P.x0 - 20, P.x1, P.y1 + 20);
      g.append("line").attr("x1", P.x0 - 20).attr("x2", P.x0 - 20).attr("y1", P.y0).attr("y2", P.y1 + 20)
        .attr("stroke", MUTED).attr("stroke-width", 3);
      g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", P.y1 + 72).attr("fill", MUTED).attr("font-size", 32)
        .attr("text-anchor", "middle").text("altitude  →");
      rotLabel(g, P.x0 - 80, (P.y0 + P.y1) / 2, "vitesse  →").attr("font-size", 32);

      const rings = g.selectAll("circle.ring").data(conds).join("circle").attr("class", "ring")
        .attr("cx", (c) => sx(c.altitude)).attr("cy", (c) => sy(c.mach))
        .attr("fill", "none").attr("stroke-width", 5).attr("stroke", (c, i) => COLORS[i]);
      const dots = g.selectAll("circle.pt").data(samp.points).join("circle").attr("class", "pt")
        .attr("cx", (d) => sx(d[0])).attr("cy", (d) => sy(d[1])).attr("r", 6);
      const tags = g.selectAll("text.tag").data(conds).join("text").attr("class", "tag")
        .attr("x", (c) => sx(c.altitude)).attr("y", (c) => sy(c.mach) - 56)
        .attr("text-anchor", "middle").attr("font-size", 46).attr("font-weight", 700)
        .attr("fill", (c, i) => COLORS[i]).text((c, i) => i + 1);

      return {
        dur: 10, simple: true, refs: [condEv, samp],
        title: "Des vols dans 6 situations différentes", subtitle: "",
        captions: [[0.6, 9.6, "Un moteur vole dans 6 situations différentes."]],
        update(u) {
          const col = seg(u, 2.0, 3.4);
          dots.attr("opacity", 0.6 * seg(u, 0.2, 1.4, d3.easeLinear))
            .attr("fill", (d) => (col <= 0 ? "#9aa7b5" : d3.interpolateRgb("#9aa7b5", COLORS[d[3]])(col)));
          const ring = seg(u, 3.4, 4.6);
          rings.attr("r", 40 * ring).attr("opacity", ring);
          tags.attr("opacity", ring);
        },
      };
    }

    function simpleEngine(g) {
      const ev = one("engine_states"), S = ev.sensors[ev.demo_sensor], n = ev.cycle.length;
      const P = { x0: 240, x1: 1680, y0: 270, y1: 840 };
      const sx = d3.scaleLinear().domain(d3.extent(ev.cycle)).range([P.x0 + 20, P.x1 - 20]);
      const padded = (arr) => { const [a, b] = d3.extent(arr), m = (b - a) * 0.08; return [a - m, b + m]; };
      const syRaw = d3.scaleLinear().domain(padded(S.raw)).range([P.y1, P.y0]);
      const syNew = d3.scaleLinear().domain(padded(S.scaled)).range([P.y1, P.y0]);
      baseline(g, P.x0, P.x1, P.y1 + 20);
      g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", P.y1 + 72).attr("fill", MUTED).attr("font-size", 32)
        .attr("text-anchor", "middle").text("temps  →");
      rotLabel(g, P.x0 - 40, (P.y0 + P.y1) / 2, "mesure d'un capteur").attr("font-size", 32);

      const dots = g.selectAll("circle.dot").data(d3.range(n)).join("circle").attr("class", "dot").attr("r", 9)
        .attr("cx", (i) => sx(ev.cycle[i])).attr("fill", (i) => COLORS[ev.cond[i]]);
      // Droite d'approximation (moindres carrés) calculée par normalise.py et lue dans le journal
      const F = ev.fit, yFit = (x) => F.slope * x + F.intercept;
      const trendPath = g.append("line").attr("stroke", "#ffffff").attr("stroke-width", 9).attr("stroke-linecap", "round")
        .attr("x1", sx(F.x0)).attr("y1", syNew(yFit(F.x0))).attr("x2", sx(F.x0)).attr("y2", syNew(yFit(F.x0)));
      const trendLabel = g.append("text").attr("x", P.x1 - 10).attr("y", P.y0 - 8)
        .attr("text-anchor", "end").attr("fill", "#ffffff").attr("font-size", 44).attr("font-weight", 700)
        .text("la droite blanche = la tendance d'usure");
      const words = avantApres(g, W / 2, 205);
      legendSituations(g, W - 80 - 358, 130);

      return {
        dur: 14, simple: true, refs: [ev],
        title: "Voir enfin l'usure du moteur", subtitle: "",
        captions: [[0.6, 13.4, "On retire l'effet de la situation de vol pour voir l'usure."]],
        update(u) {
          const fade = seg(u, 0.1, 1.0, d3.easeLinear);
          const m = seg(u, 4.5, 8.0);
          dots.attr("cy", (i) => lerp(syRaw(S.raw[i]), syNew(S.scaled[i]), m)).attr("opacity", fade * 0.92);
          words[0].attr("opacity", near(2 * m, 0) * fade);
          words[1].attr("opacity", slideOp(m));
          words[2].attr("opacity", near(2 * m, 2));
          const tr = seg(u, 9.0, 10.8);
          const xe = lerp(F.x0, F.x1, tr); // la droite se trace de gauche à droite
          trendPath.attr("x2", sx(xe)).attr("y2", syNew(yFit(xe))).attr("opacity", tr > 0 ? 1 : 0);
          trendLabel.attr("opacity", seg(u, 10.2, 11.0));
        },
      };
    }

    function simpleDistributions(g) {
      const hr = by("histogram").find((e) => e.kind === "raw"), hn = by("histogram").find((e) => e.kind === "scaled");
      const conds = one("conditions").conditions;
      const P = { x0: 240, x1: 1680, y0: 290, y1: 830 };
      const centers = (e) => e.slice(0, -1).map((a, i) => (a + e[i + 1]) / 2);
      const cRaw = centers(hr.edges), cNorm = centers(hn.edges);
      const sxRaw = d3.scaleLinear().domain([hr.edges[0], hr.edges.at(-1)]).range([P.x0, P.x1]);
      const sxNew = d3.scaleLinear().domain([hn.edges[0], hn.edges.at(-1)]).range([P.x0, P.x1]);
      const syRaw = d3.scaleLinear().domain([0, d3.max(hr.density, (a) => d3.max(a))]).range([P.y1, P.y0]);
      const syNew = d3.scaleLinear().domain([0, d3.max(hn.density, (a) => d3.max(a))]).range([P.y1, P.y0]);
      baseline(g, P.x0, P.x1, P.y1);
      g.append("text").attr("x", (P.x0 + P.x1) / 2).attr("y", P.y1 + 62).attr("fill", MUTED).attr("font-size", 32)
        .attr("text-anchor", "middle").text("valeur mesurée  →");
      const areas = conds.map((c, k) => g.append("path").attr("fill", COLORS[k]).attr("fill-opacity", 0.35)
        .attr("stroke", COLORS[k]).attr("stroke-width", 5).attr("stroke-linejoin", "round"));
      const area = d3.area().x((p) => p[0]).y0(P.y1).y1((p) => p[1]);
      const words = avantApres(g, W / 2, 205);
      legendSituations(g, W - 80 - 358, 130);

      return {
        dur: 12, simple: true, refs: [hr, hn],
        title: "Mettre toutes les situations sur la même échelle", subtitle: "",
        captions: [[0.6, 11.6, "Les 6 situations se superposent : les mesures deviennent comparables."]],
        update(u) {
          const fade = seg(u, 0.1, 1.0, d3.easeLinear);
          const m = seg(u, 4.5, 8.5);
          areas.forEach((a, k) => {
            const pts = d3.range(cRaw.length).map((i) => [
              lerp(sxRaw(cRaw[i]), sxNew(cNorm[i]), m),
              lerp(syRaw(hr.density[k][i]), syNew(hn.density[k][i]), m),
            ]);
            a.attr("d", area(pts)).attr("opacity", fade);
          });
          words[0].attr("opacity", near(2 * m, 0) * fade);
          words[1].attr("opacity", slideOp(m));
          words[2].attr("opacity", near(2 * m, 2));
        },
      };
    }

    function simpleGrid(g) {
      const ev = one("engine_states"), names = meta.sensors_all, n = ev.cycle.length;
      const constant = new Set(ev.constant_sensors);
      const cells = names.map((s) => ({ sensor: s, constant: constant.has(s), ...ev.sensors[s] }));
      const kept = cells.filter((c) => !c.constant);
      const layout21 = (i) => ({ x: 85 + (i % 7) * 250, y: 190 + Math.floor(i / 7) * 245, w: 232, h: 215 });
      const layout17 = (j) => ({ x: 70 + (j % 6) * 290, y: 190 + Math.floor(j / 6) * 245, w: 272, h: 215 });
      const keptIndex = new Map(kept.map((c, j) => [c.sensor, j]));
      const ext = (arr) => { let [a, b] = d3.extent(arr); if (b - a < 1e-9) { a -= 1; b += 1; } const m = (b - a) * 0.1; return [a - m, b + m]; };
      const info = cells.map((c) => ({ rawExt: ext(c.raw), normExt: ext(c.scaled) }));

      const cellG = g.selectAll("g.cell").data(cells).join("g").attr("class", "cell");
      cellG.append("rect").attr("rx", 10).attr("fill", "#141b23").attr("stroke", GRID).attr("stroke-width", 2);
      cellG.append("path").attr("fill", "none").attr("stroke-width", 3).attr("stroke-linejoin", "round");
      cellG.append("text").attr("class", "const").attr("fill", WARN).attr("font-size", 34).attr("font-weight", 700)
        .attr("text-anchor", "middle").text("inutile");
      cellG.append("line").attr("stroke", WARN).attr("stroke-width", 5).attr("stroke-linecap", "round");
      const words = avantApres(g, W - 80, 118, "end");
      const line = d3.line();

      return {
        dur: 13, simple: true, refs: [ev],
        title: "Garder les capteurs utiles", subtitle: "",
        captions: [
          [0.6, 3.2, "Avant : tous les capteurs semblent varier, à cause des situations de vol."],
          [3.5, 6.6, "Même traitement pour chaque capteur : on retire l'effet de la situation de vol."],
          [6.9, 9.2, "Après : certains capteurs restent plats, ils n'apportent aucune information."],
          [9.5, 12.6, "On les retire : il reste les capteurs utiles."],
        ],
        update(u) {
          const fade = seg(u, 0.1, 0.9, d3.easeLinear);
          const m = seg(u, 3.4, 6.4);
          const hi = seg(u, 6.8, 7.8);
          const re = seg(u, 9.4, 12.2);
          cellG.each(function (c, i) {
            const L0 = layout21(i), j = keptIndex.get(c.sensor), L1 = j === undefined ? L0 : layout17(j);
            const L = { x: lerp(L0.x, L1.x, re), y: lerp(L0.y, L1.y, re), w: lerp(L0.w, L1.w, re), h: L0.h };
            const e = info[i], padX = 14, top = 16, bot = 16;
            const xs = (t) => L.x + padX + (t / (n - 1)) * (L.w - 2 * padX);
            const ys = (v, ex) => L.y + L.h - bot - ((v - ex[0]) / (ex[1] - ex[0])) * (L.h - top - bot);
            const pts = d3.range(n).map((t) => [xs(t), lerp(ys(c.raw[t], e.rawExt), ys(c.scaled[t], e.normExt), m)]);
            const cg = d3.select(this);
            cg.select("rect").attr("x", L.x).attr("y", L.y).attr("width", L.w).attr("height", L.h)
              .attr("stroke", c.constant ? d3.interpolateRgb(GRID, WARN)(hi) : GRID);
            cg.select("path").attr("d", line(pts)).attr("stroke", c.constant ? d3.interpolateRgb("#7fb2ff", WARN)(hi) : "#7fb2ff");
            cg.select("text.const").attr("x", L.x + L.w / 2).attr("y", L.y + 44).attr("opacity", c.constant ? hi : 0)
              .attr("paint-order", "stroke").attr("stroke", "#141b23").attr("stroke-width", 8);
            cg.select("line").attr("x1", L.x + 18).attr("y1", L.y + L.h - 20).attr("x2", L.x + L.w - 18).attr("y2", L.y + 60)
              .attr("opacity", c.constant ? hi * 0.9 : 0);
            cg.attr("opacity", fade * (c.constant ? 1 - clamp01(re * 3) : 1));
          });
          words[0].attr("opacity", near(2 * m, 0) * fade);
          words[1].attr("opacity", slideOp(m));
          words[2].attr("opacity", near(2 * m, 2));
        },
      };
    }


    return {
      tech: [sceneConditions, sceneStats, sceneEngine, sceneDistributions, sceneGrid],
      simple: [simpleConditions, simpleEngine, simpleDistributions, simpleGrid],
    };
  },
});
