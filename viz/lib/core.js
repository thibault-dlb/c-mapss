// Cœur commun des pages de visualisation (D3) : rejeu d'un journal d'événements, chronologie, contrôles, enregistrement.
// La page ne calcule AUCUNE statistique : elle rejoue les événements du journal, dans l'ordre, avec les valeurs exactes
// écrites par les scripts Python. Seul le rythme d'affichage est choisi ici.
// Tout est piloté par le temps : render(t) dessine l'image exacte à l'instant t (en secondes), ce qui permet une
// lecture fluide ET une capture image par image identique (?render=1).
//
// Usage : VizCore.start({ journal, hint, noteFinale, scenes(ctx) -> { tech: [...], simple: [...] } })
(function () {
  "use strict";
  window.VizCore = { start };

  async function start(opts) {
    const W = 1920, H = 1080;
    const COLORS = ["#4cc9f0", "#f72585", "#ffd60a", "#80ed99", "#b388ff", "#ff9f1c"];
    const FG = "#e8eef5", MUTED = "#8a98a8", BG = "#0f1419", GRID = "#26303b", WARN = "#ff6b81", AMBER = "#ffb347";
    const FONT = "'Segoe UI', system-ui, sans-serif";
    const CARD = 2.5; // durée du carton titre avant chaque scène (s)

    const clamp01 = (x) => Math.max(0, Math.min(1, x));
    const ease = d3.easeCubicInOut;
    const seg = (u, a, b, e = ease) => e(clamp01((u - a) / (b - a))); // progression adoucie entre a et b
    const lerp = (a, b, p) => a + (b - a) * p;
    const num = (x, d = 2) => String(+x.toFixed(d)).replace(".", ",").replace("-", "−");
    const sci = (x) => x.toExponential(1).replace(".", ",").replace("e-", "e−").replace("e+", "e");
    const int = (x) => Math.round(x).toLocaleString("fr-FR");

    const params = new URLSearchParams(location.search);
    const RENDER = params.get("render") === "1";
    if (RENDER) document.body.classList.add("render");
    // « simple » (défaut) : vidéo grand public. « tech » : version détaillée (calcul bloc par bloc, μ, σ, traçabilité).
    const MODE = params.get("mode") === "tech" ? "tech" : "simple";

    // ---------------------------------------------------------------- chargement et contrôle du journal
    const stageEl = document.getElementById("stage");
    let events;
    try {
      const res = await fetch(opts.journal, { cache: "no-store" });
      if (!res.ok) throw new Error(`${opts.journal} introuvable (HTTP ${res.status}). ${opts.hint || ""}`);
      events = (await res.text()).trim().split("\n").map((l) => JSON.parse(l));
      events.forEach((e, i) => { if (e.seq !== i) throw new Error(`journal non contigu : événement n°${i} porte seq=${e.seq}`); });
      if (events.at(-1).type !== "end") throw new Error("journal incomplet : pas d'événement « end »");
    } catch (err) {
      stageEl.innerHTML = `<div style="padding:120px;font:40px ${FONT};color:${WARN}">Journal illisible : ${err.message}</div>`;
      window.ERROR = String(err.message);
      return;
    }
    const by = (type) => events.filter((e) => e.type === type);
    const one = (type) => by(type)[0];
    const head = one("journal"), meta = one("meta");

    // ---------------------------------------------------------------- scène de base
    const svg = d3.select("#stage").append("svg")
      .attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`)
      .style("font-family", FONT);
    svg.append("rect").attr("width", W).attr("height", H).attr("fill", BG);

    const header = svg.append("g");
    const headTitle = header.append("text").attr("x", 80).attr("y", 88).attr("fill", FG)
      .attr("font-size", 52).attr("font-weight", 700);
    const headSub = header.append("text").attr("x", 80).attr("y", 134).attr("fill", MUTED).attr("font-size", 30);

    const captionEl = svg.append("text").attr("x", W / 2).attr("y", 975).attr("fill", FG)
      .attr("font-size", 36).attr("text-anchor", "middle");
    const footerEl = svg.append("text").attr("x", 80).attr("y", 1026).attr("fill", MUTED).attr("font-size", 19);

    const cardG = svg.append("g");
    const cardNum = cardG.append("text").attr("x", W / 2).attr("y", 430).attr("fill", COLORS[0])
      .attr("font-size", 150).attr("font-weight", 700).attr("text-anchor", "middle");
    const cardTitle = cardG.append("text").attr("x", W / 2).attr("y", 560).attr("fill", FG)
      .attr("font-size", 78).attr("font-weight", 700).attr("text-anchor", "middle");
    const cardSub = cardG.append("text").attr("x", W / 2).attr("y", 640).attr("fill", MUTED)
      .attr("font-size", 36).attr("text-anchor", "middle");

    const progG = svg.append("g").attr("transform", "translate(80,1044)");

    function styleAxis(g) {
      g.selectAll("text").attr("fill", MUTED).attr("font-size", 24).attr("font-family", FONT);
      g.selectAll("path,line").attr("stroke", MUTED);
      return g;
    }
    function axisBottom(g, scale, y, ticks = 6, fmt = (d) => num(d, 2)) {
      return styleAxis(g.append("g").attr("transform", `translate(0,${y})`)
        .call(d3.axisBottom(scale).ticks(ticks).tickFormat(fmt)));
    }
    function axisLeft(g, scale, x, ticks = 6, fmt = (d) => num(d, 2)) {
      return styleAxis(g.append("g").attr("transform", `translate(${x},0)`)
        .call(d3.axisLeft(scale).ticks(ticks).tickFormat(fmt)));
    }
    const rotLabel = (g, x, y, text) => g.append("text")
      .attr("transform", `translate(${x},${y}) rotate(-90)`).attr("fill", MUTED).attr("font-size", 26)
      .attr("text-anchor", "middle").text(text);
    // Étiquette ambre affichée pendant tout glissement : ce n'est pas un calcul, juste de l'animation
    const slideTag = (g, x, y) => g.append("text").attr("x", x).attr("y", y).attr("fill", AMBER)
      .attr("font-size", 30).attr("font-weight", 700).attr("text-anchor", "middle")
      .text("↔ transition visuelle (pas un calcul)");
    const tagOpacity = (s) => { // s = avancement 0..2 entre 3 états réels ; visible seulement entre deux états
      const m = s >= 2 ? 0 : s - Math.floor(s);
      return clamp01(Math.min(m, 1 - m) * 8);
    };
    const near = (s, k) => clamp01(1 - Math.abs(s - k) * 4); // vaut 1 quand on est exactement sur l'état k

    const seqRange = (evs) => `#${d3.min(evs, (e) => e.seq)}–#${d3.max(evs, (e) => e.seq)}`;

    // Aides communes aux scènes
    const slideOp = (m) => clamp01(Math.min(m, 1 - m) * 8); // visible seulement entre deux états (m : 0 -> 1)
    const baseline = (g, x0, x1, y) => g.append("line").attr("x1", x0).attr("x2", x1).attr("y1", y).attr("y2", y)
      .attr("stroke", MUTED).attr("stroke-width", 3);

    // Tout ce que les scènes peuvent utiliser (voir viz/normalisation/app.js et viz/fenetres/app.js)
    const ctx = {
      d3, W, H, COLORS, FG, MUTED, BG, GRID, WARN, AMBER, FONT, CARD, clamp01, ease, seg, lerp, num, sci, int,
      params, RENDER, MODE, events, by, one, head, meta, svg, styleAxis, axisBottom, axisLeft, rotLabel,
      slideTag, tagOpacity, near, seqRange, slideOp, baseline,
    };
    const MAKERS = opts.scenes(ctx)[MODE];

    // ---------------------------------------------------------------- chronologie
    const scenes = [];
    MAKERS.forEach((make, i) => {
      const g = svg.append("g").attr("display", "none");
      const s = make(g);
      s.g = g; s.index = i;
      s.footer = `Rejeu du ${opts.journal} · événements ${seqRange(s.refs)} sur ${events.length} · calculé le ` +
        `${head.started_at.replace("T", " à ")}${head.git ? " · commit " + head.git : ""} · ` +
        "l'ordre et les valeurs sont ceux du calcul, le rythme est choisi pour la vidéo";
      scenes.push(s);
    });
    const starts = [];
    let DURATION = 0;
    scenes.forEach((s) => { starts.push(DURATION); DURATION += CARD + s.dur; });
    window.DURATION = DURATION;

    scenes.forEach((s, i) => {
      const w = ((CARD + s.dur) / DURATION) * (W - 160), x = (starts[i] / DURATION) * (W - 160);
      progG.append("rect").attr("x", x + 2).attr("width", w - 4).attr("height", 8).attr("rx", 4).attr("fill", GRID);
      s.bar = progG.append("rect").attr("x", x + 2).attr("height", 8).attr("rx", 4).attr("fill", COLORS[0]);
      s.barW = w - 4;
    });

    function locate(t) {
      for (let i = 0; i < scenes.length; i++) {
        const total = CARD + scenes[i].dur;
        if (t < starts[i] + total || i === scenes.length - 1) return { i, u: Math.min(t - starts[i], total) };
      }
    }
    const captionOpacity = (u, a, b) => clamp01(Math.min((u - a) / 0.35, (b - u) / 0.35));

    function render(t) {
      t = Math.max(0, Math.min(DURATION, t));
      const { i, u } = locate(t);
      scenes.forEach((s, k) => s.g.attr("display", k === i ? null : "none"));
      const s = scenes[i];

      const cardOp = clamp01(Math.min(u / 0.5, (CARD - u) / 0.5));
      cardG.attr("opacity", cardOp).attr("display", u < CARD ? null : "none");
      if (u < CARD) {
        cardNum.text(`${i + 1} / ${scenes.length}`);
        cardTitle.text(s.title);
        cardSub.text(MODE === "tech" ? s.subtitle : "");
      }
      captionEl.attr("font-size", MODE === "tech" ? 36 : 44);
      const ub = Math.max(0, u - CARD);
      const bodyOp = clamp01((u - CARD + 0.2) / 0.4);
      s.g.attr("opacity", bodyOp);
      header.attr("opacity", bodyOp);
      if (MODE === "tech") {
        footerEl.attr("opacity", bodyOp).text(s.footer);
      } else { // version simple : une seule mention discrète, à la fin de la dernière scène
        const lastScene = i === scenes.length - 1;
        footerEl.attr("font-size", 26).text(lastScene ? opts.noteFinale : "").attr("opacity", lastScene ? seg(ub, s.dur - 5, s.dur - 4) : 0);
      }
      headTitle.text(MODE === "tech" ? `${i + 1}. ${s.title}` : s.title);
      headSub.text(MODE === "tech" ? s.subtitle : "");
      s.update(ub);

      let cap = "", capOp = 0;
      for (const [a, b, text] of s.captions) {
        if (ub >= a && ub <= b) { cap = text; capOp = captionOpacity(ub, a, b); }
      }
      captionEl.text(cap).attr("opacity", capOp);
      scenes.forEach((sc, k) => {
        const filled = k < i ? 1 : k === i ? clamp01(u / (CARD + sc.dur)) : 0;
        sc.bar.attr("width", sc.barW * filled);
      });
      syncUI(t);
    }
    window.renderAt = render;

    // ---------------------------------------------------------------- lecture et interface
    const els = {
      play: document.getElementById("play"), prev: document.getElementById("prev"),
      next: document.getElementById("next"), scrub: document.getElementById("scrub"),
      time: document.getElementById("time"), speed: document.getElementById("speed"),
      rec: document.getElementById("rec"), msg: document.getElementById("msg"),
      countdown: document.getElementById("countdown"),
    };
    const state = { t: 0, playing: false, speed: 1, last: 0, recording: false };
    const fmtT = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

    function syncUI(t) {
      els.scrub.value = Math.round((t / DURATION) * 1000);
      els.time.textContent = `${fmtT(t)} / ${fmtT(DURATION)}`;
      els.play.textContent = state.playing ? "⏸" : "▶";
    }

    function fit() {
      const reserve = (RENDER || state.recording) ? 0 : 64;
      const s = RENDER ? 1 : Math.min(innerWidth / W, (innerHeight - reserve) / H);
      const ox = RENDER ? 0 : (innerWidth - W * s) / 2;
      const oy = RENDER ? 0 : (innerHeight - reserve - H * s) / 2;
      stageEl.style.transform = `translate(${ox}px,${oy}px) scale(${s})`;
    }
    addEventListener("resize", fit);

    function tick(now) {
      if (state.playing) {
        const dt = (now - state.last) / 1000;
        state.last = now;
        state.t += dt * state.speed;
        if (state.t >= DURATION) { state.t = DURATION; state.playing = false; onEnd(); }
        render(state.t);
      }
      requestAnimationFrame(tick);
    }
    function play() { if (state.t >= DURATION) state.t = 0; state.playing = true; state.last = performance.now(); }
    function pause() { state.playing = false; render(state.t); }
    function seek(t) { state.t = t; render(t); }
    function goScene(delta) {
      const target = Math.max(0, Math.min(scenes.length - 1, locate(state.t).i + delta));
      seek(starts[target]);
    }

    els.play.onclick = () => (state.playing ? pause() : play());
    els.prev.onclick = () => goScene(-1);
    els.next.onclick = () => goScene(1);
    els.scrub.oninput = () => seek((els.scrub.value / 1000) * DURATION);
    els.speed.onchange = () => { state.speed = +els.speed.value; };
    addEventListener("keydown", (e) => {
      if (e.target.tagName === "SELECT") return;
      if (e.code === "Space") { e.preventDefault(); els.play.onclick(); }
      else if (e.code === "ArrowRight") goScene(1);
      else if (e.code === "ArrowLeft") goScene(-1);
    });

    // ---------------------------------------------------------------- enregistrement (capture réelle de l'onglet)
    let recorder = null, chunks = [], mime = "";
    const pickMime = () => ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm"]
      .find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || "";
    function showMsg(html) { els.msg.innerHTML = html; els.msg.hidden = false; }

    async function startRecording() {
      els.msg.hidden = true;
      if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
        showMsg("Ce navigateur ne sait pas capturer l'onglet. Utiliser Chrome ou Edge, ou <code>python scripts/record_viz.py</code>.");
        return;
      }
      mime = pickMime();
      let stream;
      try {
        stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 30 }, audio: false, preferCurrentTab: true });
      } catch (err) {
        showMsg("Capture refusée ou annulée. Cliquer sur « Enregistrer » puis choisir cet onglet.");
        return;
      }
      pause();
      state.recording = true;
      document.body.classList.add("recording");
      fit();
      seek(0);
      els.countdown.hidden = false; // compte à rebours non enregistré : le recorder démarre après
      for (const k of [3, 2, 1]) { els.countdown.textContent = k; await new Promise((r) => setTimeout(r, 1000)); }
      els.countdown.hidden = true;
      await new Promise((r) => setTimeout(r, 150));

      chunks = [];
      recorder = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 12_000_000 } : undefined);
      recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      recorder.onstop = () => { stream.getTracks().forEach((tr) => tr.stop()); finishRecording(); };
      recorder.start(500);
      play();
    }
    function onEnd() {
      if (state.recording && recorder && recorder.state === "recording") setTimeout(() => recorder.stop(), 1200);
    }
    function finishRecording() {
      state.recording = false;
      document.body.classList.remove("recording");
      fit();
      const type = recorder.mimeType || mime || "video/webm";
      const ext = type.includes("mp4") ? "mp4" : "webm";
      const url = URL.createObjectURL(new Blob(chunks, { type }));
      const a = document.createElement("a");
      a.href = url; a.download = `normalisation.${ext}`;
      document.body.appendChild(a); a.click(); a.remove();
      let html = `Vidéo téléchargée : <code>normalisation.${ext}</code>.`;
      if (ext === "webm") html += `<br>Conversion en MP4 : <code>ffmpeg -i normalisation.webm -c:v libx264 -pix_fmt yuv420p normalisation.mp4</code>`;
      showMsg(html);
    }
    els.rec.onclick = () => { if (!state.recording) startRecording(); };
    addEventListener("keydown", (e) => {
      if (e.key === "Escape" && state.recording && recorder && recorder.state === "recording") recorder.stop();
    });

    // ---------------------------------------------------------------- démarrage
    fit();
    render(0);
    if (!RENDER) {
      requestAnimationFrame(tick);
      if (params.get("autoplay") === "1") play();
    }
    window.READY = true;
  }
})();
