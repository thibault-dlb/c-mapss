// Enregistrement de l'onglet (capture réelle, Chrome ou Edge) pour les pages en direct.
// Même principe que le bouton de viz/lib/core.js, en plus court : compte à rebours, capture à 30 i/s, téléchargement.
// Non testé automatiquement (action humaine : choisir l'onglet dans la boîte de dialogue du navigateur).
//
// Usage : const rec = VizRecord.attach({ button, countdownEl, msgEl, name, onLayout });
//         rec.stopSoon() arrête l'enregistrement après un court délai (ex. à la fin de l'entraînement).
(function () {
  "use strict";
  window.VizRecord = { attach };

  function attach({ button, countdownEl, msgEl, name = "entrainement", onLayout = () => {} }) {
    let recorder = null, chunks = [], mime = "";
    const pickMime = () => ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm"]
      .find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || "";
    const showMsg = (html) => { msgEl.innerHTML = html; msgEl.hidden = false; };

    async function start() {
      msgEl.hidden = true;
      if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
        showMsg("Ce navigateur ne sait pas capturer l'onglet. Utiliser Chrome ou Edge.");
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
      document.body.classList.add("recording");
      onLayout();
      countdownEl.hidden = false;
      for (const k of [3, 2, 1]) { countdownEl.textContent = k; await new Promise((r) => setTimeout(r, 1000)); }
      countdownEl.hidden = true;
      await new Promise((r) => setTimeout(r, 150));
      chunks = [];
      recorder = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 12_000_000 } : undefined);
      recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      recorder.onstop = () => { stream.getTracks().forEach((t) => t.stop()); finish(); };
      recorder.start(500);
    }
    function finish() {
      document.body.classList.remove("recording");
      onLayout();
      const type = recorder.mimeType || mime || "video/webm";
      const ext = type.includes("mp4") ? "mp4" : "webm";
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob(chunks, { type }));
      a.download = `${name}.${ext}`;
      document.body.appendChild(a); a.click(); a.remove();
      let html = `Vidéo téléchargée : <code>${name}.${ext}</code>.`;
      if (ext === "webm") html += `<br>Conversion en MP4 : <code>ffmpeg -i ${name}.webm -c:v libx264 -pix_fmt yuv420p ${name}.mp4</code>`;
      showMsg(html);
    }
    const stop = () => { if (recorder && recorder.state === "recording") recorder.stop(); };
    button.onclick = () => { if (!recorder || recorder.state !== "recording") start(); };
    addEventListener("keydown", (e) => { if (e.key === "Escape") stop(); });
    return { stopSoon: (ms = 3000) => { if (recorder && recorder.state === "recording") setTimeout(stop, ms); } };
  }
})();
