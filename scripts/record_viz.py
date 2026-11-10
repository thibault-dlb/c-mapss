"""Produit la vidéo de la normalisation en rejouant le journal, image par image (reproductible).

Usage : .\\.venv\\Scripts\\python.exe scripts\\record_viz.py [--fps 30] [--out reports\\videos\\normalisation.mp4]

La page (?render=1) expose window.renderAt(t) : chaque image est le dessin exact de l'instant t,
construit uniquement à partir de viz/normalisation/journal.jsonl. Deux exécutions sur le même journal
donnent la même vidéo. Nécessite Playwright + Chromium (voir requirements.txt) et ffmpeg dans le PATH.
"""
import argparse
import functools
import http.server
import shutil
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
VIZ = ROOT / "viz"
W, H = 1920, 1080


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mode", choices=["simple", "tech"], default="simple")
    ap.add_argument("--out", default=None, help="fichier de sortie (défaut : selon le mode)")
    ap.add_argument("--page", default="normalisation/index.html")
    ap.add_argument("--crf", type=int, default=18, help="qualité x264 (plus bas = meilleur)")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg introuvable dans le PATH.")
    if not (VIZ / "normalisation" / "journal.jsonl").exists():
        sys.exit("viz/normalisation/journal.jsonl absent : lancer scripts/normalise.py --journal")
    page_dir = Path(args.page).parent.name or "video"   # normalisation, fenetres...
    name = f"{page_dir}_{'simple' if args.mode == 'simple' else 'technique'}.mp4"
    out = Path(args.out) if args.out else ROOT / "reports" / "videos" / name
    out.parent.mkdir(parents=True, exist_ok=True)

    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(VIZ)))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    ffmpeg = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": W, "height": H})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}/{args.page}?render=1&mode={args.mode}")
            page.wait_for_function("window.READY === true || window.ERROR", timeout=30000)
            err = page.evaluate("window.ERROR")
            if err:
                sys.exit(f"La page signale une erreur : {err}")
            duration = page.evaluate("window.DURATION")
            n = int(round(duration * args.fps)) + 1
            print(f"Durée {duration:.1f} s · {args.fps} i/s · {n} images -> {out}")

            ffmpeg = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(args.fps),
                 "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                stdin=subprocess.PIPE)
            t0 = time.time()
            for i in range(n):
                page.evaluate(f"window.renderAt({min(duration, i / args.fps)})")
                ffmpeg.stdin.write(page.screenshot(type="png"))
                if i % 100 == 0:
                    print(f"  image {i}/{n} ({time.time() - t0:.0f} s)", flush=True)
            ffmpeg.stdin.close()
            if ffmpeg.wait() != 0:
                sys.exit("ffmpeg a échoué.")
            browser.close()
            if errors:
                print("Erreurs JavaScript pendant le rendu :", *errors, sep="\n  ")
                sys.exit(1)
    finally:
        httpd.shutdown()
        if ffmpeg and ffmpeg.poll() is None:
            ffmpeg.kill()
    print(f"Vidéo écrite : {out} ({out.stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    main()
