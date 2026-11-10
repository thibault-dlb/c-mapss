"""Sert viz/ en local et ouvre la page de normalisation dans le navigateur.

Usage : .\\.venv\\Scripts\\python.exe scripts\\serve_viz.py [--port 8765] [--no-browser]

La page charge journal.jsonl avec fetch() : il faut donc un serveur (pas d'ouverture par double-clic).
localhost est aussi un contexte sécurisé, requis pour l'enregistrement de l'onglet (getDisplayMedia).
"""
import argparse
import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path

VIZ = Path(__file__).resolve().parent.parent / "viz"


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # toujours relire le journal le plus récent
        super().end_headers()

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--path", default="normalisation/index.html")
    args = ap.parse_args()

    if not (VIZ / "normalisation" / "journal.jsonl").exists():
        print("Attention : viz/normalisation/journal.jsonl est absent. Lancer d'abord :")
        print("  .\\.venv\\Scripts\\python.exe scripts\\normalise.py --journal")

    handler = functools.partial(Handler, directory=str(VIZ))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://localhost:{args.port}/{args.path}"
        print(f"Page : {url}\nCtrl+C pour arrêter.")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nArrêt.")


if __name__ == "__main__":
    main()
