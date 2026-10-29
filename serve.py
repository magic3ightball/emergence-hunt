# Static dev server with caching disabled (ES modules were served stale by heuristic caching).
import http.server, sys

class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8980
http.server.ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
