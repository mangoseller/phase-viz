from contextlib import contextmanager
import sys
import numpy as np
import threading
import http
import socketserver

@contextmanager
def suppress_stdout_stderr():
    """
    Context manager to suppress stdout and stderr.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    class NullIO:
        def write(self, *args, **kwargs):
            pass
        def flush(self, *args, **kwargs):
            pass
    
    try:
        sys.stdout = NullIO()
        sys.stderr = NullIO()
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr



def silent_open_browser(url):
    """Open a browser window silently without any console output."""
    with suppress_stdout_stderr():
        import webbrowser
        webbrowser.open(url)


def start_cleanup_server(output_html, timestamp, port_suffix):
    """Start a tiny HTTP server that deletes *output_html* when the
    browser tab sends a beacon.  Returns (port, cleanup_event)."""
    cleanup_event = threading.Event()

    class CleanupHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == '/cleanup':
                self.send_response(200)
                self.end_headers()
                try:
                    data = json.loads(self.rfile.read(
                        int(self.headers['Content-Length'])).decode())
                    if data.get('token') == timestamp:
                        try:
                            os.remove(output_html)
                        except:
                            pass  # Silently handle cleanup errors
                        cleanup_event.set()
                        threading.Thread(target=self.server.shutdown,
                                        daemon=True).start()
                except:
                    pass  # Silently handle cleanup errors

        def log_message(self, *_):
            pass  # silence default logging

    # Find an available port
    base = 8000
    port = base + (hash(port_suffix) % 1000)
    for _ in range(20):
        try:
            server = socketserver.TCPServer(("127.0.0.1", port), CleanupHandler)
            break
        except OSError:
            port += 1
    else:
        # If we can't find a port, just return a dummy event
        return 0, threading.Event()

    # Start server in a separate thread
    threading.Thread(target=server.serve_forever, daemon=True).start()
    
    return port, cleanup_event

def calculate_trend_info(values):
    """Calculate trend information for a sequence of values"""
    # Calculate overall trend (increase/decrease percentage)
    if len(values) > 1:
        first_valid = next((v for v in values if not np.isnan(v)), 0)
        last_valid = next((v for v in reversed(values) if not np.isnan(v)), 0)
        if abs(first_valid) > 1e-10:  # Avoid division by zero
            change_pct = (last_valid - first_valid) / abs(first_valid) * 100
        else:
            change_pct = 0 if first_valid == last_valid else float('inf')
        
        # Calculate rate of change
        non_nan_values = [v for v in values if not np.isnan(v)]
        if len(non_nan_values) > 1:
            trend_direction = "increasing" if last_valid > first_valid else "decreasing" if last_valid < first_valid else "stable"
            return {
                "change_pct": change_pct,
                "direction": trend_direction,
                "min": min(non_nan_values),
                "max": max(non_nan_values),
                "start": first_valid,
                "end": last_valid
            }
    return None

def find_interesting_points(values, x_numeric):
    """Find interesting points in the data (min, max, significant jumps)"""
    if len(values) < 3:
        return {}
    
    interesting = {}
    non_nan = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
    
    if non_nan:
        # Find global min/max
        min_idx = min(non_nan, key=lambda x: x[1])[0]
        max_idx = max(non_nan, key=lambda x: x[1])[0]
        interesting["min"] = {"idx": min_idx, "value": values[min_idx]}
        interesting["max"] = {"idx": max_idx, "value": values[max_idx]}
        
        # Find biggest jump (could indicate a phase transition)
        diffs = []
        for i in range(1, len(values)):
            if not np.isnan(values[i]) and not np.isnan(values[i-1]):
                diffs.append((i, abs(values[i] - values[i-1])))
        
        if diffs:
            biggest_jump_idx = max(diffs, key=lambda x: x[1])[0]
            interesting["jump"] = {
                "idx": biggest_jump_idx, 
                "value": values[biggest_jump_idx],
                "prev_value": values[biggest_jump_idx-1]
            }
            
    return interesting
