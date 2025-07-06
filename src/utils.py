from contextlib import contextmanager
import sys
import numpy as np
import threading
import http.server
import socketserver
import os
import signal
import atexit
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Set, Tuple, List, Type
from inbuilt_metrics import *
import importlib.util
import inspect
import torch.nn as nn


# Track HTML files for cleanup
_temp_html_files: Set[Path] = set()
_cleanup_lock = threading.Lock()

BUILTIN_METRICS = {
    "entropy":        ("Weight Entropy",          weight_entropy_of_model),
    "connectivity":   ("Layer Connectivity",      layer_connectivity_of_model),
    "variance":       ("Parameter Variance",      parameter_variance_of_model),
    "norm_ratio":     ("Layer Wise Norm Ratio",   layer_wise_norm_ratio_of_model),
    "capacity":       ("Activation Capacity",     activation_capacity_of_model),
    "rank":           ("Weight Rank",             weight_rank_of_model),
    "gradient_flow":  ("Gradient Flow Score",     gradient_flow_score_of_model),
    "effective_rank": ("Effective Rank",          effective_rank_of_model),
    "condition":      ("Avg Condition Number",    avg_condition_number_of_model),
    "flatness":       ("Flatness Proxy",          flatness_proxy_of_model),
    "mean":           ("Mean Weight",             mean_weight_of_model),
    "skew":           ("Weight Skew",             weight_skew_of_model),
    "kurtosis":       ("Weight Kurtosis",         weight_kurtosis_of_model),
    "isotropy":       ("Isotropy",                isotropy_of_model),
    "weight_norm":    ("Weight Norm",             weight_norm_of_model),
    "spectral":       ("Spectral Norm",           spectral_norm_of_model),
    "participation":  ("Participation Ratio",     participation_ratio_of_model),
    "sparsity":       ("Sparsity",                sparsity_of_model),
    "max_activation": ("Max Activation",          max_activation_of_model),
}



def setup_file_logging(log_dir: str = "logs", max_lines: int = 5000) -> logging.Logger:
    """Set up file-based logging, new log created after max_lines."""
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    class LineCountRotatingHandler(logging.handlers.RotatingFileHandler):
        def __init__(self, filename, max_lines=max_lines, *args, **kwargs):
            super().__init__(filename, *args, **kwargs)
            self.max_lines = max_lines
            self.line_count = 0
            self._count_existing_lines()
        
        def _count_existing_lines(self):
            try:
                if os.path.exists(self.baseFilename):
                    with open(self.baseFilename, 'r') as f:
                        self.line_count = sum(1 for _ in f)
            except:
                self.line_count = 0
        
        def emit(self, record):
            """Write a record, rotating if we exceed max_lines."""
            if self.line_count >= self.max_lines:
                self.doRollover()
                self.line_count = 0
            
            super().emit(record)
            self.line_count += 1
    
    logger = logging.getLogger("phase-viz")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    log_file = log_path / "phase-viz.log"
    file_handler = LineCountRotatingHandler(
        str(log_file),
        max_lines=max_lines,
        maxBytes=0,  
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass
    logger.addHandler(NullHandler())
    return logger

logger = setup_file_logging()

@contextmanager
def suppress_stdout_stderr():
    """
    Suppress stdout and stderr.
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
    """Open a browser window silently"""
    with suppress_stdout_stderr():
        import webbrowser
        webbrowser.open(url)


def register_html_for_cleanup(filepath: Path):
    """Register an HTML file for cleanup on exit."""
    with _cleanup_lock:
        _temp_html_files.add(filepath)
        logger.info(f"Registered HTML file for cleanup: {filepath}")

def unregister_html_from_cleanup(filepath: Path):
    """Remove an HTML file from the cleanup list."""
    with _cleanup_lock:
        _temp_html_files.discard(filepath)
        logger.info(f"Unregistered HTML file from cleanup: {filepath}")

def cleanup_all_html_files():
    """Clean up all registered HTML files."""
    with _cleanup_lock:
        for filepath in _temp_html_files.copy():
            try:
                if filepath.exists():
                    filepath.unlink()
                    logger.info(f"Cleaned up HTML file: {filepath}")
            except Exception as e:
                logger.error(f"Error cleaning up {filepath}: {e}")
        _temp_html_files.clear()

# Register cleanup handlers 
def _signal_handler(signum, frame):
    """Handle termination signals."""
    logger.info(f"Received signal {signum}, cleaning up...")
    cleanup_all_html_files()
    sys.exit(0)

# Register signal handlers and atexit handler 
signal.signal(signal.SIGINT, _signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, _signal_handler)  # Termination
atexit.register(cleanup_all_html_files)


class CleanupHTTPServer:
    """HTTP server for handling browser-based cleanup requests."""
    def __init__(self, filepath: Path, cleanup_callback: Optional[callable] = None):
        self.filepath = filepath
        self.cleanup_callback = cleanup_callback
        self.server = None
        self.port = None
        self.thread = None
        self.cleanup_event = threading.Event()
        
    def start(self) -> Tuple[int, threading.Event]:
        """Start the cleanup server and return (port, cleanup_event)."""
        
        class CleanupHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(handler_self):
                if handler_self.path == '/cleanup':
                    handler_self.send_response(200)
                    handler_self.send_header('Access-Control-Allow-Origin', '*')
                    handler_self.end_headers()
                    
                    try:
                        # Clean up the file
                        if self.filepath.exists():
                            self.filepath.unlink()
                            unregister_html_from_cleanup(self.filepath)
                            logger.info(f"Browser requested cleanup of {self.filepath}")

                        self.cleanup_event.set()
                        if self.cleanup_callback:
                            self.cleanup_callback()
                    
                        # Schedule server shutdown
                        threading.Thread(
                            target=lambda: self.server.shutdown(),
                            daemon=True
                        ).start()
                    except Exception as e:
                        logger.error(f"Error during browser cleanup: {e}")
            
            def do_OPTIONS(handler_self):
                """Handle CORS preflight requests."""
                handler_self.send_response(200)
                handler_self.send_header('Access-Control-Allow-Origin', '*')
                handler_self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
                handler_self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                handler_self.end_headers()
            
            def log_message(self, *args):
                """Suppress default HTTP logging."""
                pass
        
        # Find an available port
        base_port = 8000
        for port_offset in range(100):
            port = base_port + port_offset
            try:
                self.server = socketserver.TCPServer(("127.0.0.1", port), CleanupHandler)
                self.server.socket.setsockopt(socketserver.socket.SOL_SOCKET, 
                                            socketserver.socket.SO_REUSEADDR, 1)
                self.port = port
                break
            except OSError:
                continue
        else:
            logger.error("Could not find available port for cleanup server")
            return 0, self.cleanup_event
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()   
        logger.info(f"Started cleanup server on port {self.port} for {self.filepath}")
        return self.port, self.cleanup_event
    
    def stop(self):
        """Stop the cleanup server."""
        if self.server:
            self.server.shutdown()
            if self.thread:
                self.thread.join(timeout=1)
            logger.info(f"Stopped cleanup server on port {self.port}")


def start_cleanup_server(output_html: Path) -> Tuple[int, threading.Event]:
    """Start a cleanup server for the HTML file.
    
    Args:
        output_html: Path to the HTML file
        
    Returns:
        Tuple of (port, cleanup_event)
    """
    register_html_for_cleanup(output_html)
    server = CleanupHTTPServer(output_html)
    return server.start()


def calculate_trend_info(values):
    """Calculate trend information for a sequence of values"""
    # Calculate trends
    if len(values) > 1:
        first_valid = next((v for v in values if not np.isnan(v)), 0)
        last_valid = next((v for v in reversed(values) if not np.isnan(v)), 0)
        if abs(first_valid) > 1e-10:  # Avoid division by zero
            change_pct = (last_valid - first_valid) / abs(first_valid) * 100
        else:
            change_pct = 0 if first_valid == last_valid else float('inf')

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
        min_idx = min(non_nan, key=lambda x: x[1])[0]
        max_idx = max(non_nan, key=lambda x: x[1])[0]
        interesting["min"] = {"idx": min_idx, "value": values[min_idx]}
        interesting["max"] = {"idx": max_idx, "value": values[max_idx]}    
        # Find biggest jumps
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


# Need specific err/logic for class name loading issues
class ClassNameLoadingError(Exception):
    pass 

def get_model_classes(file_path: str) -> List[str]:
    """
    Automatically discover all nn.Module subclasses in a Python file.
    
    Args:
        file_path: Path to the Python file containing model classes
        
    Returns:
        List of class names that inherit from nn.Module
    """
    try:
        spec = importlib.util.spec_from_file_location("model_module", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {file_path}")
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find all classes that inherit from nn.Module
        model_classes = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if it's a subclass of nn.Module but not nn.Module itself
            # and that it's defined in this module (not imported)
            if (issubclass(obj, nn.Module) and 
                obj != nn.Module and 
                obj.__module__ == module.__name__):
                model_classes.append(name)
        
        return model_classes
    except Exception as e:
        raise ClassNameLoadingError("Could not infer model class") from e