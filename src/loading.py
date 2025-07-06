import sys
import time
import datetime
import threading


class SimpleLoadingAnimation:
    """A simple animated loading indicator for console output."""
    
    def __init__(self, base_text="Loading", color=None):
        self.base_text = base_text
        self.color = color
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current_frame = 0
        self.stop_event = threading.Event()
        self.thread = None
        self.start_time = None
        
        # Use a lock to protect the progress data during concurrent updates
        self.lock = threading.Lock()
        self.progress = {'current': 0, 'total': 1}  
        # Support for typer colors
        self._typer_available = False
        try:
            import typer
            self._typer_available = True
            self._typer = typer
        except ImportError:
            pass
    
    def start(self, total_metrics=1):
        """Start the animation in a separate thread."""
        if self.thread is not None:
            return
            
        self.start_time = time.time()
        with self.lock:
            self.progress = {'current': 0, 'total': total_metrics}
        
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
    
    def update(self, current=None, total=None):
        """Update progress information in a thread-safe way."""
        if current is None and total is None:
            return
            
        with self.lock:
            if current is not None:
                self.progress['current'] = current
            if total is not None:
                self.progress['total'] = total
    
    def _animate(self):
        """Animation loop."""
        while not self.stop_event.is_set():
            # Get a thread-safe copy of the current progress
            with self.lock:
                progress = self.progress.copy()
            
            # TODO: Fix time estimation - doesn't work and isn't really needed 
            # Loading is too quick for it to be warranted, in the tests i've seen

            elapsed = time.time() - self.start_time
            progress_ratio = max(0.01, progress['current'] / progress['total']) if progress['total'] > 0 else 0
            # Avoid division by zero or very small numbers
            if progress_ratio > 0.01:
                eta = (elapsed / progress_ratio) - elapsed
            else:
                eta = 0
            
            elapsed_str = self._format_time(elapsed)
            eta_str = self._format_time(eta)
            
            # Create the loading text
            if progress['current'] > 0:
                metrics_text = f"({progress['current']}/{progress['total']} metrics completed)"
                text = f"{self.frames[self.current_frame]} {self.base_text} {metrics_text} - {elapsed_str} elapsed, ~{eta_str} remaining"
            else:
                text = f"{self.frames[self.current_frame]} {self.base_text}... {elapsed_str} elapsed"
            
            # Apply color if available
            if self._typer_available and self.color:
                try:
                    text = self._typer.style(text, fg=self.color)
                except:
                    pass

            sys.stdout.write("\r" + " " * 100)  # Clear line
            sys.stdout.write("\r" + text)
            sys.stdout.flush()
            
            # Update animation frame
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            time.sleep(0.1)
    
    def stop(self, final_text=None):
        """Stop the animation."""
        if self.thread is None:
            return
            
        self.stop_event.set()
        self.thread.join(0.5)
        self.thread = None
        
        # Calculate total time
        if self.start_time is not None:
            total_time = time.time() - self.start_time
            time_str = self._format_time(total_time)   
            # Create completion message
            if final_text:
                message = f"{final_text} (completed in {time_str})"
            else:
                message = f"✓ {self.base_text} completed in {time_str}"
            
            if self._typer_available and self.color:
                try:
                    message = self._typer.style(message, fg=self.color)
                except:
                    pass
                
            sys.stdout.write("\r" + " " * 100) 
            sys.stdout.write("\r" + message + "\n")
            sys.stdout.flush()
    
    def _format_time(self, seconds):
        """Format seconds into a human-readable time string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            seconds = seconds % 60
            return f"{minutes}m {int(seconds)}s"
        else:
            return str(datetime.timedelta(seconds=int(seconds)))