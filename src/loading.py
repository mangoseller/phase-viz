import os
import sys
import time
import datetime
import threading
from typing import Optional

# Environment variable to silence GTK messages - set it as early as possible
os.environ['GTK_DEBUG'] = '0'
os.environ['NO_AT_BRIDGE'] = '1'  # This specifically addresses the atk-bridge message

class ProgressBar:
    """A customizable command-line progress bar with animation."""
    
    def __init__(
        self, 
        total: int, 
        desc: str = "", 
        bar_length: int = 30, 
        fill_char: str = "█",
        empty_char: str = "░",
        unit: str = "it",
        leave: bool = True,
        color: Optional[str] = None,
    ):
        """
        Initialize a new progress bar.
        
        Args:
            total: Total number of items to process
            desc: Description to display before the progress bar
            bar_length: Length of the progress bar in characters
            fill_char: Character to use for filled portion of the bar
            empty_char: Character to use for empty portion of the bar
            unit: Unit of items being processed
            leave: Whether to leave the progress bar after completion
            color: Color to use for the progress bar (uses typer colors)
        """
        self.total = total
        self.desc = desc
        self.bar_length = bar_length
        self.fill_char = fill_char
        self.empty_char = empty_char
        self.unit = unit
        self.leave = leave
        self.color = color
        
        # Internal state
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_print_len = 0
        self.completed = False
        
        # Animation state
        self.animation_frames = ["Computing...", "Computing.", "Computing..", "Computing..."]
        self.animation_index = 0
        self.animation_timer = None
        self.stop_animation = threading.Event()
        
        # Support for typer colors
        self._typer_available = False
        try:
            import typer
            self._typer_available = True
            self._typer = typer
        except ImportError:
            pass
        
        # Start animation if total > 0
        if total > 0:
            self._start_animation()
    
    def _start_animation(self):
        """Start the animation thread for the computing text."""
        if self.animation_timer is None:
            self.stop_animation.clear()
            self.animation_timer = threading.Thread(target=self._animate_computing)
            self.animation_timer.daemon = True
            self.animation_timer.start()
    
    def _animate_computing(self):
        """Animate the computing text."""
        while not self.stop_animation.is_set() and not self.completed:
            self.animation_index = (self.animation_index + 1) % len(self.animation_frames)
            self._render()  # Just render, don't update progress
            time.sleep(0.3)  # Animation speed
    
    def _render(self):
        """Render the progress bar without updating progress."""
        if self.completed:
            return
            
        # Calculate progress
        progress = min(1.0, self.current / self.total)
        filled_length = int(self.bar_length * progress)
        
        # Create the bar
        bar = self.fill_char * filled_length + self.empty_char * (self.bar_length - filled_length)
        
        # Calculate time stats
        now = time.time()
        elapsed = now - self.start_time
        if progress > 0:
            eta = elapsed / progress - elapsed
            eta_str = self._format_time(eta)
        else:
            eta_str = "?"
        
        # Create the progress line with animated computing text
        percent = int(progress * 100)
        animation_text = self.animation_frames[self.animation_index]
        line = f"{self.desc} |{bar}| {self.current}/{self.total} [{percent}%] - ETA: {eta_str} - {animation_text}"
        
        # Apply color if available and specified
        if self._typer_available and self.color:
            line = self._typer.style(line, fg=getattr(self._typer.colors, self.color.upper(), None))
        
        # Clear the previous line
        sys.stdout.write("\r" + " " * self.last_print_len)
        
        # Print the new line
        sys.stdout.write("\r" + line)
        sys.stdout.flush()
        
        self.last_print_len = len(line)
    
    def update(self, current: Optional[int] = None):
        """
        Update the progress bar.
        
        Args:
            current: Current progress (if None, increment by 1)
        """
        if self.completed:
            return
            
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        # Check if we're done
        if self.current >= self.total:
            self.finish()
            return
            
        # Only update rendering if enough time has passed
        now = time.time()
        if now - self.last_update_time < 0.1 and self.current < self.total:
            return
        
        self.last_update_time = now
        self._render()
    
    def finish(self, message: str = "Completed"):
        """Complete the progress bar with a final update."""
        if self.completed:
            return
            
        # Stop the animation
        self.stop_animation.set()
        if self.animation_timer:
            self.animation_timer.join(0.5)  # Wait briefly for animation to stop
        
        self.completed = True
        self.current = self.total
        
        # Calculate progress
        filled_length = self.bar_length
        
        # Create the final bar
        bar = self.fill_char * filled_length
        
        # Create the completion line
        line = f"{self.desc} |{bar}| {self.total}/{self.total} [100%] - {message}"
        
        # Apply color if available and specified
        if self._typer_available and self.color:
            line = self._typer.style(line, fg=getattr(self._typer.colors, self.color.upper(), None))
        
        # Clear the previous line
        sys.stdout.write("\r" + " " * self.last_print_len)
        
        # Print the new line
        sys.stdout.write("\r" + line)
        
        # If we're done, print a newline
        if self.leave:
            sys.stdout.write("\n")
        else:
            sys.stdout.write("\r" + " " * self.last_print_len + "\r")
        
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


class AnimatedText:
    """Simple class to provide animated text in the console."""
    
    def __init__(self, base_text="Processing", color=None):
        self.base_text = base_text
        self.color = color
        self.frames = ["...", ".", "..", "..."]
        self.current_frame = 0
        self.stop_event = threading.Event()
        self.thread = None
    
    def start(self):
        """Start the animation in a separate thread."""
        if self.thread is not None:
            return
            
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
        
    def _animate(self):
        """Animation loop."""
        while not self.stop_event.is_set():
            text = f"{self.base_text}{self.frames[self.current_frame]}"
            if self.color:
                try:
                    import typer
                    text = typer.style(text, fg=self.color)
                except ImportError:
                    pass
                
            # Clear line and print the text
            sys.stdout.write("\r" + " " * 50)
            sys.stdout.write("\r" + text)
            sys.stdout.flush()
            
            # Update frame and sleep
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            time.sleep(0.3)
    
    def stop(self, final_text=None):
        """Stop the animation."""
        if self.thread is None:
            return
            
        self.stop_event.set()
        self.thread.join(0.5)
        self.thread = None
        
        # Print final message if provided
        if final_text:
            try:
                import typer
                if self.color:
                    final_text = typer.style(final_text, fg=self.color)
            except ImportError:
                pass
                
            sys.stdout.write("\r" + " " * 50)
            sys.stdout.write("\r" + final_text + "\n")
            sys.stdout.flush()