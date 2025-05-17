import tkinter as tk
import time
from config import GUI_UPDATE_INTERVAL
import queue

class BatchUpdateManager:
    """Manages batch updates to the GUI to prevent lag."""
    
    def __init__(self):
        self.pending_updates = []
        self.last_update_time = 0
        self.update_running = False
    
    def add_update(self, update_func, *args, **kwargs):
        """Add an update function to the pending queue."""
        self.pending_updates.append((update_func, args, kwargs))
    
    def process_updates(self, force=False):
        """Process pending updates if interval has elapsed or force is True."""
        current_time = time.time()
        
        # Skip if another update is already in progress
        if self.update_running:
            return False
            
        # Check if it's time for an update
        if force or (current_time - self.last_update_time >= GUI_UPDATE_INTERVAL):
            if self.pending_updates:
                self.update_running = True
                self.last_update_time = current_time
                
                # Process all pending updates
                while self.pending_updates:
                    update_func, args, kwargs = self.pending_updates.pop(0)
                    try:
                        update_func(*args, **kwargs)
                    except Exception as e:
                        print(f"Error in GUI update: {e}", flush=True)
                
                self.update_running = False
                return True
        
        return False

class GUIQueueProcessor:
    """Handles processing message queue for GUI updates."""
    
    def __init__(self, gui_queue, output_handler, status_handler, overlay_handler=None):
        self.gui_queue = gui_queue
        self.output_handler = output_handler
        self.status_handler = status_handler
        self.overlay_handler = overlay_handler
        self.batch_manager = BatchUpdateManager()
        self.threads_completed = 0
        self.expected_threads = 0
        self.initial_status_shown = False
        self._check_queue_job_id = None
    
    def set_expected_threads(self, count):
        """Set the number of threads expected to complete."""
        self.threads_completed = 0
        self.expected_threads = count
    
    def check_gui_queue(self, post_complete_callback=None):
        """Process messages from the GUI queue with batched updates."""
        process_queue = True
        all_threads_finished_this_cycle = False
        any_updates = False
        
        while process_queue:
            try:
                message = self.gui_queue.get_nowait()
                
                # Process the message
                if message is None:
                    self.threads_completed += 1
                    print(f"GUI received None signal. Threads completed: {self.threads_completed}/{self.expected_threads}", flush=True)
                    
                    # Check if all expected threads have completed
                    if self.threads_completed >= self.expected_threads:
                        print(f"All {self.expected_threads} threads signaled completion.", flush=True)
                        all_threads_finished_this_cycle = True
                        process_queue = False  # Stop processing queue items for this cycle
                
                elif isinstance(message, str):
                    any_updates = True
                    
                    # Translation messages
                    if message.startswith("[EN"):
                        if self.initial_status_shown:
                            self.batch_manager.add_update(self.status_handler, "Translating...")
                            self.initial_status_shown = False
                        
                        self.batch_manager.add_update(self.output_handler, message, "translation")
                        
                        # Also update overlay if available
                        if self.overlay_handler:
                            # Extract translation from message
                            translation = message.split("] ", 1)[-1].strip()
                            self.batch_manager.add_update(self.overlay_handler, translation)
                    
                    # Critical errors
                    elif "CRITICAL ERROR" in message:
                        status_part = message.split("CRITICAL ERROR:", 1)[-1].strip() if "CRITICAL ERROR:" in message else "Critical Error Occurred"
                        if len(status_part) > 80: 
                            status_part = status_part[:77] + "..."
                        
                        # Critical errors should be updated immediately
                        self.status_handler(f"CRITICAL ERROR: {status_part}")
                        self.output_handler(f"*** {message} ***", "error")
                        self.initial_status_shown = False
                        process_queue = False
                    
                    # Initialization status
                    elif message.startswith("__INIT_STATUS__"):
                        status_part = message.split("__INIT_STATUS__", 1)[-1]
                        self.batch_manager.add_update(self.status_handler, status_part)
                        self.batch_manager.add_update(self.output_handler, status_part, "info")
                        self.initial_status_shown = True
                    
                    # Error messages
                    elif message.startswith("[ERROR") or message.startswith("[Warning"):
                        # Skip common PyTorch warnings that are not relevant to users
                        if "warning: nn" in message.lower() or "torch.functional" in message:
                            # Just print to console but don't show in UI
                            print(f"Suppressed: {message}", flush=True)
                            continue
                            
                        self.batch_manager.add_update(self.output_handler, message, "error")
                        self.initial_status_shown = False
                    
                    # Info messages
                    elif message.startswith("["): 
                        status_part = message.split("] ", 1)[-1]
                        significant_init = ("initialized" in status_part or 
                                            "Capturing" in status_part or 
                                            "terminated" in status_part or 
                                            "finished" in status_part)
                                            
                        if significant_init and "CRITICAL" not in status_part:
                            if len(status_part) > 80: 
                                status_part = status_part[:77] + "..."
                            self.batch_manager.add_update(self.status_handler, status_part)
                            self.initial_status_shown = True
                            
                        self.batch_manager.add_update(self.output_handler, message, "info")
                    
                    # Other messages
                    else:
                        self.batch_manager.add_update(self.output_handler, message, "info")
            
            except tk.TclError as e:
                if "invalid command name" not in str(e):
                    print(f"Error: GUI TclError in check_gui_queue: {e}", flush=True)
                process_queue = False
                self._check_queue_job_id = None
                return  # Exit check_gui_queue
                
            except queue.Empty:
                # Queue is empty, exit the loop silently
                process_queue = False
                
            except Exception as e:
                # Only log actual errors (not empty queue exceptions)
                print(f"Error processing GUI queue: {e}", flush=True)
                try: 
                    self.output_handler(f"[ERROR] GUI Queue Error: {e}", "error")
                except: 
                    pass
                process_queue = False
        
        # Process any pending updates
        if any_updates:
            self.batch_manager.process_updates()
        
        # Handle thread completion
        if all_threads_finished_this_cycle and post_complete_callback:
            # Call the finalization function outside the queue processing loop
            return post_complete_callback()
        else:
            # Reschedule check if not all threads are done
            return True  # Continue checking

def create_queue_processor(root, gui_queue, output_handler, status_handler, overlay_handler=None):
    """Create and configure a GUI queue processor."""
    processor = GUIQueueProcessor(gui_queue, output_handler, status_handler, overlay_handler)
    
    def schedule_check(post_complete_callback=None):
        if root.winfo_exists():
            def check_wrapper():
                try:
                    continue_checking = processor.check_gui_queue(post_complete_callback)
                    if continue_checking:
                        # Dynamic timeout: shorter if we had updates, longer if idle
                        if processor.gui_queue.qsize() > 0:
                            delay = 50  # Faster updates when queue has items
                        else:
                            delay = 250  # Longer delay when queue is empty
                        processor._check_queue_job_id = root.after(delay, check_wrapper)
                except Exception as e:
                    print(f"Error in queue check wrapper: {e}", flush=True)
                    # Always reschedule to maintain the check loop
                    processor._check_queue_job_id = root.after(500, check_wrapper)
            
            # Initial schedule
            processor._check_queue_job_id = root.after(100, check_wrapper)
    
    def cancel_check():
        if processor._check_queue_job_id and root.winfo_exists():
            try:
                root.after_cancel(processor._check_queue_job_id)
                processor._check_queue_job_id = None
                print("Cancelled queue processor check.", flush=True)
            except tk.TclError as e:
                print(f"Error cancelling queue check: {e}", flush=True)
            except Exception as e:
                print(f"Unexpected error cancelling queue check: {e}", flush=True)
    
    processor.schedule_check = schedule_check
    processor.cancel_check = cancel_check
    
    return processor 