# astrakairos/planner/gui/utilities.py

"""
Utility functions and classes for the GUI components.
"""

import tkinter as tk
from tkinter import ttk
from datetime import datetime
from typing import Optional
import pytz


class GUIUtilities:
    """Utility class for common GUI operations."""
    
    def __init__(self, parent_app):
        self.app = parent_app
    
    def create_scrollable_frame(self, parent):
        """Create a scrollable frame widget."""
        # Create main frame
        main_frame = ttk.Frame(parent)
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure parent grid
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Configure main frame grid
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def configure_canvas_width(event):
            # Make the canvas content width match the canvas width
            canvas_width = event.width
            canvas.itemconfig(canvas_frame_id, width=canvas_width)
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        
        canvas_frame_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Configure>", configure_canvas_width)
        
        # Add mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # Configure scrollable frame to expand horizontally
        scrollable_frame.columnconfigure(0, weight=1)
        
        # Pack canvas and scrollbar with proper sticky
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        return scrollable_frame

    @staticmethod
    def configure_text_widget_scroll(text_widget):
        """Configure proper mouse wheel scrolling for text widgets."""
        def _on_text_mousewheel(event):
            text_widget.yview_scroll(int(-1*(event.delta/120)), "units")
            return "break"  # Prevent event from bubbling up
        
        def _bind_text_mousewheel(event):
            text_widget.bind("<MouseWheel>", _on_text_mousewheel)
        
        def _unbind_text_mousewheel(event):
            text_widget.unbind("<MouseWheel>")
        
        text_widget.bind('<Enter>', _bind_text_mousewheel)
        text_widget.bind('<Leave>', _unbind_text_mousewheel)


def format_ra_hours(hours: float) -> str:
    """Format right ascension in hours to HMS format."""
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h:02d}h{m:02d}m"


def format_dec_degrees(degrees: float) -> str:
    """Format declination in degrees to DMS format."""
    sign = "+" if degrees >= 0 else "-"
    abs_deg = abs(degrees)
    d = int(abs_deg)
    m = int((abs_deg - d) * 60)
    return f"{sign}{d:02d}°{m:02d}'"


def format_time(dt_obj: Optional[datetime]) -> str:
    """Format datetime object to time string."""
    return dt_obj.strftime('%H:%M %Z') if dt_obj else "N/A"


def format_time_utc(dt_obj: Optional[datetime]) -> str:
    """Format datetime object to UTC time string."""
    if dt_obj and hasattr(dt_obj, 'astimezone'):
        utc_time = dt_obj.astimezone(pytz.utc)
        return utc_time.strftime('%H:%M UTC')
    return "N/A"


def validate_coordinate_range(value: float, min_val: float, max_val: float, name: str) -> bool:
    """Validate that a coordinate value is within acceptable range."""
    if not (min_val <= value <= max_val):
        return False
    return True


def safe_float_conversion(value: str, default: float = 0.0) -> float:
    """Safely convert string to float with fallback."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int_conversion(value: str, default: int = 0) -> int:
    """Safely convert string to int with fallback."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
