# astrakairos/planner/gui/data_export.py

"""
Data export functionality for the GUI.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from ...config import EXPORT_FORMATS


class ExportManager:
    """Manages data export functionality."""
    
    def __init__(self, parent_app):
        self.app = parent_app
    
    def create_export_section(self, parent):
        """Create the export options section for imported data."""
        frame = ttk.LabelFrame(parent, text="🔄 Export Options", padding="10")
        frame.grid(row=9, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        
        # Export format selection
        ttk.Label(frame, text="Export Format:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.app.export_format_var = tk.StringVar(value="csv")
        format_combo = ttk.Combobox(frame, textvariable=self.app.export_format_var, width=30)
        format_combo['values'] = [f"{fmt['name']} ({fmt['extension']})" for fmt in EXPORT_FORMATS.values()]
        format_combo['state'] = 'readonly'
        format_combo.grid(row=0, column=1, sticky="ew", padx=5)
        
        # Export button
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=0, column=2, padx=5)
        
        ttk.Button(button_frame, text=" Export Data", 
                  command=self._export_imported_data).grid(row=0, column=0)
        
        # Export options
        options_frame = ttk.Frame(frame)
        options_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        self.app.include_metadata_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Include metadata", 
                       variable=self.app.include_metadata_var).grid(row=0, column=0, sticky="w")
        
        self.app.include_coordinates_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Include coordinates", 
                       variable=self.app.include_coordinates_var).grid(row=0, column=1, sticky="w", padx=(20, 0))
        
        self.app.include_analysis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Include analysis results", 
                       variable=self.app.include_analysis_var).grid(row=0, column=2, sticky="w", padx=(20, 0))
        
        return frame
    
    def create_progress_section(self, parent):
        """Create the progress and status section."""
        frame = ttk.LabelFrame(parent, text="📊 Status", padding="10")
        frame.grid(row=11, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        
        # Status label
        ttk.Label(frame, text="Status:").grid(row=0, column=0, sticky="w")
        self.app.status_var = tk.StringVar(value="Ready")
        self.app.status_label = ttk.Label(frame, textvariable=self.app.status_var, foreground='green')
        self.app.status_label.grid(row=0, column=1, sticky="w", padx=(10, 0))
        
        # Progress bar
        ttk.Label(frame, text="Progress:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.app.progress_var = tk.DoubleVar()
        self.app.progress_bar = ttk.Progressbar(frame, mode='determinate', variable=self.app.progress_var)
        self.app.progress_bar.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(5, 0))
        
        # Results counter
        self.app.results_var = tk.StringVar(value="No results")
        results_label = ttk.Label(frame, textvariable=self.app.results_var, font=('Arial', 9))
        results_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))
        
        return frame
    
    def _export_imported_data(self):
        """Export imported CSV data."""
        search_manager = self.app.search_manager
        if not hasattr(search_manager, 'imported_csv_data') or search_manager.imported_csv_data is None:
            messagebox.showwarning("No Data", "No imported data to export. Please import a CSV file first.")
            return
        
        # Get selected format
        format_text = self.app.export_format_var.get()
        format_key = None
        for key, fmt in EXPORT_FORMATS.items():
            if format_text.startswith(fmt['name']):
                format_key = key
                break
        
        if not format_key:
            messagebox.showerror("Error", "Invalid export format selected.")
            return
        
        # Convert DataFrame to list of dictionaries for export
        data_to_export = search_manager.imported_csv_data.to_dict('records')
        
        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        format_info = EXPORT_FORMATS[format_key]
        default_filename = f"astrakairos_export_{timestamp}{format_info['extension']}"
        
        # Get save location
        filename = filedialog.asksaveasfilename(
            initialname=default_filename,
            defaultextension=format_info['extension'],
            filetypes=[(format_info['name'], f"*{format_info['extension']}"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            self._perform_export(filename, format_key, data_to_export)
            messagebox.showinfo("Export Complete", f"Data exported to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
    
    def _perform_export(self, filename: str, format_key: str, data: List[Dict[str, Any]]):
        """Perform the actual export based on format."""
        self.app.status_var.set("Exporting...")
        self.app.progress_var.set(0)
        
        if format_key == 'csv':
            self._export_csv(filename, data)
        elif format_key == 'json':
            self._export_json(filename, data)
        elif format_key == 'fits':
            self._export_fits(filename, data)
        elif format_key == 'votable':
            self._export_votable(filename, data)
        elif format_key == 'latex':
            self._export_latex(filename, data)
        else:
            raise ValueError(f"Unsupported export format: {format_key}")
        
        self.app.status_var.set("Export complete")
        self.app.progress_var.set(100)
    
    def _export_csv(self, filename: str, data: List[Dict[str, Any]]):
        """Export results to CSV format."""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            if not data:
                return
            
            # Get all possible field names
            fieldnames = set()
            for item in data:
                fieldnames.update(item.keys())
            fieldnames = sorted(list(fieldnames))
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, item in enumerate(data):
                # Add metadata if requested
                if self.app.include_metadata_var.get():
                    item = item.copy()  # Don't modify original
                    item['export_date'] = datetime.now().isoformat()
                    if hasattr(self.app, 'location_manager') and self.app.location_manager.selected_location:
                        item['observatory'] = self.app.location_manager.selected_location.get('name', '')
                
                writer.writerow(item)
                
                # Update progress
                self.app.progress_var.set((i + 1) / len(data) * 100)
                self.app.root.update_idletasks()
    
    def _export_json(self, filename: str, data: List[Dict[str, Any]]):
        """Export results to JSON format."""
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'format_version': '1.0',
                'total_results': len(data)
            },
            'results': data
        }
        
        if self.app.include_metadata_var.get() and hasattr(self.app, 'location_manager'):
            if self.app.location_manager.selected_location:
                export_data['metadata']['observatory'] = self.app.location_manager.selected_location.get('name', '')
        
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)
    
    def _export_fits(self, filename: str, data: List[Dict[str, Any]]):
        """Export results to FITS format."""
        messagebox.showinfo("FITS Export", 
                          "FITS export requires astropy.io.fits\\nPlease install: pip install astropy")
    
    def _export_votable(self, filename: str, data: List[Dict[str, Any]]):
        """Export results to VOTable format."""
        messagebox.showinfo("VOTable Export", 
                          "VOTable export requires astropy.io.votable\\nPlease install: pip install astropy")
    
    def _export_latex(self, filename: str, data: List[Dict[str, Any]]):
        """Export results to LaTeX table format."""
        with open(filename, 'w', encoding='utf-8') as texfile:
            texfile.write("\\\\begin{table}[h]\\n")
            texfile.write("\\\\centering\\n")
            texfile.write("\\\\caption{Binary Star Search Results}\\n")
            texfile.write("\\\\label{tab:binary_stars}\\n")
            texfile.write("\\\\begin{tabular}{|c|c|c|c|c|c|c|}\\n")
            texfile.write("\\\\hline\\n")
            texfile.write("Name & RA & Dec & Magnitude & Separation & Position Angle & Epoch \\\\\\\\\\n")
            texfile.write("\\\\hline\\n")
            
            for item in data:
                texfile.write(f"{item.get('name', '')} & ")
                texfile.write(f"{item.get('ra', '')} & ")
                texfile.write(f"{item.get('dec', '')} & ")
                texfile.write(f"{item.get('magnitude', '')} & ")
                texfile.write(f"{item.get('separation', '')} & ")
                texfile.write(f"{item.get('position_angle', '')} & ")
                texfile.write(f"{item.get('epoch', '')} \\\\\\\\\\n")
            
            texfile.write("\\\\hline\\n")
            texfile.write("\\\\end{tabular}\\n")
            texfile.write("\\\\end{table}\\n")
