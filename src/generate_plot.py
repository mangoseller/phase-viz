# TODO: debug and read this carefully - fix html file generation deletion
from pathlib import Path
from typing import Sequence
import datetime as dt
import plotly.graph_objects as go
import numpy as np

def plot_metric_interactive(
    checkpoint_names: Sequence[str],
    values: Sequence[float],
    metric_name: str,
) -> None:  # noqa: D401 – simple signature kept intentionally
    """Render an interactive line-plot (Plotly) of *metric* over checkpoints.
    
    Args:
        checkpoint_names: List of checkpoint identifiers
        values: List of metric values corresponding to each checkpoint
        metric_name: Name of the metric being visualized
    
    Returns:
        None: Outputs are saved to HTML and PNG files
    """
    
    if len(checkpoint_names) != len(values):
        raise ValueError(
            "checkpoint_names and values must have identical length; "
            f"got {len(checkpoint_names)} vs {len(values)}."
        )
    
    # --- Convert to numeric axis for zoom behavior -------------------
    x_numeric = list(range(len(checkpoint_names)))
    
    # --- Calculate trend info for annotations ---
    def calculate_trend_info(values):
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
    
    trend_info = calculate_trend_info(values)
    
    # --- Minimalist design color palette and styling ---
    colors = {
        "primary": "#3498db",        # Blue for main line
        "markers": "#e74c3c",        # Red for point markers
        "background": "#f9f9f9",     # Light background
        "paper_bg": "#ffffff",       # White for paper elements
        "plot_bg": "#ffffff",        # White for plot area
        "grid": "#e1e1e1",           # Light gray for grid lines
        "text": "#2c3e50",           # Dark blue/gray for text
        "annotation": "#7f8c8d",     # Gray for annotation text
        "highlight": "#9b59b6",      # Purple for highlights
        "moving_avg": "#2ecc71",     # Green for moving average
        "button_bg": "#ecf0f1",      # Light gray for buttons
        "button_text": "#2c3e50"     # Dark blue/gray for button text
    }
    
    # --- Detect interesting points ---
    def find_interesting_points(values, x_numeric):
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
    
    interesting_points = find_interesting_points(values, x_numeric)
    
    # --- Create annotations for interesting points ---
    annotations = []
    for point_type, point_data in interesting_points.items():
        if point_type == "min":
            annotations.append(dict(
                x=point_data["idx"],
                y=point_data["value"],
                text="Min",
                showarrow=True,
                arrowhead=2,
                arrowcolor=colors["annotation"],
                arrowsize=1,
                arrowwidth=1.5,
                font=dict(family="Inter, system-ui, sans-serif", color=colors["annotation"], size=10),
                ax=0,
                ay=-30
            ))
        elif point_type == "max":
            annotations.append(dict(
                x=point_data["idx"],
                y=point_data["value"],
                text="Max",
                showarrow=True,
                arrowhead=2,
                arrowcolor=colors["annotation"],
                arrowsize=1,
                arrowwidth=1.5,
                font=dict(family="Inter, system-ui, sans-serif", color=colors["annotation"], size=10),
                ax=0,
                ay=-30
            ))
        elif point_type == "jump" and abs(point_data["value"] - point_data["prev_value"]) > 0.1 * (max(values) - min(values)):
            annotations.append(dict(
                x=point_data["idx"],
                y=point_data["value"],
                text="Significant Change",
                showarrow=True,
                arrowhead=2,
                arrowcolor=colors["highlight"],
                arrowsize=1,
                arrowwidth=1.5,
                font=dict(family="Inter, system-ui, sans-serif", color=colors["highlight"], size=10),
                ax=30,
                ay=-30
            ))
    
    # --- Create main figure ---
    fig = go.Figure()
    
    # Add main trace
    fig.add_trace(
        go.Scatter(
            x=x_numeric,
            y=values,
            mode="lines+markers",
            name=metric_name,
            line=dict(color=colors["primary"], width=2, shape="spline", smoothing=0.3),
            marker=dict(
                symbol="circle", 
                size=6, 
                color=colors["markers"],
                line=dict(color=colors["plot_bg"], width=1)
            ),
            hovertemplate=(
                f"<b>Checkpoint:</b> %{{customdata}}<br>"
                f"<b>{metric_name}:</b> %{{y:.6f}}<extra></extra>"
            ),
            customdata=checkpoint_names,
        )
    )
    
    # Add moving average if we have enough points
    if len(values) > 5:
        window_size = max(3, len(values) // 10)
        ma_values = []
        for i in range(len(values)):
            start = max(0, i - window_size // 2)
            end = min(len(values), i + window_size // 2 + 1)
            valid_values = [values[j] for j in range(start, end) if not np.isnan(values[j])]
            if valid_values:
                ma_values.append(sum(valid_values) / len(valid_values))
            else:
                ma_values.append(np.nan)
                
        fig.add_trace(
            go.Scatter(
                x=x_numeric,
                y=ma_values,
                mode="lines",
                name=f"Moving Avg ({window_size} pts)",
                line=dict(color=colors["moving_avg"], width=1.5, dash="dash"),
                hoverinfo="skip",
                visible="legendonly"  # Hidden by default
            )
        )
    
    # --- Create range selector instead of buttons ---
    # This replaces the problematic rangeslider and buttons
    
    # --- Create title with trend info ---
    title_text = f"<b>{metric_name} over Checkpoints</b>"
    if trend_info:
        direction_symbol = "↗" if trend_info["direction"] == "increasing" else "↘" if trend_info["direction"] == "decreasing" else "→"
        # Choose color based on direction (green for increasing, red for decreasing)
        trend_color = "#27ae60" if trend_info["direction"] == "increasing" else "#e74c3c" if trend_info["direction"] == "decreasing" else colors["text"]
        title_text += f" <span style='font-size:0.9em;color:{trend_color};'>{direction_symbol} {abs(trend_info['change_pct']):.1f}% change</span>"
    
    # --- Update layout with enhanced styling ---
    fig.update_layout(
        template="plotly_white",  # Use white template for minimalist design
        paper_bgcolor=colors["paper_bg"],
        plot_bgcolor=colors["plot_bg"],
        font=dict(
            family="Inter, system-ui, -apple-system, sans-serif",  # More professional font stack
            size=12,
            color=colors["text"]
        ),
        title=dict(
            text=title_text,
            x=0.01, 
            xanchor="left",
            font=dict(family="Inter, system-ui, sans-serif", size=20, color=colors["text"])
        ),
        xaxis=dict(
            title="Checkpoint",
            # Handle overlapping labels by using a subset of ticks
            tickmode="array",
            tickvals=x_numeric[::max(1, len(x_numeric) // 10)] if len(x_numeric) > 10 else x_numeric,
            ticktext=[checkpoint_names[i] for i in range(0, len(checkpoint_names), max(1, len(x_numeric) // 10))] if len(x_numeric) > 10 else checkpoint_names,
            tickangle=45,  # Angle labels to avoid overlap
            gridcolor=colors["grid"],
            showspikes=True,
            spikethickness=1,
            spikedash="solid",
            spikecolor=colors["annotation"],
            spikemode="across",
            color=colors["text"]
        ),
        yaxis=dict(
            title=dict(text=metric_name, font=dict(size=14, color=colors["text"])),
            gridcolor=colors["grid"],
            showspikes=True,
            spikethickness=1,
            spikedash="solid",
            spikecolor=colors["annotation"],
            zeroline=True,
            zerolinecolor=colors["grid"],
            zerolinewidth=1.5,
            color=colors["text"]
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=colors["paper_bg"],
            font_size=12,
            font_family="Inter, system-ui, sans-serif",
            font_color=colors["text"],
            bordercolor=colors["grid"]
        ),
        margin=dict(l=50, r=50, t=80, b=80),
        height=600,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor=colors["paper_bg"],
            bordercolor=colors["grid"],
            borderwidth=1,
            font=dict(color=colors["text"])
        ),
        # Combine all annotations
        annotations=[
            *annotations,
            # Statistics positioned at the top center of the plot for better visibility
            dict(
                xref="paper",
                yref="paper",
                x=0.5,  # Center horizontally
                y=1.04,  # Positioned at the top
                text=f"Min: {min(values):.4f} | Max: {max(values):.4f} | Mean: {sum(values)/len(values):.4f}",
                showarrow=False,
                font=dict(family="Inter, system-ui, sans-serif", size=11, color=colors["text"]),
                align="center"  # Center-align the text
            )
        ]
    )
    
    # --- Add statistics display with improved styling ---
    stats_html = f"""
    <div id="stats-panel" style="padding: 15px; background-color: {colors['paper_bg']}; border: 1px solid {colors['grid']}; 
        border-radius: 4px; margin-top: 20px; color: {colors['text']}; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 600px; margin-left: auto; margin-right: auto;">
        <h4 style="margin-top: 0; color: {colors['text']}; border-bottom: 1px solid {colors['grid']}; padding-bottom: 8px; font-family: 'Inter', system-ui, sans-serif; font-weight: 500;">
            Metric Statistics</h4>
        <table style="width: 100%; border-collapse: collapse; font-family: 'Inter', system-ui, sans-serif;">
            <tr><td style="padding: 5px 10px;"><b>Mean:</b></td><td>{sum(values)/len(values):.6f}</td></tr>
            <tr><td style="padding: 5px 10px;"><b>Min:</b></td><td>{min(values):.6f}</td></tr>
            <tr><td style="padding: 5px 10px;"><b>Max:</b></td><td>{max(values):.6f}</td></tr>
            <tr><td style="padding: 5px 10px;"><b>Start:</b></td><td>{values[0]:.6f}</td></tr>
            <tr><td style="padding: 5px 10px;"><b>End:</b></td><td>{values[-1]:.6f}</td></tr>
            <tr><td style="padding: 5px 10px;"><b>Change:</b></td>
                <td style="color: {'#27ae60' if values[-1] > values[0] else '#e74c3c' if values[-1] < values[0] else colors['text']}">
                    {values[-1] - values[0]:.6f} ({((values[-1] - values[0]) / abs(values[0]) * 100):.2f}%)
                </td></tr>
        </table>
    </div>
    """
    
    # Create output filename
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_metric = metric_name.replace(" ", "_")
    output_html = Path(f"{safe_metric}_over_checkpoints_{timestamp}.html").absolute()
    
    # Add JavaScript for improved interactivity and temporary file cleanup
    custom_js = """
    <script>
        // Store file path for cleanup
        const tempFilePath = '""" + str(output_html) + """';
        
        // Add event listener for page unload to clean up the file
        window.addEventListener('beforeunload', function() {
            // Create a cleanup request
            try {
                navigator.sendBeacon('/cleanup', JSON.stringify({
                    path: tempFilePath,
                    token: '""" + timestamp + """'
                }));
            } catch(e) {
                console.error("Error in cleanup:", e);
            }
        });
        
        // Add custom interactivity
        document.addEventListener('DOMContentLoaded', function() {
            // Create container for buttons
            var btnContainer = document.createElement('div');
            btnContainer.style.position = 'absolute';
            btnContainer.style.top = '10px';
            btnContainer.style.right = '10px';  // Positioned on right
            btnContainer.style.zIndex = 1000;
            
            // Create download button
            var downloadBtn = document.createElement('button');
            downloadBtn.innerText = 'Download PNG';
            downloadBtn.style.padding = '8px 12px';
            downloadBtn.style.backgroundColor = '#ecf0f1';
            downloadBtn.style.color = '#2c3e50';
            downloadBtn.style.fontWeight = '500';
            downloadBtn.style.border = '1px solid #bdc3c7';
            downloadBtn.style.borderRadius = '4px';
            downloadBtn.style.cursor = 'pointer';
            downloadBtn.style.marginRight = '10px';
            downloadBtn.style.fontFamily = "'Inter', system-ui, sans-serif";
            downloadBtn.style.fontSize = '12px';
            downloadBtn.style.transition = 'all 0.2s ease';
            
            downloadBtn.onmouseover = function() {
                this.style.backgroundColor = '#dfe6e9';
            };
            
            downloadBtn.onmouseout = function() {
                this.style.backgroundColor = '#ecf0f1';
            };
            
            downloadBtn.onclick = function() {
                Plotly.downloadImage(
                    document.getElementsByClassName('js-plotly-plot')[0], 
                    {format: 'png', width: 1200, height: 800, filename: 'model_metric_plot'}
                );
            };
            
            btnContainer.appendChild(downloadBtn);
            
            // Create CSV export button
            var csvBtn = document.createElement('button');
            csvBtn.innerText = 'Export CSV';
            csvBtn.style.padding = '8px 12px';
            csvBtn.style.backgroundColor = '#ecf0f1';
            csvBtn.style.color = '#2c3e50';
            csvBtn.style.fontWeight = '500';
            csvBtn.style.border = '1px solid #bdc3c7';
            csvBtn.style.borderRadius = '4px';
            csvBtn.style.cursor = 'pointer';
            csvBtn.style.fontFamily = "'Inter', system-ui, sans-serif";
            csvBtn.style.fontSize = '12px';
            csvBtn.style.transition = 'all 0.2s ease';
            
            csvBtn.onmouseover = function() {
                this.style.backgroundColor = '#dfe6e9';
            };
            
            csvBtn.onmouseout = function() {
                this.style.backgroundColor = '#ecf0f1';
            };
            
            csvBtn.onclick = function() {
                // Get the data from the plot
                var plotData = document.getElementsByClassName('js-plotly-plot')[0].data;
                var x = plotData[0].customdata;
                var y = plotData[0].y;
                
                // Create CSV content
                var csvContent = "Checkpoint,Value\\n";
                for (var i = 0; i < x.length; i++) {
                    csvContent += x[i] + "," + y[i] + "\\n";
                }
                
                // Create and trigger download
                var blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                var link = document.createElement('a');
                var url = URL.createObjectURL(blob);
                link.setAttribute('href', url);
                link.setAttribute('download', 'model_metric_data.csv');
                link.style.visibility = 'hidden';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            };
            
            btnContainer.appendChild(csvBtn);
            
            // Add to document
            document.getElementsByClassName('js-plotly-plot')[0].appendChild(btnContainer);
            
            // Add stats panel below the plot
            var plotContainer = document.getElementsByClassName('js-plotly-plot')[0].parentElement;
            var statsPanel = document.getElementById('stats-panel');
            if (statsPanel) {
                // It's already there from the HTML injection
            } else {
                // Create it manually
                var statsDiv = document.createElement('div');
                statsDiv.innerHTML = `""" + stats_html + """`;
                plotContainer.appendChild(statsDiv.firstElementChild);
            }
            
            // Add range selector for better viewing experience
            var plot = document.getElementsByClassName('js-plotly-plot')[0];
            var rangeSelector = document.createElement('div');
            rangeSelector.id = 'custom-range-selector';
            rangeSelector.style.margin = '20px auto';
            rangeSelector.style.maxWidth = '600px';
            rangeSelector.style.padding = '10px';
            rangeSelector.style.textAlign = 'center';
            rangeSelector.style.fontFamily = "'Inter', system-ui, sans-serif";
            
            rangeSelector.innerHTML = `
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px; color: #2c3e50;">
                    <span style="font-size: 12px; font-weight: 500;">View Range:</span>
                    <span id="range-display" style="font-size: 12px;"></span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <button id="view-all" class="range-btn" style="padding: 6px 12px; background-color: #ecf0f1; color: #2c3e50; border: 1px solid #bdc3c7; border-radius: 4px; cursor: pointer; font-size: 12px; flex: 0 0 auto;">All Data</button>
                    <div style="flex-grow: 1; position: relative; height: 30px;">
                        <input type="range" id="range-start" min="0" max="100" value="0" step="5" style="position: absolute; width: 100%; pointer-events: none; opacity: 0.7; height: 5px; top: 12px; background: transparent;">
                        <input type="range" id="range-end" min="0" max="100" value="100" step="5" style="position: absolute; width: 100%; pointer-events: none; opacity: 0.7; height: 5px; top: 12px; background: transparent;">
                    </div>
                </div>
            `;
            
            // Insert range selector before stats panel
            plotContainer.insertBefore(rangeSelector, statsPanel || null);
            
            // Implement range selector functionality
            var viewAllBtn = document.getElementById('view-all');
            var rangeStart = document.getElementById('range-start');
            var rangeEnd = document.getElementById('range-end');
            var rangeDisplay = document.getElementById('range-display');
            var plotlyDiv = document.getElementsByClassName('js-plotly-plot')[0];
            var plotData = plotlyDiv.data;
            var xValues = [];
            
            if (plotData && plotData[0]) {
                xValues = plotData[0].customdata;
            }
            
            function updateRangeDisplay() {
                var startPct = parseInt(rangeStart.value);
                var endPct = parseInt(rangeEnd.value);
                var startIdx = Math.floor(startPct / 100 * (xValues.length - 1));
                var endIdx = Math.floor(endPct / 100 * (xValues.length - 1));
                
                if (xValues.length > 0) {
                    rangeDisplay.textContent = `${xValues[startIdx]} to ${xValues[endIdx]}`;
                } else {
                    rangeDisplay.textContent = `${startPct}% - ${endPct}%`;
                }
                
                // Update the plot's x-axis range
                var plotLayout = plotlyDiv._fullLayout;
                if (plotLayout && plotLayout.xaxis) {
                    Plotly.relayout(plotlyDiv, {
                        'xaxis.range': [startIdx, endIdx]
                    });
                }
            }
            
            rangeStart.addEventListener('input', function() {
                if (parseInt(rangeStart.value) > parseInt(rangeEnd.value) - 10) {
                    rangeStart.value = parseInt(rangeEnd.value) - 10;
                }
                updateRangeDisplay();
            });
            
            rangeEnd.addEventListener('input', function() {
                if (parseInt(rangeEnd.value) < parseInt(rangeStart.value) + 10) {
                    rangeEnd.value = parseInt(rangeStart.value) + 10;
                }
                updateRangeDisplay();
            });
            
            viewAllBtn.addEventListener('click', function() {
                rangeStart.value = 0;
                rangeEnd.value = 100;
                updateRangeDisplay();
            });
            
            // Initialize range display
            updateRangeDisplay();
            
            // Apply minimal styling to the entire page
            document.body.style.backgroundColor = '#f9f9f9';
            document.body.style.color = '#2c3e50';
            document.body.style.fontFamily = "'Inter', system-ui, sans-serif";
            document.body.style.margin = '0';
            document.body.style.padding = '20px';
            
            // Create a container for the title
            var titleContainer = document.createElement('div');
            titleContainer.style.textAlign = 'center';
            titleContainer.style.marginBottom = '20px';
            titleContainer.innerHTML = '<h1 style="color:#2c3e50; font-family: \\'Inter\\', system-ui, sans-serif; font-weight: 600; font-size: 24px;">Neural Network Training Metrics</h1>';
            
            // Insert before the plot
            var plotElement = document.getElementsByClassName('js-plotly-plot')[0].parentElement;
            document.body.insertBefore(titleContainer, plotElement);
            
            // Load Inter font
            var fontLink = document.createElement('link');
            fontLink.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap';
            fontLink.rel = 'stylesheet';
            document.head.appendChild(fontLink);
        });
    </script>
    """
    
    # Add meta tags for proper mobile display and improved styling
    meta_tags = """
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    body {
        margin: 0;
        padding: 20px;
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #f9f9f9;
        color: #2c3e50;
    }
    
    /* Custom styling for range inputs */
    input[type=range] {
        -webkit-appearance: none;
        width: 100%;
        background: transparent;
    }
    
    input[type=range]::-webkit-slider-thumb {
        -webkit-appearance: none;
        height: 15px;
        width: 15px;
        border-radius: 50%;
        background: #3498db;
        cursor: pointer;
        margin-top: -6px;
        pointer-events: auto;
        border: 1px solid white;
    }
    
    input[type=range]::-moz-range-thumb {
        height: 15px;
        width: 15px;
        border-radius: 50%;
        background: #3498db;
        cursor: pointer;
        pointer-events: auto;
        border: 1px solid white;
    }
    
    input[type=range]::-webkit-slider-runnable-track {
        width: 100%;
        height: 3px;
        cursor: pointer;
        background: #bdc3c7;
        border-radius: 1.5px;
    }
    
    input[type=range]::-moz-range-track {
        width: 100%;
        height: 3px;
        cursor: pointer;
        background: #bdc3c7;
        border-radius: 1.5px;
    }
    
    button.range-btn:hover {
        background-color: #dfe6e9;
    }
    
    /* Improve plot responsiveness */
    .js-plotly-plot {
        max-width: 100%;
        margin: 0 auto;
    }
    </style>
    """
    
    # Create cleanup server to handle the file deletion
    import threading
    import http.server
    import socketserver
    import json
    import os
    
    def start_cleanup_server():
        class CleanupHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == '/cleanup':
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length).decode('utf-8')
                    try:
                        data = json.loads(post_data)
                        if data.get('token') == timestamp:  # Verify the token
                            file_path = data.get('path')
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                # Also try to remove the PNG if it exists
                                png_path = file_path.replace('.html', '.png')
                                if os.path.exists(png_path):
                                    os.remove(png_path)
                                print(f"Cleaned up temporary files: {file_path}")
                    except Exception as e:
                        print(f"Cleanup error: {e}")
                    
                    self.send_response(200)
                    self.end_headers()
                
            def log_message(self, format, *args):
                # Suppress logging
                return
        
        # Find an available port
        port = 8000
        while port < 8100:  # Try up to port 8099
            try:
                httpd = socketserver.TCPServer(("", port), CleanupHandler)
                break
            except OSError:
                port += 1
                if port >= 8100:
                    print("Warning: Could not find an open port for cleanup server")
                    return None
        
        # Run the server in a separate thread
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True  # So the thread will exit when the main program exits
        server_thread.start()
        print(f"Cleanup server started on port {port}")
        
        return port
    
    # Write HTML with custom JS included
    with open(output_html, 'w') as f:
        raw_html = fig.to_html(include_plotlyjs="cdn", full_html=True)
        # Insert meta tags in the head
        enhanced_html = raw_html.replace('<head>', f'<head>{meta_tags}')
        # Insert custom JS before the closing body tag
        enhanced_html = enhanced_html.replace('</body>', f'{stats_html}{custom_js}</body>')
        f.write(enhanced_html)
    
    # Start the cleanup server
    cleanup_port = start_cleanup_server()
    
    if cleanup_port:
        # Update the cleanup URL in the HTML
        with open(output_html, 'r') as f:
            html_content = f.read()
        
        # Update the URL in the sendBeacon call
        updated_html = html_content.replace('/cleanup', f'http://localhost:{cleanup_port}/cleanup')
        
        with open(output_html, 'w') as f:
            f.write(updated_html)
    
    # Open the HTML in a browser
    import webbrowser
    try:
        webbrowser.open(output_html.as_uri())
    except ValueError:
        # Fallback if URI conversion fails
        webbrowser.open(str(output_html))
    
    # Try to export PNG as well
    try:
        png_path = output_html.with_suffix(".png")
        fig.write_image(png_path, scale=2, width=1200, height=800)
    except (ValueError, ImportError):
        print("Note: PNG export requires additional dependencies (kaleido). Install with 'pip install kaleido'.")
    
    return
