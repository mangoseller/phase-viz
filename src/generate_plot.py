from pathlib import Path
from typing import Sequence, Dict, List, Optional, Tuple
import datetime as dt
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import os
import json
import threading
import http.server
import socketserver
import webbrowser
import random
import sys
from contextlib import contextmanager

from utils import (
    suppress_stdout_stderr, 
    silent_open_browser, 
    start_cleanup_server, 
    find_interesting_points, 
    calculate_trend_info,
    logger,
    register_html_for_cleanup,
    unregister_html_from_cleanup
)

os.environ['GTK_MODULES'] = ''  

#TODO: split this into pure JS at some point


def create_annotations(interesting_points, colors):
    """Create annotations for interesting points in the data"""
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
        elif point_type == "jump" and abs(point_data["value"] - point_data["prev_value"]) > 0.1 * (max(point_data["value"], point_data["prev_value"]) - min(point_data["value"], point_data["prev_value"])):
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
    
    return annotations

def plot_metric_interactive(
    checkpoint_names: Sequence[str],
    values: Sequence[float] = None,
    metric_name: str = None,
    metrics_data: Dict[str, List[float]] = None,
) -> None:
    """Render an interactive line-plot (Plotly) of metrics over checkpoints.
    
    Args:
        checkpoint_names: List of checkpoint identifiers
        values: (Optional) List of metric values for a single metric
        metric_name: (Optional) Name of the single metric
        metrics_data: (Optional) Dictionary mapping metric names to values
    
    Returns:
        None: Outputs are saved to HTML and PNG files
    """
    
    # Convert single metric to metrics_data format if provided
    if metrics_data is None:
        if values is not None and metric_name is not None:
            metrics_data = {metric_name: values}
        else:
            raise ValueError("Either (values, metric_name) or metrics_data must be provided")
    
    # Validate input lengths and convert to floats
    for name, vals in metrics_data.items():
        if len(checkpoint_names) != len(vals):
            raise ValueError(
                f"checkpoint_names and values for {name} must have identical length; "
                f"got {len(checkpoint_names)} vs {len(vals)}."
            )
        # Ensure all values are floats
        try:
            metrics_data[name] = [float(v) if v is not None else float('nan') for v in vals]
        except (TypeError, ValueError) as e:
            logger.error(f"Error converting metric '{name}' values to float: {e}")
            raise ValueError(f"Metric '{name}' contains non-numeric values: {e}")
    
    # --- Set up color palette ---
    # Create a color palette with enough distinct colors for all metrics
    base_colors = [
        "#3498db",  # Blue
        "#e74c3c",  # Red
        "#2ecc71",  # Green
        "#9b59b6",  # Purple
        "#f39c12",  # Orange
        "#1abc9c",  # Turquoise
        "#d35400",  # Pumpkin
        "#34495e",  # Dark Blue/Gray
        "#16a085",  # Green Sea
        "#27ae60",  # Nephritis
        "#2980b9",  # Belize Hole
        "#8e44ad",  # Wisteria
        "#c0392b",  # Pomegranate
    ]
    
    # Ensure we have enough colors for all metrics
    metric_colors = {}
    for i, name in enumerate(metrics_data.keys()):
        metric_colors[name] = base_colors[i % len(base_colors)]
    
    colors = {
        "background": "#f9f9f9",     # Light background
        "paper_bg": "#ffffff",       # White for paper elements
        "plot_bg": "#ffffff",        # White for plot area
        "grid": "#e1e1e1",           # Light gray for grid lines
        "text": "#2c3e50",           # Dark blue/gray for text
        "annotation": "#7f8c8d",     # Gray for annotation text
        "highlight": "#9b59b6",      # Purple for highlights
        "button_bg": "#ecf0f1",      # Light gray for buttons
        "button_text": "#2c3e50",    # Dark blue/gray for button text
        "metrics": metric_colors,    # Colors for each metric line
    }
    
    # --- Convert to numeric axis for zoom behavior ---
    x_numeric = list(range(len(checkpoint_names)))
    
    # --- Create main figure ---
    fig = go.Figure()
    
    # --- Calculate overall min/max for y-axis scaling ---
    all_values = []
    for vals in metrics_data.values():
        all_values.extend([v for v in vals if not np.isnan(v)])
    
    y_min = min(all_values) if all_values else 0
    y_max = max(all_values) if all_values else 1
    y_range_padding = (y_max - y_min) * 0.1  # Add 10% padding
    
    # --- Add traces for each metric ---
    annotations_by_metric = {}
    for i, (metric_name, values) in enumerate(metrics_data.items()):
        # Calculate trend info
        trend_info = calculate_trend_info(values)
        
        # Find interesting points
        interesting_points = find_interesting_points(values, x_numeric)
        
        # Create annotations (but store them by metric for toggling)
        metric_annotations = create_annotations(interesting_points, colors)
        annotations_by_metric[metric_name] = metric_annotations
        
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
            
            # Add moving average trace (hidden by default)
            fig.add_trace(
                go.Scatter(
                    x=x_numeric,
                    y=ma_values,
                    mode="lines",
                    name=f"{metric_name} - Moving Avg ({window_size} pts)",
                    line=dict(
                        color=colors["metrics"][metric_name], 
                        width=1.5, 
                        dash="dash"
                    ),
                    hoverinfo="skip",
                    visible=False,  # Hidden by default
                    legendgroup=metric_name,
                )
            )
        
        # Add main trace for this metric
        fig.add_trace(
            go.Scatter(
                x=x_numeric,
                y=values,
                mode="lines+markers",
                name=metric_name,
                line=dict(
                    color=colors["metrics"][metric_name], 
                    width=2, 
                    shape="spline", 
                    smoothing=0.3
                ),
                marker=dict(
                    symbol="circle", 
                    size=6, 
                    color=colors["metrics"][metric_name],
                    line=dict(color=colors["plot_bg"], width=1)
                ),
                hovertemplate=(
                    f"<b>Checkpoint:</b> %{{customdata}}<br>"
                    f"<b>{metric_name}:</b> %{{y:.6f}}<extra></extra>"
                ),
                customdata=checkpoint_names,
                legendgroup=metric_name,
                visible=i == 0,  # Only the first metric is visible by default
            )
        )
    
    # --- Set initial annotations (only for the first metric) ---
    first_metric = next(iter(metrics_data.keys()))
    annotations = annotations_by_metric[first_metric]
    
    # --- Add statistics for first metric ---
    first_values = metrics_data[first_metric]
    stats_annotations = [
        # Statistics positioned at the top center of the plot
        dict(
            xref="paper",
            yref="paper",
            x=0.5,  # Center horizontally
            y=1.04,  # Positioned at the top
            text=f"Min: {min(first_values):.4f} | Max: {max(first_values):.4f} | Mean: {sum(first_values)/len(first_values):.4f}",
            showarrow=False,
            font=dict(family="Inter, system-ui, sans-serif", size=11, color=colors["text"]),
            align="center"  # Center-align the text
        )
    ]
    
    # --- Prepare dropdown menu options ---
    dropdown_options = []
    
    # Add option for each individual metric
    for i, metric_name in enumerate(metrics_data.keys()):
        # Create visibility list for this option (only this metric is visible)
        # Account for both main traces and moving average traces
        visibility = []
        for j, name in enumerate(metrics_data.keys()):
            # For each metric, we have potentially 2 traces (main and moving avg)
            # So we need to set visibility for both
            if name in metrics_data:
                has_ma = len(metrics_data[name]) > 5  # Check if it has moving average
                
                if name == metric_name:
                    # This metric should be visible (but not its moving avg)
                    visibility.extend([True, False] if has_ma else [True])
                else:
                    # Other metrics should be hidden
                    visibility.extend([False, False] if has_ma else [False])
        
        # Add dropdown option
        dropdown_options.append(
            dict(
                args=[
                    {"visible": visibility},
                    {"annotations": annotations_by_metric[metric_name] + stats_annotations},
                    {'title.text': f"<b>{metric_name} over Checkpoints</b>"}
                ],
                label=metric_name,
                method="update"
            )
        )
    
    # Add "All Metrics" option
    all_visibility = []
    for name in metrics_data.keys():
        has_ma = len(metrics_data[name]) > 5
        all_visibility.extend([True, False] if has_ma else [True])  # Show all main traces but not MAs
    
    # Calculate min and max for all metrics for the "All" option stats
    all_min = min(min(vals) for vals in metrics_data.values())
    all_max = max(max(vals) for vals in metrics_data.values())
    all_mean = sum(sum(vals) for vals in metrics_data.values()) / sum(len(vals) for vals in metrics_data.values())
    
    # Combined annotations for "All" option
    all_stats_annotation = dict(
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.04,
        text=f"Min: {all_min:.4f} | Max: {all_max:.4f} | Mean: {all_mean:.4f}",
        showarrow=False,
        font=dict(family="Inter, system-ui, sans-serif", size=11, color=colors["text"]),
        align="center"
    )
    
    dropdown_options.append(
        dict(
            args=[
                {"visible": all_visibility},
                {"annotations": [all_stats_annotation]},  # No individual metric annotations when showing all
                {'title.text': "<b>All Metrics over Checkpoints</b>"}
            ],
            label="All Metrics",
            method="update"
        )
    )
    
    # --- Get title for first selected metric ---
    metric_title = f"<b>{first_metric} over Checkpoints</b>"
    if len(metrics_data) == 1:
        trend_info = calculate_trend_info(list(metrics_data.values())[0])
        if trend_info:
            direction_symbol = "↗" if trend_info["direction"] == "increasing" else "↘" if trend_info["direction"] == "decreasing" else "→"
            trend_color = "#27ae60" if trend_info["direction"] == "increasing" else "#e74c3c" if trend_info["direction"] == "decreasing" else colors["text"]
            metric_title += f" <span style='font-size:0.9em;color:{trend_color};'>{direction_symbol} {abs(trend_info['change_pct']):.1f}% change</span>"
    
    # --- Update layout with enhanced styling ---
    fig.update_layout(
        template="plotly_white",  # Use white template for minimalist design
        paper_bgcolor=colors["paper_bg"],
        plot_bgcolor=colors["plot_bg"],
        font=dict(
            family="Inter, system-ui, -apple-system, sans-serif",
            size=12,
            color=colors["text"]
        ),
        title=dict(
            text=metric_title,
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
            title=dict(text="Metric Value", font=dict(size=14, color=colors["text"])),
            gridcolor=colors["grid"],
            showspikes=True,
            spikethickness=1,
            spikedash="solid",
            spikecolor=colors["annotation"],
            zeroline=True,
            zerolinecolor=colors["grid"],
            zerolinewidth=1.5,
            color=colors["text"],
            # Set range with padding to avoid data being too close to the edge
            range=[y_min - y_range_padding, y_max + y_range_padding]
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
        # Set initial annotations
        annotations=annotations + stats_annotations,
        # Add dropdown menu
        updatemenus=[
            dict(
                buttons=dropdown_options,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor=colors["button_bg"],
                font=dict(color=colors["button_text"], size=12),
                bordercolor=colors["grid"],
                borderwidth=1,
            )
        ]
    )
    
    # --- Create stats HTML for all metrics ---
    all_stats_html = f"""
    <div id="stats-panel" style="padding: 15px; background-color: {colors['paper_bg']}; border: 1px solid {colors['grid']}; 
        border-radius: 4px; margin-top: 20px; color: {colors['text']}; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 800px; margin-left: auto; margin-right: auto;">
        <h4 style="margin-top: 0; color: {colors['text']}; border-bottom: 1px solid {colors['grid']}; padding-bottom: 8px; font-family: 'Inter', system-ui, sans-serif; font-weight: 500;">
            Metric Statistics</h4>
        <div class="metrics-stats-container" style="display: flex; flex-wrap: wrap; gap: 20px;">
    """
    
    # Add a stats panel for each metric
    for metric_name, values in metrics_data.items():
        trend_color = ""
        if values[-1] > values[0]:
            trend_color = "#27ae60"  # Green for increasing
        elif values[-1] < values[0]:
            trend_color = "#e74c3c"  # Red for decreasing
        else:
            trend_color = colors["text"]  # Default for stable
            
        all_stats_html += f"""
        <div class="metric-stats" id="stats-{metric_name.replace(' ', '-').lower()}" style="flex: 1; min-width: 300px;">
            <h5 style="color: {colors['metrics'][metric_name]}; margin-top: 0; font-family: 'Inter', system-ui, sans-serif;">
                {metric_name}</h5>
            <table style="width: 100%; border-collapse: collapse; font-family: 'Inter', system-ui, sans-serif;">
                <tr><td style="padding: 3px 10px;"><b>Mean:</b></td><td>{sum(values)/len(values):.6f}</td></tr>
                <tr><td style="padding: 3px 10px;"><b>Min:</b></td><td>{min(values):.6f}</td></tr>
                <tr><td style="padding: 3px 10px;"><b>Max:</b></td><td>{max(values):.6f}</td></tr>
                <tr><td style="padding: 3px 10px;"><b>Start:</b></td><td>{values[0]:.6f}</td></tr>
                <tr><td style="padding: 3px 10px;"><b>End:</b></td><td>{values[-1]:.6f}</td></tr>
                <tr><td style="padding: 3px 10px;"><b>Change:</b></td>
                    <td style="color: {trend_color}">
                        {values[-1] - values[0]:.6f} ({((values[-1] - values[0]) / abs(values[0]) * 100) if values[0] != 0 else 0:.2f}%)
                    </td></tr>
            </table>
        </div>
        """
    
    all_stats_html += """
        </div>
    </div>
    """
    
    # Create output filename
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
    output_html = Path(f"metrics_over_checkpoints_{timestamp}_{random_suffix}.html").absolute()
    
    # Start cleanup server
    port, cleanup_event = start_cleanup_server(output_html, timestamp, random_suffix)
    logger.info(f"Started cleanup server on port {port} for {output_html}")
    
    # Add JavaScript for improved interactivity and cleanup
    custom_js = f"""
    <script>
        // Cleanup configuration
        const cleanupPort = {port};
        const cleanupURL = 'http://localhost:' + cleanupPort + '/cleanup';
        let cleanupAttempted = false;
        
        // Metrics data for JS interactions
        const metricsData = {json.dumps(metrics_data)};
        const checkpointNames = {json.dumps(list(checkpoint_names))};
        const metricColors = {json.dumps(colors['metrics'])};
        
        // Function to send cleanup request
        async function sendCleanupRequest() {{
            if (cleanupAttempted) return;
            cleanupAttempted = true;
            
            try {{
                // Try using fetch first
                await fetch(cleanupURL, {{
                    method: 'POST',
                    mode: 'no-cors',
                    keepalive: true,
                    body: JSON.stringify({{cleanup: true}})
                }});
            }} catch (e) {{
                // If fetch fails, try sendBeacon
                try {{
                    navigator.sendBeacon(cleanupURL, JSON.stringify({{cleanup: true}}));
                }} catch (e2) {{
                    console.error('Cleanup failed:', e2);
                }}
            }}
        }}
        
        // Multiple cleanup triggers
        window.addEventListener('beforeunload', sendCleanupRequest);
        window.addEventListener('unload', sendCleanupRequest);
        
        // Also cleanup on visibility change (tab switching)
        document.addEventListener('visibilitychange', function() {{
            if (document.visibilityState === 'hidden') {{
                sendCleanupRequest();
            }}
        }});
        
        // Cleanup on page hide (mobile browsers)
        window.addEventListener('pagehide', sendCleanupRequest);
        
        // Add custom interactivity
        document.addEventListener('DOMContentLoaded', function() {{
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
            
            downloadBtn.onmouseover = function() {{
                this.style.backgroundColor = '#dfe6e9';
            }};
            
            downloadBtn.onmouseout = function() {{
                this.style.backgroundColor = '#ecf0f1';
            }};
            
            downloadBtn.onclick = function() {{
                Plotly.downloadImage(
                    document.getElementsByClassName('js-plotly-plot')[0], 
                    {{format: 'png', width: 1200, height: 800, filename: 'model_metrics_plot'}}
                );
            }};
            
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
            
            csvBtn.onmouseover = function() {{
                this.style.backgroundColor = '#dfe6e9';
            }};
            
            csvBtn.onmouseout = function() {{
                this.style.backgroundColor = '#ecf0f1';
            }};
            
            csvBtn.onclick = function() {{
                // Get all metrics data
                let csvContent = "Checkpoint";
                
                // Add header row with metric names
                for (const metricName in metricsData) {{
                    csvContent += "," + metricName;
                }}
                csvContent += "\\n";
                
                // Add data rows
                for (let i = 0; i < checkpointNames.length; i++) {{
                    csvContent += checkpointNames[i];
                    for (const metricName in metricsData) {{
                        csvContent += "," + metricsData[metricName][i];
                    }}
                    csvContent += "\\n";
                }}
                
                // Create and trigger download
                const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
                const link = document.createElement('a');
                const url = URL.createObjectURL(blob);
                link.setAttribute('href', url);
                link.setAttribute('download', 'model_metrics_data.csv');
                link.style.visibility = 'hidden';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }};
            
            btnContainer.appendChild(csvBtn);
            
            // Add to document
            document.getElementsByClassName('js-plotly-plot')[0].appendChild(btnContainer);
            
            // Get the dropdown buttons
            var dropdownButtons = document.querySelector('.updatemenu-container');
            if (dropdownButtons) {{
                dropdownButtons.style.fontFamily = "'Inter', system-ui, sans-serif";
                dropdownButtons.style.fontSize = '12px';
            }}
            
            // Update stats panel visibility when a metric is selected
            var plotDiv = document.getElementsByClassName('js-plotly-plot')[0];
            if (plotDiv && plotDiv._context) {{
                var origUpdate = plotDiv._context.plotlyServerURL;
                plotDiv.on('plotly_click', function() {{
                    console.log('Plot clicked');
                }});
                
                plotDiv.on('plotly_afterplot', function() {{
                    console.log('After plot');
                }});
                
                // Custom update handler for dropdown selection
                plotDiv.on('plotly_afterupdate', function(data) {{
                    const currentButton = document.querySelector('.updatemenu-item.active');
                    if (currentButton) {{
                        const selectedMetric = currentButton.textContent.trim();
                        
                        // Update stats visibility
                        for (const metricName in metricsData) {{
                            const statsPanel = document.getElementById(`stats-${{metricName.replace(' ', '-').toLowerCase()}}`);
                            if (statsPanel) {{
                                if (selectedMetric === "All Metrics" || selectedMetric === metricName) {{
                                    statsPanel.style.display = '';
                                }} else {{
                                    statsPanel.style.display = 'none';
                                }}
                            }}
                        }}
                    }}
                }});
            }}
            
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
            
            // Add stats panel to document
            var plotContainer = document.getElementsByClassName('js-plotly-plot')[0].parentElement;
            var statsHtml = `{all_stats_html.replace('`', '\\`').replace("'", "\\'").replace('"', '\\"')}`;
            var statsDiv = document.createElement('div');
            statsDiv.innerHTML = statsHtml;
            
            // Insert range selector and stats panel
            plotContainer.appendChild(rangeSelector);
            plotContainer.appendChild(statsDiv.firstElementChild);
            
            // Initially, hide all stats panels except for the first one
            const firstMetric = Object.keys(metricsData)[0];
            for (const metricName in metricsData) {{
                if (metricName !== firstMetric) {{
                    const statsPanel = document.getElementById(`stats-${{metricName.replace(' ', '-').toLowerCase()}}`);
                    if (statsPanel) {{
                        statsPanel.style.display = 'none';
                    }}
                }}
            }}
            
            // Implement range selector functionality
            var viewAllBtn = document.getElementById('view-all');
            var rangeStart = document.getElementById('range-start');
            var rangeEnd = document.getElementById('range-end');
            var rangeDisplay = document.getElementById('range-display');
            var plotlyDiv = document.getElementsByClassName('js-plotly-plot')[0];
            
            function updateRangeDisplay() {{
                var startPct = parseInt(rangeStart.value);
                var endPct = parseInt(rangeEnd.value);
                var startIdx = Math.floor(startPct / 100 * (checkpointNames.length - 1));
                var endIdx = Math.floor(endPct / 100 * (checkpointNames.length - 1));
                
                if (checkpointNames.length > 0) {{
                    rangeDisplay.textContent = `${{checkpointNames[startIdx]}} to ${{checkpointNames[endIdx]}}`;
                }} else {{
                    rangeDisplay.textContent = `${{startPct}}% - ${{endPct}}%`;
                }}
                
                // Update the plot's x-axis range
                var plotLayout = plotlyDiv._fullLayout;
                if (plotLayout && plotLayout.xaxis) {{
                    Plotly.relayout(plotlyDiv, {{
                        'xaxis.range': [startIdx, endIdx]
                    }});
                }}
            }}
            
            rangeStart.addEventListener('input', function() {{
                if (parseInt(rangeStart.value) > parseInt(rangeEnd.value) - 10) {{
                    rangeStart.value = parseInt(rangeEnd.value) - 10;
                }}
                updateRangeDisplay();
            }});
            
            rangeEnd.addEventListener('input', function() {{
                if (parseInt(rangeEnd.value) < parseInt(rangeStart.value) + 10) {{
                    rangeEnd.value = parseInt(rangeStart.value) + 10;
                }}
                updateRangeDisplay();
            }});
            
            viewAllBtn.addEventListener('click', function() {{
                rangeStart.value = 0;
                rangeEnd.value = 100;
                updateRangeDisplay();
            }});
            
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
        }});
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
    
    /* Better dropdown styling */
    .updatemenu-container {
        font-family: 'Inter', system-ui, sans-serif !important;
        font-size: 12px !important;
    }
    
    .updatemenu-item {
        font-family: 'Inter', system-ui, sans-serif !important;
    }
    </style>
    """

    # Generate the final HTML
    raw_html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    html_with_head = raw_html.replace("<head>", f"<head>{meta_tags}")
    final_html = html_with_head.replace("</body>", f"{custom_js}</body>")
    
    try:
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(final_html)
        
        logger.info(f"Created HTML file: {output_html}")
        
        # Open browser silently
        with suppress_stdout_stderr():
            webbrowser.open(output_html.as_uri())
        
        # Wait for cleanup to complete
        cleanup_event.wait()
        logger.info(f"Cleanup completed for {output_html}")
        
    except Exception as e:
        logger.error(f"Error creating/opening HTML file: {e}")
        # Make sure to clean up if something goes wrong
        unregister_html_from_cleanup(output_html)
        if output_html.exists():
            try:
                output_html.unlink()
            except:
                pass
        raise
    
    # Try to save PNG version
    try:
        with suppress_stdout_stderr():
            png_path = output_html.with_suffix(".png")
            fig.write_image(png_path, scale=2, width=1200, height=800)
            logger.info(f"Saved PNG file: {png_path}")
    except Exception as e:
        logger.warning(f"Could not save PNG file: {e}")
        pass  # PNG export is optional


    