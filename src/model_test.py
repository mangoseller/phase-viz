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
                "prev_value": values[biggest_jump_idx-1],
                "min_value": min(values),
                "max_value": max(values)
            }
 
    return interesting
 
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
        elif point_type == "jump":
            # Calculate threshold for significant change based on the values in this jump
            jump_value = point_data["value"]
            prev_value = point_data["prev_value"]
            jump_threshold = 0.1 * abs(max(jump_value, prev_value) - min(jump_value, prev_value))
 
            if abs(jump_value - prev_value) > jump_threshold:
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
 
def normalize_values(metrics_data):
    """Normalize each metric's values to a 0-1 scale"""
    normalized_data = {}
 
    for metric_name, values in metrics_data.items():
        # Filter out NaN values
        valid_values = [v for v in values if not np.isnan(v)]
        if not valid_values:
            normalized_data[metric_name] = values.copy()  # Keep as is if all NaN
            continue
 
        min_val = min(valid_values)
        max_val = max(valid_values)
 
        # Check if range is too small to normalize
        if max_val - min_val < 1e-10:
            # Just center the values at 0.5 if they're all the same
            normalized_data[metric_name] = [0.5 if not np.isnan(v) else np.nan for v in values]
        else:
            # Normalize to 0-1 range
            normalized_data[metric_name] = [
                (v - min_val) / (max_val - min_val) if not np.isnan(v) else np.nan 
                for v in values
            ]
 
    return normalized_data
 
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
 
    # Validate input lengths
    for name, vals in metrics_data.items():
        if len(checkpoint_names) != len(vals):
            raise ValueError(
                f"checkpoint_names and values for {name} must have identical length; "
                f"got {len(checkpoint_names)} vs {len(vals)}."
            )
 
    # --- Normalize data for multi-metric views ---
    normalized_metrics = normalize_values(metrics_data)
 
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
 
    # --- Calculate y-axis ranges for each metric ---
    y_ranges = {}
    for metric_name, values in metrics_data.items():
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            y_min = min(valid_values)
            y_max = max(valid_values)
            y_range_padding = (y_max - y_min) * 0.1  # Add 10% padding
            y_ranges[metric_name] = [y_min - y_range_padding, y_max + y_range_padding]
        else:
            y_ranges[metric_name] = [0, 1]  # Default range if no valid values
 
    # --- Calculate overall min/max for y-axis scaling (for "All" normalized view) ---
    y_norm_min = 0
    y_norm_max = 1
 
    # --- Add traces for each metric (original values) ---
    annotations_by_metric = {}
 
    # Track trace indices for each metric
    trace_indices = {}
    current_trace_idx = 0
 
    for i, (metric_name, values) in enumerate(metrics_data.items()):
        # Calculate trend info
        trend_info = calculate_trend_info(values)
 
        # Find interesting points
        interesting_points = find_interesting_points(values, x_numeric)
 
        # Create annotations (but store them by metric for toggling)
        metric_annotations = create_annotations(interesting_points, colors)
        annotations_by_metric[metric_name] = metric_annotations
 
        # Add main trace for this metric's raw values
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
        trace_indices[f"{metric_name}_raw"] = current_trace_idx
        current_trace_idx += 1
 
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
                    name=f"{metric_name} - MA",
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
            trace_indices[f"{metric_name}_ma"] = current_trace_idx
            current_trace_idx += 1
 
    # --- Add traces for normalized values (hidden by default) ---
    for i, (metric_name, values) in enumerate(normalized_metrics.items()):
        fig.add_trace(
            go.Scatter(
                x=x_numeric,
                y=values,
                mode="lines+markers",
                name=f"{metric_name} (Normalized)",
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
                    f"<b>{metric_name} (Normalized):</b> %{{y:.6f}}<extra></extra>"
                ),
                customdata=checkpoint_names,
                legendgroup=f"{metric_name}_norm",
                visible=False,  # Hidden by default
            )
        )
        trace_indices[f"{metric_name}_norm"] = current_trace_idx
        current_trace_idx += 1
 
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
 
    # --- Create buttons for multi-select ---
    # Instead of a dropdown, we'll use checkboxes implemented as buttons
    button_list = []
 
    # Individual metric buttons (act as toggles)
    for metric_name in metrics_data.keys():
        button_list.append(
            dict(
                args=[
                    {"visible": [False] * len(fig.data)},  # Start with all hidden
                    {"annotations": [], "yaxis.title.text": f"{metric_name}", "yaxis.range": y_ranges[metric_name]},
                    {"title.text": f"<b>{metric_name} over Checkpoints</b>"}
                ],
                label=metric_name,
                method="update"
            )
        )
 
    # Add "Normalized View" button
    button_list.append(
        dict(
            args=[
                {"visible": [False] * len(fig.data)},  # Start with all hidden
                {"annotations": [], "yaxis.title.text": "Normalized Value (0-1)", "yaxis.range": [0, 1]},
                {"title.text": "<b>Multiple Metrics over Checkpoints (Normalized)</b>"}
            ],
            label="Normalized View",
            method="update"
        )
    )
 
    # --- Create HTML content for custom checkboxes ---
    checkbox_html = """
    <div id="metric-toggles" style="
        position: absolute;
        top: 80px;
        left: 10px;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #e1e1e1;
        border-radius: 4px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        z-index: 1000;
        max-width: 250px;
        font-family: 'Inter', system-ui, sans-serif;
    ">
        <div style="font-weight: 500; margin-bottom: 8px; color: #2c3e50; border-bottom: 1px solid #e1e1e1; padding-bottom: 5px;">
            Select Metrics:
        </div>
        <div id="metric-checkbox-container" style="display: flex; flex-direction: column; gap: 6px;">
    """
 
    # Add a checkbox for each metric
    for metric_name, color in colors["metrics"].items():
        checkbox_html += f"""
        <label class="metric-toggle" style="display: flex; align-items: center; gap: 5px; cursor: pointer; user-select: none;">
            <input type="checkbox" data-metric="{metric_name}" style="cursor: pointer;" 
                   {('checked="checked"' if metric_name == first_metric else '')}>
            <span class="metric-color" style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background-color: {color};"></span>
            <span style="font-size: 14px;">{metric_name}</span>
        </label>
        """
 
    # Add special view options
    checkbox_html += """
        <div style="margin-top: 8px; border-top: 1px solid #e1e1e1; padding-top: 5px;"></div>
        <label class="view-toggle" style="display: flex; align-items: center; gap: 5px; cursor: pointer; user-select: none;">
            <input type="checkbox" id="normalized-view" style="cursor: pointer;">
            <span style="font-size: 14px;">Normalized View</span>
        </label>
        <div style="margin-top: 8px;">
            <button id="select-all" style="
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 12px;
                color: #2c3e50;
                cursor: pointer;
                transition: background-color 0.2s;
                margin-right: 5px;
            ">All</button>
            <button id="select-none" style="
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 12px;
                color: #2c3e50;
                cursor: pointer;
                transition: background-color 0.2s;
            ">None</button>
        </div>
    </div>
    </div>
    """
 
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
            text=f"<b>{first_metric} over Checkpoints</b>",
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
            title=dict(text=first_metric, font=dict(size=14, color=colors["text"])),
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
            range=y_ranges[first_metric],
            autorange=False,  # Disable autorange to use our custom range
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
 
    # Add JavaScript for improved interactivity and stats toggling
    custom_js = f"""
    <script>
        // Store file path for cleanup
        const tempFilePath = '{str(output_html)}';
        const timestamp = '{timestamp}';
 
        // Metrics data for JS interactions
        const metricsData = {json.dumps(metrics_data)};
        const normalizedData = {json.dumps(normalized_metrics)};
        const checkpointNames = {json.dumps(list(checkpoint_names))};
        const metricColors = {json.dumps(colors['metrics'])};
        const traceIndices = {json.dumps(trace_indices)};
        const yRanges = {json.dumps(y_ranges)};
 
        // Add event listener for page unload to clean up the file
        window.addEventListener('beforeunload', function() {{
            // Create a cleanup request
            try {{
                const cleanupURL = 'http://localhost:{random_suffix}/cleanup';
                navigator.sendBeacon(cleanupURL, JSON.stringify({{
                    path: tempFilePath,
                    token: timestamp
                }}));
            }} catch(e) {{
                console.error("Error in cleanup:", e);
            }}
        }});
 
        // Add custom interactivity
        document.addEventListener('DOMContentLoaded', function() {{
            // Add the metric toggles to the document
            const metricTogglesHTML = `{checkbox_html.replace('`', '\\`').replace("'", "\\'").replace('"', '\\"')}`;
            const togglesDiv = document.createElement('div');
            togglesDiv.innerHTML = metricTogglesHTML;
            document.getElementsByClassName('js-plotly-plot')[0].appendChild(togglesDiv.firstChild);
 
            // Create container for buttons
            var btnContainer = document.createElement('div');
            btnContainer.style.position = 'absolute';
            btnContainer.style.top = '10px';
            btnContainer.style.right = '10px';
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
 
            // Add stats panel to document
            var plotContainer = document.getElementsByClassName('js-plotly-plot')[0].parentElement;
            var statsHtml = `{all_stats_html.replace('`', '\\`').replace("'", "\\'").replace('"', '\\"')}`;
            var statsDiv = document.createElement('div');
            statsDiv.innerHTML = statsHtml;
 
            // Add range selector for better viewing experience
            var rangeSelector = document.createElement('div');
            rangeSelector.id = 'custom-range-selector';
            rangeSelector.style.margin = '20px auto';
            rangeSelector.style.maxWidth = '800px';
            rangeSelector.style.padding = '15px';
            rangeSelector.style.textAlign = 'center';
            rangeSelector.style.fontFamily = "'Inter', system-ui, sans-serif";
            rangeSelector.style.backgroundColor = "#ffffff";
            rangeSelector.style.border = "1px solid #e1e1e1";
            rangeSelector.style.borderRadius = "4px";
            rangeSelector.style.boxShadow = "0 1px 3px rgba(0,0,0,0.05)";
 
            rangeSelector.innerHTML = `
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px; color: #2c3e50;">
                    <span style="font-size: 14px; font-weight: 500;">View Range:</span>
                    <span id="range-display" style="font-size: 14px;"></span>
                </div>
                <div style="display: flex; align-items: center; gap: 15px;">
                    <button id="view-all" class="range-btn" style="
                        padding: 8px 14px; 
                        background-color: #ecf0f1; 
                        color: #2c3e50; 
                        border: 1px solid #bdc3c7; 
                        border-radius: 4px; 
                        cursor: pointer; 
                        font-size: 13px; 
                        flex: 0 0 auto;
                        font-weight: 500;
                        transition: all 0.2s;
                    ">Reset Zoom</button>
                    <div style="flex-grow: 1; position: relative; height: 40px;">
                        <div style="
                            position: absolute;
                            width: 100%;
                            height: 4px;
                            background-color: #e1e1e1;
                            top: 18px;
                            border-radius: 2px;
                        "></div>
                        <input type="range" id="range-start" min="0" max="100" value="0" step="1" style="
                            position: absolute; 
                            width: 100%; 
                            pointer-events: auto; 
                            opacity: 1; 
                            height: 30px; 
                            top: 5px; 
                            background: transparent;
                            z-index: 2;
                        ">
                        <input type="range" id="range-end" min="0" max="100" value="100" step="1" style="
                            position: absolute; 
                            width: 100%; 
                            pointer-events: auto; 
                            opacity: 1; 
                            height: 30px; 
                            top: 5px; 
                            background: transparent;
                            z-index: 2;
                        ">
                    </div>
                </div>
            `;
 
            // Insert range selector and stats panel
            plotContainer.appendChild(rangeSelector);
            plotContainer.appendChild(statsDiv.firstElementChild);
 
            // Handle metric checkbox interactions
            function updatePlot() {{
                // Get the selected metrics
                const selectedMetrics = [];
                document.querySelectorAll('.metric-toggle input:checked').forEach(checkbox => {{
                    selectedMetrics.push(checkbox.dataset.metric);
                }});
 
                // Check if normalized view is enabled
                const normalizedView = document.getElementById('normalized-view').checked;
 
                // Update visibility for each trace
                const visibility = Array(Object.keys(traceIndices).length).fill(false);
 
                // If no metrics are selected, show only the first one
                if (selectedMetrics.length === 0 && !normalizedView) {{
                    const firstMetric = Object.keys(metricsData)[0];
                    visibility[traceIndices[`${{firstMetric}}_raw`]] = true;
 
                    // Update y-axis range and title for this single metric
                    Plotly.relayout(document.getElementsByClassName('js-plotly-plot')[0], {{
                        'yaxis.range': yRanges[firstMetric],
                        'yaxis.title.text': firstMetric,
                        'title.text': `<b>${{firstMetric}} over Checkpoints</b>`
                    }});
 
                    // Update stats panel visibility
                    updateStatsVisibility([firstMetric]);
                    return;
                }}
 
                // If normalized view is enabled, show normalized traces
                if (normalizedView) {{
                    for (const metric of selectedMetrics) {{
                        visibility[traceIndices[`${{metric}}_norm`]] = true;
                    }}
 
                    // Update y-axis for normalized view
                    Plotly.relayout(document.getElementsByClassName('js-plotly-plot')[0], {{
                        'yaxis.range': [0, 1],
                        'yaxis.title.text': 'Normalized Value (0-1)',
                        'title.text': '<b>Multiple Metrics over Checkpoints (Normalized)</b>'
                    }});
                }} else {{
                    // Show raw values for selected metrics
                    for (const metric of selectedMetrics) {{
                        visibility[traceIndices[`${{metric}}_raw`]] = true;
 
                        // Also show MA if it exists
                        if (traceIndices[`${{metric}}_ma`] !== undefined) {{
                            // Moving averages stay hidden by default
                            // visibility[traceIndices[`${{metric}}_ma`]] = true;
                        }}
                    }}
 
                    // If only one metric is selected, use its y-axis range
                    if (selectedMetrics.length === 1) {{
                        const metric = selectedMetrics[0];
                        Plotly.relayout(document.getElementsByClassName('js-plotly-plot')[0], {{
                            'yaxis.range': yRanges[metric],
                            'yaxis.title.text': metric,
                            'title.text': `<b>${{metric}} over Checkpoints</b>`
                        }});
                    }} else {{
                        // For multiple metrics, use auto range
                        Plotly.relayout(document.getElementsByClassName('js-plotly-plot')[0], {{
                            'yaxis.autorange': true,
                            'yaxis.title.text': 'Metric Value',
                            'title.text': '<b>Multiple Metrics over Checkpoints</b>'
                        }});
                    }}
                }}
 
                // Update trace visibility
                Plotly.restyle(document.getElementsByClassName('js-plotly-plot')[0], {{
                    'visible': visibility
                }});
 
                // Update stats panel visibility
                updateStatsVisibility(selectedMetrics);
            }}
 
            // Update stats panel visibility
            function updateStatsVisibility(visibleMetrics) {{
                // Hide all stats panels first
                document.querySelectorAll('.metric-stats').forEach(panel => {{
                    panel.style.display = 'none';
                }});
 
                // Show only the selected ones
                for (const metric of visibleMetrics) {{
                    const panel = document.getElementById(`stats-${{metric.replace(' ', '-').toLowerCase()}}`);
                    if (panel) {{
                        panel.style.display = '';
                    }}
                }}
            }}
 
            // Add event listeners to checkboxes
            document.querySelectorAll('.metric-toggle input').forEach(checkbox => {{
                checkbox.addEventListener('change', updatePlot);
            }});
 
            // Add event listener to normalized view checkbox
            document.getElementById('normalized-view').addEventListener('change', function() {{
                // If normalized view is enabled, make sure at least one metric is selected
                if (this.checked) {{
                    const anyChecked = Array.from(document.querySelectorAll('.metric-toggle input')).some(cb => cb.checked);
                    if (!anyChecked) {{
                        // If no metrics are selected, select all
                        document.querySelectorAll('.metric-toggle input').forEach(cb => {{
                            cb.checked = true;
                        }});
                    }}
                }}
                updatePlot();
            }});
 
            // Add event listeners to "All" and "None" buttons
            document.getElementById('select-all').addEventListener('click', function() {{
                document.querySelectorAll('.metric-toggle input').forEach(cb => {{
                    cb.checked = true;
                }});
                updatePlot();
            }});
 
            document.getElementById('select-none').addEventListener('click', function() {{
                document.querySelectorAll('.metric-toggle input').forEach(cb => {{
                    cb.checked = false;
                }});
                updatePlot();
            }});
 
            // Initially, only the first metric is checked
            // This is already set in the HTML
 
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
 
            // Make the sliders look good
            function styleRangeInputs() {{
                const rangeInputs = document.querySelectorAll('input[type="range"]');
                rangeInputs.forEach(input => {{
                    input.style.webkitAppearance = 'none';
                    input.style.appearance = 'none';
                    input.style.outline = 'none';
                    input.style.background = 'transparent';
 
                    // Style the thumb
                    const thumbStyles = `
                        input[type=range]::-webkit-slider-thumb {{
                            -webkit-appearance: none;
                            appearance: none;
                            width: 18px;
                            height: 18px;
                            border-radius: 50%;
                            background: #3498db;
                            cursor: pointer;
                            border: 2px solid white;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                            margin-top: -7px;
                            z-index: 3;
                            position: relative;
                        }}
 
                        input[type=range]::-moz-range-thumb {{
                            width: 18px;
                            height: 18px;
                            border-radius: 50%;
                            background: #3498db;
                            cursor: pointer;
                            border: 2px solid white;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                            z-index: 3;
                            position: relative;
                        }}
                    `;
 
                    // Add the styles to the document
                    const style = document.createElement('style');
                    style.textContent = thumbStyles;
                    document.head.appendChild(style);
                }});
            }}
 
            // Style the range inputs
            styleRangeInputs();
 
            rangeStart.addEventListener('input', function() {{
                if (parseInt(rangeStart.value) > parseInt(rangeEnd.value) - 2) {{
                    rangeStart.value = parseInt(rangeEnd.value) - 2;
                }}
                updateRangeDisplay();
            }});
 
            rangeEnd.addEventListener('input', function() {{
                if (parseInt(rangeEnd.value) < parseInt(rangeStart.value) + 2) {{
                    rangeEnd.value = parseInt(rangeStart.value) + 2;
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
 
            // Style any buttons on hover
            document.querySelectorAll('button').forEach(button => {{
                button.addEventListener('mouseover', function() {{
                    this.style.backgroundColor = '#dfe6e9';
                }});
 
                button.addEventListener('mouseout', function() {{
                    this.style.backgroundColor = '#ecf0f1';
                }});
 
                button.addEventListener('mousedown', function() {{
                    this.style.backgroundColor = '#bdc3c7';
                }});
 
                button.addEventListener('mouseup', function() {{
                    this.style.backgroundColor = '#dfe6e9';
                }});
            }});
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
 
    input[type=range]::-webkit-slider-runnable-track {
        width: 100%;
        height: 4px;
        cursor: pointer;
        background: #bdc3c7;
        border-radius: 2px;
    }
 
    input[type=range]::-moz-range-track {
        width: 100%;
        height: 4px;
        cursor: pointer;
        background: #bdc3c7;
        border-radius: 2px;
    }
 
    button.range-btn:hover {
        background-color: #dfe6e9;
    }
 
    /* Improve plot responsiveness */
    .js-plotly-plot {
        max-width: 100%;
        margin: 0 auto;
    }
 
    /* Better checkbox styling */
    input[type="checkbox"] {
        -webkit-appearance: none;
        appearance: none;
        background-color: #fff;
        margin: 0;
        font: inherit;
        color: currentColor;
        width: 16px;
        height: 16px;
        border: 1px solid #bdc3c7;
        border-radius: 3px;
        display: grid;
        place-content: center;
    }
 
    input[type="checkbox"]::before {
        content: "";
        width: 10px;
        height: 10px;
        transform: scale(0);
        transition: 120ms transform ease-in-out;
        box-shadow: inset 1em 1em #3498db;
        transform-origin: center;
        clip-path: polygon(14% 44%, 0 65%, 50% 100%, 100% 16%, 80% 0%, 43% 62%);
    }
 
    input[type="checkbox"]:checked::before {
        transform: scale(1);
    }
 
    input[type="checkbox"]:focus {
        outline: 1px solid #3498db;
    }
 
    #metric-toggles label:hover {
        background-color: rgba(236, 240, 241, 0.5);
        border-radius: 3px;
    }
    </style>
    """
 
    # ---- 1️⃣  start the cleanup server first ---------------------------------
    port, done = start_cleanup_server(output_html, timestamp, random_suffix)
    
    # ---- 2️⃣  build the JS that uses that port --------------------------------
    custom_js = f"""
    <script>
        const tempFilePath = '{str(output_html)}';
        const cleanupURL   = 'http://localhost:{port}/cleanup';
        const token        = '{timestamp}';
    
        window.addEventListener('beforeunload', () => {{
            navigator.sendBeacon(cleanupURL, JSON.stringify({{
                path: tempFilePath,
                token: token
            }}));
        }});
    </script>
    """
    
    # ---- 3️⃣  write the HTML file *now* ---------------------------------------
    raw_html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    html_with_head   = raw_html.replace("<head>", f"<head>{meta_tags}")
    final_html       = html_with_head.replace("</body>", f"{custom_js}</body>")
    
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(final_html)
    
    # ---- 4️⃣  open it in the browser ------------------------------------------
    webbrowser.open(output_html.as_uri())
    
    print("Waiting for browser to close…")
    done.wait()   
        
    # Try to export PNG as well
    try:
        png_path = output_html.with_suffix(".png")
        fig.write_image(png_path, scale=2, width=1200, height=800)
    except (ValueError, ImportError):
        print("Note: PNG export requires additional dependencies (kaleido). Install with 'pip install kaleido'.")
    
    return 
    
    
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
                        os.remove(output_html)
                        print(f"✓ deleted {output_html}")
                        cleanup_event.set()
                        threading.Thread(target=self.server.shutdown,
                                         daemon=True).start()
                except Exception as e:
                    print("cleanup error:", e)

        def log_message(self, *_):        # silence default logging
            pass

    # ---------- THESE LINES WERE ACCIDENTALLY INDENTED ----------
    base = 8000
    port = base + (hash(port_suffix) % 1000)
    for _ in range(20):
        try:
            server = socketserver.TCPServer(("127.0.0.1", port), CleanupHandler)
            break
        except OSError:
            port += 1
    else:
        raise RuntimeError("No free port for cleanup server")

    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Cleanup server on :{port}")

    return port, cleanup_event               # <-- real return