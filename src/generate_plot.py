from pathlib import Path
from typing import Sequence, Dict, List, Optional, Tuple
import datetime as dt
import numpy as np
import os
import json
import threading
import random
import sys

from utils import (
    suppress_stdout_stderr, 
    start_cleanup_server, 
    find_interesting_points, 
    calculate_trend_info,
    logger,
    register_html_for_cleanup,
    unregister_html_from_cleanup
)

os.environ['GTK_MODULES'] = ''


def plot_metric_interactive(
    checkpoint_names: Sequence[str],
    values: Sequence[float] = None,
    metric_name: str = None,
    metrics_data: Dict[str, List[float]] = None,
    many_metrics=False
) -> None:
    """Render an interactive line-plot using React and D3.js.
    
    Args:
        checkpoint_names: List of checkpoint identifiers
        values: (Optional) List of metric values for a single metric
        metric_name: (Optional) Name of the single metric
        metrics_data: (Optional) Dictionary mapping metric names to values
    
    Returns:
        None: Outputs are saved to HTML file
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
    
    # Prepare data for React component
    react_data = {
        "checkpoints": list(checkpoint_names),
        "metrics": metrics_data,
        "metricsList": list(metrics_data.keys()),
        "startSeparate": many_metrics
    }
    
    # Create output filename
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
    output_html = Path(f"metrics_over_checkpoints_{timestamp}_{random_suffix}.html").absolute()
    
    # Start cleanup server
    port, cleanup_event = start_cleanup_server(output_html, timestamp, random_suffix)
    logger.info(f"Started cleanup server on port {port} for {output_html}")
    
    # Generate the React-based HTML
    html_content = fr"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phase-Viz</title>
    
    <!-- Load React and ReactDOM from CDN -->
    <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    
    <!-- Load D3.js -->
    <script src="https://d3js.org/d3.v7.min.js"></script>
    
    <!-- Load Babel for JSX transformation -->
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    
    <!-- Modern fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            overflow-x: hidden;
            position: relative;
            min-height: 100vh;
        }}
        
        /* Subtle animated background */
        body::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 80% 50%, rgba(244, 63, 94, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 40% 20%, rgba(34, 197, 94, 0.05) 0%, transparent 50%);
            z-index: -1;
            animation: gradientShift 30s ease infinite;
        }}
        
        @keyframes gradientShift {{
            0%, 100% {{ transform: translate(0, 0) scale(1); }}
            33% {{ transform: translate(-10px, -10px) scale(1.05); }}
            66% {{ transform: translate(10px, -5px) scale(0.95); }}
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 1.5rem 2rem;
            position: relative;
            z-index: 1;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 2rem;
            position: relative;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 600;
            color: #f3f4f6;
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
        }}
        
        .header p {{
            font-size: 1.1rem;
            color: #9ca3af;
            font-weight: 400;
            transition: all 0.3s ease;
        }}
        
        .controls {{
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .metric-selector-container {{
            position: relative;
            min-width: 200px;
        }}
        
        .metric-selector-button {{
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            color: #e0e0e0;
            font-size: 0.95rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            outline: none;
            width: 100%;
            text-align: left;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .metric-selector-button:hover {{
            background: #222;
            border-color: #444;
        }}
        
        .metric-selector-dropdown {{
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            margin-top: 4px;
            max-height: 300px;
            overflow-y: auto;
            z-index: 100;
            display: none;
        }}
        
        .metric-selector-dropdown.open {{
            display: block;
        }}
        
        .metric-option {{
            padding: 0.75rem 1.25rem;
            cursor: pointer;
            transition: background 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .metric-option:hover {{
            background: #222;
        }}
        
        .metric-option input[type="checkbox"] {{
            width: 16px;
            height: 16px;
            cursor: pointer;
        }}
        
        .overlay-button {{
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            color: #e0e0e0;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .overlay-button:hover {{
            background: #222;
            border-color: #444;
        }}
        
        .overlay-button.active {{
            background: #6366f1;
            border-color: #6366f1;
            color: white;
        }}
        
        .phase-button {{
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            color: #e0e0e0;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 0.95rem;
        }}
        
        .phase-button:hover {{
            background: #222;
            border-color: #444;
        }}
        
        .phase-button.active {{
            background: #f59e0b;
            border-color: #f59e0b;
            color: white;
        }}
        
        .action-button {{
            background: #6366f1;
            border: none;
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 0.95rem;
        }}
        
        .action-button:hover {{
            background: #4f46e5;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }}
        
        .action-button:active {{
            transform: translateY(0);
        }}
        
        .charts-grid {{
            display: grid;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        .chart-wrapper {{
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            min-height: 550px; /* Ensure minimum height */
        }}
        
        .chart-wrapper:hover {{
            border-color: #444;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}
        
        .chart-wrapper.expanded {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 90vw;
            height: 90vh;
            z-index: 1000;
            cursor: default;
        }}
        
        .chart-header {{
            padding: 1rem 1.5rem;
            border-bottom: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        
        .chart-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #f3f4f6;
        }}
        
        .expand-icon {{
            width: 20px;
            height: 20px;
            opacity: 0.6;
            transition: opacity 0.2s ease;
        }}
        
        .chart-wrapper:hover .expand-icon {{
            opacity: 1;
        }}
        
        .chart-container {{
            padding: 1.25rem;
            position: relative;
        }}
        
        .chart {{
            width: 100%;
            display: block;
        }}
        
        .stats-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }}
        
        .stat-card {{
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 1.25rem;
            transition: all 0.2s ease;
        }}
        
        .stat-card:hover {{
            border-color: #444;
            transform: translateY(-2px);
        }}
        
        .stat-card h3 {{
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: #f3f4f6;
        }}
        
        .stat-card .metric-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }}
        
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
        }}
        
        .stat-item {{
            display: flex;
            justify-content: space-between;
            font-size: 0.875rem;
        }}
        
        .stat-label {{
            color: #9ca3af;
        }}
        
        .stat-value {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
            color: #e0e0e0;
        }}
        
        .phase-info {{
            margin-top: 1rem;
            padding: 0.75rem;
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 6px;
            font-size: 0.875rem;
        }}
        
        .phase-info strong {{
            color: #f59e0b;
        }}
        
        .loading {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 400px;
            gap: 1rem;
        }}
        
        .loading-spinner {{
            width: 40px;
            height: 40px;
            border: 3px solid rgba(99, 102, 241, 0.2);
            border-top-color: #6366f1;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        
        .tooltip {{
            position: absolute;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 0.75rem;
            pointer-events: none;
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.2s ease;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }}
        
        .tooltip.visible {{
            opacity: 1;
        }}
        
        .tooltip-title {{
            font-weight: 600;
            margin-bottom: 0.25rem;
            color: #f3f4f6;
        }}
        
        .tooltip-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem;
            color: #9ca3af;
        }}
        
        .legend {{
            display: flex;
            gap: 1.5rem;
            margin-top: 1rem;
            justify-content: center;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
            color: #9ca3af;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            transition: all 0.2s ease;
        }}
        
        .legend-item:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        
        .legend-color {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
        }}
        
        .overlay {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            z-index: 999;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
        }}
        
        .overlay.visible {{
            opacity: 1;
            pointer-events: all;
        }}
        
        /* Grid layouts based on number of metrics */
        .charts-grid.single {{ grid-template-columns: 1fr; }}
        .charts-grid.double {{ 
            grid-template-columns: 1fr; 
            grid-template-rows: repeat(2, minmax(500px, 1fr));
            gap: 2rem;
        }}
        .charts-grid.triple {{ grid-template-columns: repeat(3, 1fr); }}
        .charts-grid.quad {{ grid-template-columns: repeat(2, 1fr); }}
        .charts-grid.many {{ grid-template-columns: repeat(3, 1fr); }}
        
        @media (max-width: 1200px) {{
            .charts-grid.triple {{ grid-template-columns: repeat(2, 1fr); }}
            .charts-grid.many {{ grid-template-columns: repeat(2, 1fr); }}
        }}
        
        @media (max-width: 768px) {{
            .charts-grid.double {{ grid-template-columns: 1fr; }}
            .charts-grid.triple {{ grid-template-columns: 1fr; }}
            .charts-grid.quad {{ grid-template-columns: 1fr; }}
            .charts-grid.many {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div id="root"></div>
    
    <script type="text/babel">
        const {{ useState, useEffect, useRef, useMemo }} = React;
        
        // Data passed from Python
        const DATA = {json.dumps(react_data)};
        const START_SEPARATE= DATA.startSeparate;
        const CLEANUP_PORT = {port};
        
        // Color palette for metrics
        const COLOR_PALETTE = [
            '#6366f1', '#ec4899', '#10b981', '#f59e0b', '#3b82f6',
            '#8b5cf6', '#ef4444', '#14b8a6', '#f97316', '#06b6d4'
        ];
        
        // Create color mapping
        const metricColors = {{}};
        DATA.metricsList.forEach((metric, i) => {{
            metricColors[metric] = COLOR_PALETTE[i % COLOR_PALETTE.length];
        }});
        
        // Phase transition detection threshold
        const PHASE_TRANSITION_THRESHOLD = 0.2; // 20% change
        
        function Chart({{ metric, data, isExpanded, onToggleExpand, showPhaseTransitions }}) {{
            const svgRef = useRef(null);
            const containerRef = useRef(null);
            const [hoveredPoint, setHoveredPoint] = useState(null);
            const [dimensions, setDimensions] = useState({{ width: 0, height: 0 }});
            
            // Handle resize
            useEffect(() => {{
                const handleResize = () => {{
                    if (containerRef.current) {{
                        const {{ clientWidth, clientHeight }} = containerRef.current;
                        setDimensions({{ width: clientWidth, height: clientHeight }});
                    }}
                }};
                
                handleResize();
                window.addEventListener('resize', handleResize);
                
                // Small delay to ensure DOM is ready
                const timer = setTimeout(handleResize, 100);
                
                return () => {{
                    window.removeEventListener('resize', handleResize);
                    clearTimeout(timer);
                }};
            }}, [isExpanded]);
            
            // Calculate statistics
            const stats = useMemo(() => {{
                const values = data.filter(v => !isNaN(v));
                if (values.length === 0) return null;
                
                const min = Math.min(...values);
                const max = Math.max(...values);
                const mean = values.reduce((a, b) => a + b, 0) / values.length;
                const start = values[0];
                const end = values[values.length - 1];
                const change = end - start;
                const changePercent = start !== 0 ? (change / Math.abs(start)) * 100 : 0;
                
                return {{
                    min, max, mean, start, end, change, changePercent,
                    minIndex: data.indexOf(min),
                    maxIndex: data.indexOf(max)
                }};
            }}, [data]);
            
            // Calculate phase transitions
            const phaseTransitions = useMemo(() => {{
                const transitions = [];
                for (let i = 1; i < data.length; i++) {{
                    if (!isNaN(data[i]) && !isNaN(data[i-1])) {{
                        const change = Math.abs(data[i] - data[i-1]);
                        const baseValue = Math.abs(data[i-1]) || 1;
                        const relativeChange = change / baseValue;
                        
                        if (relativeChange > PHASE_TRANSITION_THRESHOLD) {{
                            transitions.push({{
                                index: i,
                                change: relativeChange,
                                from: data[i-1],
                                to: data[i]
                            }});
                        }}
                    }}
                }}
                return transitions;
            }}, [data]);
            
            // Render chart with D3
            useEffect(() => {{
                if (!svgRef.current || !containerRef.current || !stats) return;
                
                const container = containerRef.current;
                const width = container.clientWidth;
                // Dynamic height based on number of selected metrics
                const baseHeight = window.innerHeight - 400; // Account for header, controls, stats
                const numMetrics = document.querySelectorAll('.chart-wrapper').length;
                let height;
                
                if (isExpanded) {{
                    height = 600;
                }} else if (numMetrics === 1) {{
                    height = Math.max(500, baseHeight * 0.8);
                }} else if (numMetrics === 2) {{
                    // Special handling for 2 metrics side by side
                    height = Math.max(450, Math.min(600, (baseHeight - 100) * 0.45));
                }} else {{
                    height = 350;
                }}
                
                const margin = {{ top: 20, right: 80, bottom: 140, left: 100 }};
                const innerWidth = width - margin.left - margin.right;
                const innerHeight = height - margin.top - margin.bottom;
                
                // Clear previous chart
                d3.select(svgRef.current).selectAll('*').remove();
                
                const svg = d3.select(svgRef.current)
                    .attr('width', width)
                    .attr('height', height);
                
                const g = svg.append('g')
                    .attr('transform', `translate(${{margin.left}},${{margin.top}})`);
                
                // Create scales
                const xScale = d3.scaleLinear()
                    .domain([0, DATA.checkpoints.length - 1])
                    .range([0, innerWidth]);
                
                const validValues = data.filter(v => !isNaN(v));
                const yPadding = (stats.max - stats.min) * 0.1 || 0.1;
                const yScale = d3.scaleLinear()
                    .domain([stats.min - yPadding, stats.max + yPadding])
                    .range([innerHeight, 0]);
                
                // Add phase transition indicators
                if (showPhaseTransitions && phaseTransitions.length > 0) {{
                    phaseTransitions.forEach(transition => {{
                        const x = xScale(transition.index - 0.5);
                        const width = xScale(1) - xScale(0);
                        
                        g.append('rect')
                            .attr('x', x)
                            .attr('y', 0)
                            .attr('width', width)
                            .attr('height', innerHeight)
                            .style('fill', 'rgba(245, 158, 11, 0.1)')
                            .style('stroke', '#f59e0b')
                            .style('stroke-width', 1)
                            .style('stroke-dasharray', '4,2');
                    }});
                }}
                
                // Grid lines
                g.append('g')
                    .attr('class', 'grid')
                    .attr('transform', `translate(0,${{innerHeight}})`)
                    .call(d3.axisBottom(xScale)
                        .tickSize(-innerHeight)
                        .tickFormat('')
                        .ticks(10))
                    .style('stroke-dasharray', '3,3')
                    .style('opacity', 0.3);
                
                g.append('g')
                    .attr('class', 'grid')
                    .call(d3.axisLeft(yScale)
                        .tickSize(-innerWidth)
                        .tickFormat('')
                        .ticks(6))
                    .style('stroke-dasharray', '3,3')
                    .style('opacity', 0.3);
                
                // Axes
                const xAxis = g.append('g')
                    .attr('transform', `translate(0,${{innerHeight}})`)
                    .call(d3.axisBottom(xScale)
                        .tickFormat(i => {{
                            const checkpoint = DATA.checkpoints[i];
                            if (!checkpoint) return '';
                            
                            // Use simple checkpoint numbers
                            return `CP${{i + 1}}`;
                        }}));
                
                xAxis.selectAll('text')
                    .style('text-anchor', 'middle')
                    .style('fill', '#9ca3af')
                    .style('font-size', '11px')
                    .style('cursor', 'pointer')
                    .each(function(d, i) {{
                        const text = d3.select(this);
                        const checkpointName = DATA.checkpoints[d];
                        if (checkpointName) {{
                            // Add hover effect to show full name
                            text.on('mouseenter', function() {{
                                text.text(checkpointName)
                                    .style('font-weight', '600')
                                    .style('fill', '#f3f4f6');
                                
                                // Adjust position if text is too long
                                const bbox = this.getBBox();
                                if (bbox.width > 100) {{
                                    text.style('text-anchor', 'end')
                                        .attr('dx', '-.8em')
                                        .attr('dy', '.15em')
                                        .attr('transform', 'rotate(-45)');
                                }}
                            }})
                            .on('mouseleave', function() {{
                                text.text(`CP${{d + 1}}`)
                                    .style('font-weight', 'normal')
                                    .style('fill', '#9ca3af')
                                    .style('text-anchor', 'middle')
                                    .attr('dx', null)
                                    .attr('dy', null)
                                    .attr('transform', null);
                            }});
                            
                            // Add title for tooltip
                            text.append('title').text(checkpointName);
                        }}
                    }});
                
                xAxis.select('.domain').style('stroke', '#333');
                xAxis.selectAll('.tick line').style('stroke', '#333');
                
                // Add x-axis label
                g.append('text')
                    .attr('x', innerWidth / 2)
                    .attr('y', innerHeight + margin.bottom - 10)
                    .style('text-anchor', 'middle')
                    .style('fill', '#9ca3af')
                    .style('font-size', '12px')
                    .text('Checkpoint');
                
                const yAxis = g.append('g')
                    .call(d3.axisLeft(yScale)
                        .tickFormat(d => d.toFixed(4)));
                
                yAxis.selectAll('text')
                    .style('fill', '#9ca3af')
                    .style('font-size', isExpanded ? '12px' : '10px');
                
                yAxis.select('.domain').style('stroke', '#333');
                yAxis.selectAll('.tick line').style('stroke', '#333');
                
                // Line generator
                const line = d3.line()
                    .x((d, i) => xScale(i))
                    .y(d => yScale(d))
                    .curve(d3.curveMonotoneX)
                    .defined(d => !isNaN(d));
                
                // Add gradient
                const gradient = svg.append('defs')
                    .append('linearGradient')
                    .attr('id', `gradient-${{metric.replace(/\s+/g, '-')}}`)
                    .attr('gradientUnits', 'userSpaceOnUse')
                    .attr('x1', 0).attr('y1', yScale(stats.max))
                    .attr('x2', 0).attr('y2', yScale(stats.min));
                
                gradient.append('stop')
                    .attr('offset', '0%')
                    .attr('stop-color', metricColors[metric])
                    .attr('stop-opacity', 0.8);
                
                gradient.append('stop')
                    .attr('offset', '100%')
                    .attr('stop-color', metricColors[metric])
                    .attr('stop-opacity', 0.1);
                
                // Area under curve
                const area = d3.area()
                    .x((d, i) => xScale(i))
                    .y0(innerHeight)
                    .y1(d => yScale(d))
                    .curve(d3.curveMonotoneX)
                    .defined(d => !isNaN(d));
                
                g.append('path')
                    .datum(data)
                    .attr('fill', `url(#gradient-${{metric.replace(/\s+/g, '-')}})`)
                    .attr('d', area)
                    .attr('opacity', 0.3);
                
                // Draw line
                const path = g.append('path')
                    .datum(data)
                    .attr('fill', 'none')
                    .attr('stroke', metricColors[metric])
                    .attr('stroke-width', 2)
                    .attr('d', line);
                
                // Animate line
                const totalLength = path.node().getTotalLength();
                path
                    .attr('stroke-dasharray', totalLength + ' ' + totalLength)
                    .attr('stroke-dashoffset', totalLength)
                    .transition()
                    .duration(1000)
                    .ease(d3.easeQuadInOut)
                    .attr('stroke-dashoffset', 0);
                
                // Add data points
                const points = g.selectAll('.point')
                    .data(data)
                    .enter()
                    .filter(d => !isNaN(d))
                    .append('circle')
                    .attr('cx', (d, i) => xScale(i))
                    .attr('cy', d => yScale(d))
                    .attr('r', 0)
                    .attr('fill', metricColors[metric])
                    .style('cursor', 'pointer');
                
                points
                    .transition()
                    .duration(1000)
                    .delay((d, i) => i * 30)
                    .attr('r', 3);
                
                // Add max/min highlights
                if (stats.maxIndex >= 0) {{
                    g.append('circle')
                        .attr('cx', xScale(stats.maxIndex))
                        .attr('cy', yScale(data[stats.maxIndex]))
                        .attr('r', 0)
                        .attr('fill', '#3b82f6')
                        .attr('stroke', '#fff')
                        .attr('stroke-width', 2)
                        .transition()
                        .duration(1000)
                        .delay(1000)
                        .attr('r', 6);
                }}
                
                if (stats.minIndex >= 0) {{
                    g.append('circle')
                        .attr('cx', xScale(stats.minIndex))
                        .attr('cy', yScale(data[stats.minIndex]))
                        .attr('r', 0)
                        .attr('fill', '#ef4444')
                        .attr('stroke', '#fff')
                        .attr('stroke-width', 2)
                        .transition()
                        .duration(1000)
                        .delay(1000)
                        .attr('r', 6);
                }}
                
                // Hover effects
                points
                    .on('mouseenter', function(event, d) {{
                        const i = data.indexOf(d);
                        setHoveredPoint({{
                            checkpoint: DATA.checkpoints[i],
                            value: d,
                            x: event.pageX,
                            y: event.pageY
                        }});
                        
                        d3.select(this)
                            .transition()
                            .duration(200)
                            .attr('r', 5);
                    }})
                    .on('mouseleave', function() {{
                        setHoveredPoint(null);
                        
                        d3.select(this)
                            .transition()
                            .duration(200)
                            .attr('r', 3);
                    }});
                    
            }}, [data, metric, stats, isExpanded, showPhaseTransitions, phaseTransitions, dimensions]);
            
            return (
                <div 
                    className={{`chart-wrapper ${{isExpanded ? 'expanded' : ''}}`}}
                    onClick={{() => {{
                        if (!isExpanded && onToggleExpand) {{
                            onToggleExpand();
                        }}
                    }}}}
                >
                    <div className="chart-header">
                        <h3 className="chart-title">{{metric}}</h3>
                        {{!isExpanded && (
                            <svg className="expand-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" 
                                    d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                            </svg>
                        )}}
                    </div>
                    <div className="chart-container" ref={{containerRef}}>
                        <svg ref={{svgRef}} className="chart"></svg>
                        {{stats && (
                            <div className="legend">
                                <div className="legend-item">
                                    <div className="legend-color" style={{{{ backgroundColor: metricColors[metric] }}}}></div>
                                    <span>{{metric}}</span>
                                </div>
                                <div className="legend-item">
                                    <div className="legend-color" style={{{{ backgroundColor: '#3b82f6' }}}}></div>
                                    <span>Maximum</span>
                                </div>
                                <div className="legend-item">
                                    <div className="legend-color" style={{{{ backgroundColor: '#ef4444' }}}}></div>
                                    <span>Minimum</span>
                                </div>
                            </div>
                        )}}
                        {{showPhaseTransitions && phaseTransitions.length > 0 && (
                            <div className="phase-info">
                                <strong>Phase transitions detected at:</strong>
                                <ul style={{{{ marginTop: '0.5rem', paddingLeft: '1.5rem' }}}}>
                                    {{phaseTransitions.map((t, i) => (
                                        <li key={{i}}>
                                            Checkpoint {{DATA.checkpoints[t.index]}}: {{(t.change * 100).toFixed(1)}}% change
                                        </li>
                                    ))}}
                                </ul>
                            </div>
                        )}}
                    </div>
                    {{hoveredPoint && (
                        <div 
                            className="tooltip visible"
                            style={{{{ 
                                left: `${{hoveredPoint.x + 10}}px`, 
                                top: `${{hoveredPoint.y - 10}}px` 
                            }}}}
                        >
                            <div className="tooltip-title">{{hoveredPoint.checkpoint}}</div>
                            <div className="tooltip-value">
                                Value: {{hoveredPoint.value.toFixed(6)}}
                            </div>
                            <div className="tooltip-value" style={{{{ fontSize: '0.75rem', marginTop: '0.25rem', color: '#9ca3af' }}}}>
                                (Checkpoint {{DATA.checkpoints.indexOf(hoveredPoint.checkpoint) + 1}})
                            </div>
                        </div>
                    )}}
                </div>
            );
        }}
        
        function OverlayChart({{ metrics, data, showPhaseTransitions, onMetricClick }}) {{
            const svgRef = useRef(null);
            const containerRef = useRef(null);
            const [hoveredLine, setHoveredLine] = useState(null);
            const [hoveredPoint, setHoveredPoint] = useState(null);
            
            useEffect(() => {{
                if (!svgRef.current || !containerRef.current) return;
                
                const container = containerRef.current;
                const width = container.clientWidth;
                // Use most of the available screen height
                const baseHeight = window.innerHeight - 400;
                const height = Math.max(600, baseHeight * 0.8);
                const margin = {{ top: 20, right: 250, bottom: 140, left: 100 }};
                const innerWidth = width - margin.left - margin.right;
                const innerHeight = height - margin.top - margin.bottom;
                
                // Clear previous chart
                d3.select(svgRef.current).selectAll('*').remove();
                
                const svg = d3.select(svgRef.current)
                    .attr('width', width)
                    .attr('height', height);
                
                const g = svg.append('g')
                    .attr('transform', `translate(${{margin.left}},${{margin.top}})`);
                
                // Create scales
                const xScale = d3.scaleLinear()
                    .domain([0, DATA.checkpoints.length - 1])
                    .range([0, innerWidth]);
                
                // For overlay, normalize all metrics to 0-1
                const yScale = d3.scaleLinear()
                    .domain([0, 1])
                    .range([innerHeight, 0]);
                
                // Normalize data for each metric
                const normalizedData = {{}};
                metrics.forEach(metric => {{
                    const values = data[metric];
                    const validValues = values.filter(v => !isNaN(v));
                    const min = Math.min(...validValues);
                    const max = Math.max(...validValues);
                    const range = max - min || 1;
                    
                    normalizedData[metric] = values.map(v => 
                        isNaN(v) ? v : (v - min) / range
                    );
                }});
                
                // Add phase transition indicators for all metrics
                if (showPhaseTransitions) {{
                    const allTransitions = new Set();
                    metrics.forEach(metric => {{
                        const values = data[metric];
                        for (let i = 1; i < values.length; i++) {{
                            if (!isNaN(values[i]) && !isNaN(values[i-1])) {{
                                const change = Math.abs(values[i] - values[i-1]);
                                const baseValue = Math.abs(values[i-1]) || 1;
                                const relativeChange = change / baseValue;
                                
                                if (relativeChange > PHASE_TRANSITION_THRESHOLD) {{
                                    allTransitions.add(i);
                                }}
                            }}
                        }}
                    }});
                    
                    allTransitions.forEach(index => {{
                        const x = xScale(index - 0.5);
                        const width = xScale(1) - xScale(0);
                        
                        g.append('rect')
                            .attr('x', x)
                            .attr('y', 0)
                            .attr('width', width)
                            .attr('height', innerHeight)
                            .style('fill', 'rgba(245, 158, 11, 0.1)')
                            .style('stroke', '#f59e0b')
                            .style('stroke-width', 1)
                            .style('stroke-dasharray', '4,2');
                    }});
                }}
                
                // Grid lines
                g.append('g')
                    .attr('class', 'grid')
                    .attr('transform', `translate(0,${{innerHeight}})`)
                    .call(d3.axisBottom(xScale)
                        .tickSize(-innerHeight)
                        .tickFormat('')
                        .ticks(10))
                    .style('stroke-dasharray', '3,3')
                    .style('opacity', 0.3);
                
                g.append('g')
                    .attr('class', 'grid')
                    .call(d3.axisLeft(yScale)
                        .tickSize(-innerWidth)
                        .tickFormat('')
                        .ticks(5))
                    .style('stroke-dasharray', '3,3')
                    .style('opacity', 0.3);
                
                // Axes
                const xAxis = g.append('g')
                    .attr('transform', `translate(0,${{innerHeight}})`)
                    .call(d3.axisBottom(xScale)
                        .tickFormat(i => {{
                            const checkpoint = DATA.checkpoints[i];
                            if (!checkpoint) return '';
                            return `CP${{i + 1}}`;
                        }}));
                
                xAxis.selectAll('text')
                    .style('text-anchor', 'middle')
                    .style('fill', '#9ca3af')
                    .style('font-size', '10px')
                    .style('cursor', 'pointer')
                    .each(function(d, i) {{
                        const text = d3.select(this);
                        const checkpointName = DATA.checkpoints[d];
                        if (checkpointName) {{
                            // Add hover effect to show full name
                            text.on('mouseenter', function() {{
                                text.text(checkpointName)
                                    .style('font-weight', '600')
                                    .style('fill', '#f3f4f6');
                                
                                // Adjust position if text is too long
                                const bbox = this.getBBox();
                                if (bbox.width > 100) {{
                                    text.style('text-anchor', 'end')
                                        .attr('dx', '-.8em')
                                        .attr('dy', '.15em')
                                        .attr('transform', 'rotate(-45)');
                                }}
                            }})
                            .on('mouseleave', function() {{
                                text.text(`CP${{d + 1}}`)
                                    .style('font-weight', 'normal')
                                    .style('fill', '#9ca3af')
                                    .style('text-anchor', 'middle')
                                    .attr('dx', null)
                                    .attr('dy', null)
                                    .attr('transform', null);
                            }});
                            
                            // Add title for tooltip
                            text.append('title').text(checkpointName);
                        }}
                    }});
                
                xAxis.select('.domain').style('stroke', '#333');
                xAxis.selectAll('.tick line').style('stroke', '#333');
                
                // Add x-axis label
                g.append('text')
                    .attr('x', innerWidth / 2)
                    .attr('y', innerHeight + margin.bottom - 10)
                    .style('text-anchor', 'middle')
                    .style('fill', '#9ca3af')
                    .style('font-size', '12px')
                    .text('Checkpoint');
                
                const yAxis = g.append('g')
                    .call(d3.axisLeft(yScale)
                    .ticks(5)
                    .tickFormat(d3.format('.2f'))
                    );              
                yAxis.selectAll('text')
                    .style('fill', '#9ca3af')
                    .style('font-size', '11px');
                
                yAxis.select('.domain').style('stroke', '#333');
                yAxis.selectAll('.tick line').style('stroke', '#333');
                
                // Y-axis label
                g.append('text')
                    .attr('transform', 'rotate(-90)')
                    .attr('y', -40)
                    .attr('x', -innerHeight / 2)
                    .style('text-anchor', 'middle')
                    .style('fill', '#9ca3af')
                    .style('font-size', '12px')
                    .text('Normalized Value');
                
                // Line generator
                const line = d3.line()
                    .x((d, i) => xScale(i))
                    .y(d => yScale(d))
                    .curve(d3.curveMonotoneX)
                    .defined(d => !isNaN(d));
                
                // Draw lines for each metric
                metrics.forEach((metric, idx) => {{
                    const path = g.append('path')
                        .datum(normalizedData[metric])
                        .attr('fill', 'none')
                        .attr('stroke', metricColors[metric])
                        .attr('stroke-width', 2)
                        .attr('d', line)
                        .style('opacity', hoveredLine === null || hoveredLine === metric ? 1 : 0.3)
                        .style('transition', 'opacity 0.3s ease');
                    
                    // Animate line drawing
                    const totalLength = path.node().getTotalLength();
                    path
                        .attr('stroke-dasharray', totalLength + ' ' + totalLength)
                        .attr('stroke-dashoffset', totalLength)
                        .transition()
                        .duration(1000)
                        .delay(idx * 100)
                        .ease(d3.easeQuadInOut)
                        .attr('stroke-dashoffset', 0);
                    
                    // Invisible wider path for hover detection
                    g.append('path')
                        .datum(normalizedData[metric])
                        .attr('fill', 'none')
                        .attr('stroke', 'transparent')
                        .attr('stroke-width', 20)
                        .attr('d', line)
                        .style('cursor', 'pointer')
                        .on('click', function(event) {{
                            event.stopPropagation();
                            if (onMetricClick) {{
                                onMetricClick(metric);
                            }}
                        }})
                        .on('mouseenter', function() {{
                            setHoveredLine(metric);
                            // Update all line opacities
                            g.selectAll('path')
                                .filter(function() {{
                                    return d3.select(this).attr('stroke') !== 'transparent';
                                }})
                                .style('opacity', function() {{
                                    const color = d3.select(this).attr('stroke');
                                    return color === metricColors[metric] ? 1 : 0.3;
                                }});
                        }})
                        .on('mouseleave', function() {{
                            setHoveredLine(null);
                            // Reset all line opacities
                            g.selectAll('path')
                                .filter(function() {{
                                    return d3.select(this).attr('stroke') !== 'transparent';
                                }})
                                .style('opacity', 1);
                        }});
                    
                    // Add data points
                    const points = g.selectAll(`.point-${{metric.replace(/\s+/g, '-')}}`)
                        .data(normalizedData[metric])
                        .enter()
                        .filter(d => !isNaN(d))
                        .append('circle')
                        .attr('class', `point-${{metric.replace(/\s+/g, '-')}}`)
                        .attr('cx', (d, i) => xScale(i))
                        .attr('cy', d => yScale(d))
                        .attr('r', 0)
                        .attr('fill', metricColors[metric])
                        .style('cursor', 'pointer')
                        .style('opacity', hoveredLine === null || hoveredLine === metric ? 1 : 0.3);
                    
                    points
                        .transition()
                        .duration(1000)
                        .delay((d, i) => idx * 100 + i * 20)
                        .attr('r', 2);
                    
                    points
                        .on('mouseenter', function(event, d) {{
                            const i = normalizedData[metric].indexOf(d);
                            const originalValue = data[metric][i];
                            setHoveredPoint({{
                                metric,
                                checkpoint: DATA.checkpoints[i],
                                value: originalValue,
                                normalizedValue: d,
                                x: event.pageX,
                                y: event.pageY
                            }});
                            
                            d3.select(this)
                                .transition()
                                .duration(200)
                                .attr('r', 4);
                        }})
                        .on('mouseleave', function() {{
                            setHoveredPoint(null);
                            
                            d3.select(this)
                                .transition()
                                .duration(200)
                                .attr('r', 2);
                        }});
                }});
                
                // Legend
                const legend = svg.append('g')
                    .attr('transform', `translate(${{width - margin.right + 20}}, ${{margin.top}})`);
                
                metrics.forEach((metric, i) => {{
                    const legendItem = legend.append('g')
                        .attr('transform', `translate(0, ${{i * 25}})`)
                        .style('cursor', 'pointer')
                        .on('click', function(event) {{
                            event.stopPropagation();
                            if (onMetricClick) {{
                                onMetricClick(metric);
                            }}
                        }})
                        .on('mouseenter', function() {{
                            setHoveredLine(metric);
                            g.selectAll('path')
                                .filter(function() {{
                                    return d3.select(this).attr('stroke') !== 'transparent';
                                }})
                                .style('opacity', function() {{
                                    const color = d3.select(this).attr('stroke');
                                    return color === metricColors[metric] ? 1 : 0.3;
                                }});
                        }})
                        .on('mouseleave', function() {{
                            setHoveredLine(null);
                            g.selectAll('path')
                                .filter(function() {{
                                    return d3.select(this).attr('stroke') !== 'transparent';
                                }})
                                .style('opacity', 1);
                        }});
                    
                    legendItem.append('circle')
                        .attr('r', 6)
                        .attr('fill', metricColors[metric]);
                    
                    legendItem.append('text')
                        .attr('x', 12)
                        .attr('y', 4)
                        .style('fill', '#9ca3af')
                        .style('font-size', '11px')
                        .text(() => {{
                            // Truncate long metric names
                            const maxLength = 25;
                            if (metric.length > maxLength) {{
                                return metric.substring(0, maxLength - 3) + '...';
                            }}
                            return metric;
                        }})
                        .append('title')
                        .text(metric); // Full name on hover
                }});
                
            }}, [metrics, data, showPhaseTransitions]);
            
            return (
                <div className="chart-wrapper">
                    <div className="chart-header">
                        <h3 className="chart-title"> {{metrics.length === DATA.metricsList.length 
                ? 'All Metrics (Normalized)' 
                : `Viewing ${{metrics.length}} Normalized Metrics`}}</h3>
                        <span style={{{{ fontSize: '0.75rem', color: '#6b7280', fontStyle: 'italic' }}}}>
                            Click any line to view details
                        </span>
                    </div>
                    <div className="chart-container" ref={{containerRef}}>
                        <svg ref={{svgRef}} className="chart"></svg>
                    </div>
                    {{hoveredPoint && (
                        <div 
                            className="tooltip visible"
                            style={{{{ 
                                left: `${{hoveredPoint.x + 10}}px`, 
                                top: `${{hoveredPoint.y - 10}}px` 
                            }}}}
                        >
                            <div className="tooltip-title">{{hoveredPoint.metric}}</div>
                            <div className="tooltip-value">
                                {{hoveredPoint.checkpoint}}: {{hoveredPoint.value.toFixed(6)}}
                                <br />
                                <span style={{{{ fontSize: '0.8em', color: '#666' }}}}>
                                    Normalized: {{hoveredPoint.normalizedValue.toFixed(3)}}
                                </span>
                            </div>
                            <div className="tooltip-value" style={{{{ fontSize: '0.75rem', marginTop: '0.25rem', color: '#9ca3af' }}}}>
                                (Checkpoint {{DATA.checkpoints.indexOf(hoveredPoint.checkpoint) + 1}})
                            </div>
                        </div>
                    )}}
                </div>
            );
        }}
        
        function MetricsVisualization() {{
            const [selectedMetrics, setSelectedMetrics] = useState(DATA.metricsList);
            const [expandedMetric, setExpandedMetric] = useState(null);
            const [subtitle, setSubtitle] = useState('Visualizing training dynamics across checkpoints');
            const [dropdownOpen, setDropdownOpen] = useState(false);
            const [overlayMode, setOverlayMode] = useState(!START_SEPARATE); // Start with overlay for "All"
            const [showPhaseTransitions, setShowPhaseTransitions] = useState(false);
            const [fullscreenFromOverlay, setFullscreenFromOverlay] = useState(null); // Track fullscreen from overlay
            const [previousOverlayMetrics, setPreviousOverlayMetrics] = useState(null); // Remember overlay state
            
            // Calculate statistics for all metrics
            const allStats = useMemo(() => {{
                const result = {{}};
                DATA.metricsList.forEach(metric => {{
                    const values = DATA.metrics[metric].filter(v => !isNaN(v));
                    if (values.length > 0) {{
                        const min = Math.min(...values);
                        const max = Math.max(...values);
                        const mean = values.reduce((a, b) => a + b, 0) / values.length;
                        const start = values[0];
                        const end = values[values.length - 1];
                        const change = end - start;
                        const changePercent = start !== 0 ? (change / Math.abs(start)) * 100 : 0;
                        
                        result[metric] = {{
                            min, max, mean, start, end, change, changePercent
                        }};
                    }}
                }});
                return result;
            }}, []);
            
            // Handle metric selection
            const handleMetricToggle = (metric) => {{
                setSelectedMetrics(prev => {{
                    if (prev.includes(metric)) {{
                        const newSelection = prev.filter(m => m !== metric);
                        if (newSelection.length === 0) {{
                            // Don't allow empty selection
                            return prev;
                        }}
                        return newSelection;
                    }} else {{
                        return [...prev, metric];
                    }}
                }});
                setOverlayMode(false); // Switch to separate view when manually selecting
                setFullscreenFromOverlay(null); // Clear fullscreen state
                
                // Force a re-render of charts after a small delay
                setTimeout(() => {{
                    window.dispatchEvent(new Event('resize'));
                }}, 100);
            }};
            
            const handleSelectAll = () => {{
                setSelectedMetrics(DATA.metricsList);
                setOverlayMode(true);
                setDropdownOpen(false);
                setFullscreenFromOverlay(null); // Clear fullscreen state
                setPreviousOverlayMetrics(null); // Clear previous state
                setSubtitle('Visualizing training dynamics across checkpoints');
                
                // Force a re-render of charts
                setTimeout(() => {{
                    window.dispatchEvent(new Event('resize'));
                }}, 100);
            }};
            
            // Close dropdown when clicking outside
            useEffect(() => {{
                const handleClickOutside = (event) => {{
                    if (dropdownOpen && !event.target.closest('.metric-selector-container')) {{
                        setDropdownOpen(false);
                    }}
                }};
                
                document.addEventListener('click', handleClickOutside);
                return () => document.removeEventListener('click', handleClickOutside);
            }}, [dropdownOpen]);
            
            // Update subtitle based on selection
            useEffect(() => {{
                if (fullscreenFromOverlay) {{
                    setSubtitle(`Viewing ${{fullscreenFromOverlay}} (click outside to return to overlay)`);
                }} else if (selectedMetrics.length === DATA.metricsList.length && overlayMode) {{
                    setSubtitle('Visualizing training dynamics across checkpoints');
                }} else if (selectedMetrics.length === 1) {{
                    setSubtitle(`Displaying change in ${{selectedMetrics[0]}}`);
                }} else if (overlayMode) {{
                    setSubtitle(`Comparing ${{selectedMetrics.length}} metrics in overlay view`);
                }} else {{
                    setSubtitle(`Viewing ${{selectedMetrics.length}} metrics`);
                }}
            }}, [selectedMetrics, overlayMode, fullscreenFromOverlay]);
            
            // Export data as CSV
            const exportCSV = () => {{
                let csv = 'Checkpoint';
                DATA.metricsList.forEach(metric => {{
                    csv += ',' + metric;
                }});
                csv += '\\n';
                
                DATA.checkpoints.forEach((checkpoint, i) => {{
                    csv += checkpoint;
                    DATA.metricsList.forEach(metric => {{
                        csv += ',' + DATA.metrics[metric][i];
                    }});
                    csv += '\\n';
                }});
                
                const blob = new Blob([csv], {{ type: 'text/csv' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'metrics_data.csv';
                a.click();
                URL.revokeObjectURL(url);
            }};
            
            // Download all charts as PNG
            const downloadPNG = () => {{
                // For simplicity, we'll download the current view
                alert('PNG download would capture the current visualization');
            }};
            
            // Determine grid class based on number of metrics
            const getGridClass = () => {{
                if (overlayMode) return 'single';
                const count = selectedMetrics.length;
                if (count === 1) return 'single';
                if (count === 2) return 'double';
                if (count === 3) return 'triple';
                if (count === 4) return 'quad';
                return 'many';
            }};
            
            return (
                <div className="container">
                    <div className="header">
                        <h1>Phase-Viz</h1>
                        <p>{{subtitle}}</p>
                    </div>
                    
                    <div className="controls">
                        <div className="metric-selector-container">
                            <button 
                                className="metric-selector-button"
                                onClick={{() => setDropdownOpen(!dropdownOpen)}}
                            >
                                <span>
                                    {{selectedMetrics.length === DATA.metricsList.length 
                                        ? 'All Metrics' 
                                        : `${{selectedMetrics.length}} Selected`}}
                                </span>
                                <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
                                    <path d="M2.5 4.5l3.5 3.5 3.5-3.5" stroke="currentColor" strokeWidth="1.5" fill="none"/>
                                </svg>
                            </button>
                            
                            <div className={{`metric-selector-dropdown ${{dropdownOpen ? 'open' : ''}}`}}>
                                <div className="metric-option" onClick={{handleSelectAll}}>
                                    <input 
                                        type="checkbox" 
                                        checked={{selectedMetrics.length === DATA.metricsList.length}}
                                        readOnly
                                    />
                                    <span>All Metrics</span>
                                </div>
                                {{DATA.metricsList.map(metric => (
                                    <div 
                                        key={{metric}} 
                                        className="metric-option"
                                        onClick={{() => handleMetricToggle(metric)}}
                                    >
                                        <input 
                                            type="checkbox" 
                                            checked={{selectedMetrics.includes(metric)}}
                                            readOnly
                                        />
                                        <span>{{metric}}</span>
                                    </div>
                                ))}}
                            </div>
                        </div>
                        
                        {{selectedMetrics.length > 1 && (
                            <button 
                                className={{`overlay-button ${{overlayMode ? 'active' : ''}}`}}
                                onClick={{() => setOverlayMode(!overlayMode)}}
                            >
                                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                                    <path d="M2 2h5v5H2zM9 2h5v5H9zM2 9h5v5H2zM9 9h5v5H9z" opacity={{overlayMode ? 0.3 : 1}}/>
                                    <path d="M4 4h8v8H4z" opacity={{overlayMode ? 1 : 0.3}}/>
                                </svg>
                                {{overlayMode ? 'Separate Views' : 'Overlay Plots'}}
                            </button>
                        )}}
                        
                        <button 
                            className={{`phase-button ${{showPhaseTransitions ? 'active' : ''}}`}}
                            onClick={{() => setShowPhaseTransitions(!showPhaseTransitions)}}
                        >
                            Phase Transitions
                        </button>
                        
                        <button className="action-button" onClick={{exportCSV}}>
                            Export CSV
                        </button>
                        
                        <button className="action-button" onClick={{downloadPNG}}>
                            Download PNG
                        </button>
                    </div>
                    
                    {{overlayMode && selectedMetrics.length > 1 ? (
                        <div className="charts-grid single">
                            <OverlayChart 
                                metrics={{selectedMetrics}}
                                data={{DATA.metrics}}
                                showPhaseTransitions={{showPhaseTransitions}}
                                onMetricClick={{(metric) => {{
                                    // Save current state and show fullscreen
                                    setPreviousOverlayMetrics(selectedMetrics);
                                    setFullscreenFromOverlay(metric);
                                    setSelectedMetrics([metric]);
                                    setOverlayMode(false);
                                }}}}
                            />
                        </div>
                    ) : fullscreenFromOverlay ? (
                        // Show fullscreen view from overlay click
                        <>
                            <div className="charts-grid single">
                                <Chart
                                    metric={{fullscreenFromOverlay}}
                                    data={{DATA.metrics[fullscreenFromOverlay]}}
                                    isExpanded={{false}}
                                    onToggleExpand={{() => {{}}}}
                                    showPhaseTransitions={{showPhaseTransitions}}
                                />
                            </div>
                            <div style={{{{ textAlign: 'center', marginTop: '1rem' }}}}>
                                <button 
                                    className="action-button"
                                    onClick={{() => {{
                                        // Return to overlay view
                                        setFullscreenFromOverlay(null);
                                        setSelectedMetrics(previousOverlayMetrics || DATA.metricsList);
                                        setOverlayMode(true);
                                        setPreviousOverlayMetrics(null);
                                    }}}}
                                >
                                    ← Back to Overlay View
                                </button>
                            </div>
                        </>
                    ) : (
                        <div className={{`charts-grid ${{getGridClass()}}`}}>
                            {{selectedMetrics.map(metric => (
                                <Chart
                                    key={{`${{metric}}-${{selectedMetrics.length}}`}}
                                    metric={{metric}}
                                    data={{DATA.metrics[metric]}}
                                    isExpanded={{expandedMetric === metric}}
                                    onToggleExpand={{() => setExpandedMetric(
                                        expandedMetric === metric ? null : metric
                                    )}}
                                    showPhaseTransitions={{showPhaseTransitions}}
                                />
                            ))}}
                        </div>
                    )}}
                    
                    {{expandedMetric && (
                        <div 
                            className="overlay visible"
                            onClick={{() => setExpandedMetric(null)}}
                        />
                    )}}
                    
                    <div className="stats-container">
                        {{selectedMetrics.map(metric => allStats[metric] && (
                            <div key={{metric}} className="stat-card">
                                <h3>
                                    <span 
                                        className="metric-color" 
                                        style={{{{ backgroundColor: metricColors[metric] }}}}
                                    ></span>
                                    {{metric}}
                                </h3>
                                <div className="stat-grid">
                                    <div className="stat-item">
                                        <span className="stat-label">Mean</span>
                                        <span className="stat-value">{{allStats[metric].mean.toFixed(6)}}</span>
                                    </div>
                                    <div className="stat-item">
                                        <span className="stat-label">Min</span>
                                        <span className="stat-value">{{allStats[metric].min.toFixed(6)}}</span>
                                    </div>
                                    <div className="stat-item">
                                        <span className="stat-label">Max</span>
                                        <span className="stat-value">{{allStats[metric].max.toFixed(6)}}</span>
                                    </div>
                                    <div className="stat-item">
                                        <span className="stat-label">Start</span>
                                        <span className="stat-value">{{allStats[metric].start.toFixed(6)}}</span>
                                    </div>
                                    <div className="stat-item">
                                        <span className="stat-label">End</span>
                                        <span className="stat-value">{{allStats[metric].end.toFixed(6)}}</span>
                                    </div>
                                    <div className="stat-item">
                                        <span className="stat-label">Change</span>
                                        <span 
                                            className="stat-value"
                                            style={{{{
                                                color: allStats[metric].change > 0 ? '#10b981' : 
                                                      allStats[metric].change < 0 ? '#ef4444' : '#9ca3af'
                                            }}}}
                                        >
                                            {{allStats[metric].change > 0 ? '+' : ''}}
                                            {{allStats[metric].changePercent.toFixed(2)}}%
                                        </span>
                                    </div>
                                </div>
                            </div>
                        ))}}
                    </div>
                </div>
            );
        }}
        
        // Cleanup handling
        function setupCleanup() {{
            let cleanupAttempted = false;
            
            async function sendCleanupRequest() {{
                if (cleanupAttempted) return;
                cleanupAttempted = true;
                
                try {{
                    await fetch(`http://localhost:${{CLEANUP_PORT}}/cleanup`, {{
                        method: 'POST',
                        mode: 'no-cors',
                        keepalive: true,
                        body: JSON.stringify({{ cleanup: true }})
                    }});
                }} catch (e) {{
                    try {{
                        navigator.sendBeacon(
                            `http://localhost:${{CLEANUP_PORT}}/cleanup`,
                            JSON.stringify({{ cleanup: true }})
                        );
                    }} catch (e2) {{
                        console.error('Cleanup failed:', e2);
                    }}
                }}
            }}
            
            window.addEventListener('beforeunload', sendCleanupRequest);
            window.addEventListener('unload', sendCleanupRequest);
            document.addEventListener('visibilitychange', function() {{
                if (document.visibilityState === 'hidden') {{
                    sendCleanupRequest();
                }}
            }});
            window.addEventListener('pagehide', sendCleanupRequest);
        }}
        
        // Initialize
        setupCleanup();
        
        // Render the app
        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<MetricsVisualization />);
    </script>
</body>
</html>"""
    
    try:
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"Created HTML file: {output_html}")
        
        # Open browser
        with suppress_stdout_stderr():
            import webbrowser
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