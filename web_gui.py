#!/usr/bin/env python3

import os
import sys
import logging
import argparse
from flask import Flask, render_template, Response, jsonify, request
import threading
import json
import time
from pathlib import Path
import psutil
from memory_management import HypergraphMemory
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import numpy as np
from adaptive_heartbeat import AdaptiveHeartbeat

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='web_gui.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# Global variables
app = Flask(__name__)
memory = None
activity_regulator = None
heartbeat_system = AdaptiveHeartbeat()
heartbeat_thread = None

# Start the heartbeat in a separate thread
def start_heartbeat_thread():
    global heartbeat_thread
    if heartbeat_thread is None or not heartbeat_thread.is_alive():
        heartbeat_thread = threading.Thread(target=heartbeat_system.start, daemon=True)
        heartbeat_thread.start()
        logger.info("Started adaptive heartbeat thread")

# Memory for storing historical data
system_history = {
    'cpu': [],
    'memory': [],
    'disk': [],
    'network': [],
    'heartbeat': [],
    'timestamps': []
}

# Store heartbeat logs
heartbeat_logs = []

# HTML Template for the web interface
html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Echo System Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            width: 95%;
            margin: 0 auto;
            padding: 20px 0;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .tabs {
            display: flex;
            margin-top: 20px;
            border-bottom: 1px solid #ddd;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border: 1px solid transparent;
            border-bottom: none;
            border-radius: 5px 5px 0 0;
            margin-right: 5px;
        }
        .tab.active {
            background-color: white;
            border-color: #ddd;
            color: #2c3e50;
            font-weight: bold;
        }
        .tab:hover:not(.active) {
            background-color: #eee;
        }
        .tab-content {
            display: none;
            background-color: white;
            padding: 20px;
            border: 1px solid #ddd;
            border-top: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .tab-content.active {
            display: block;
        }
        .status-container {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .status-card {
            background-color: white;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            flex: 1;
            min-width: 200px;
            margin: 0 10px 10px 0;
        }
        .status-card h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .chart-container {
            background-color: white;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .chart-container h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .chart {
            width: 100%;
            height: 300px;
            margin-top: 15px;
        }
        .chart img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .control-panel {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 15px;
        }
        .control-card {
            background-color: #f9f9f9;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            flex: 1;
            min-width: 250px;
        }
        .btn {
            padding: 8px 15px;
            background-color: #2c3e50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-top: 10px;
        }
        .btn:hover {
            background-color: #1a252f;
        }
        .btn.danger {
            background-color: #e74c3c;
        }
        .btn.danger:hover {
            background-color: #c0392b;
        }
        .btn.warning {
            background-color: #f39c12;
        }
        .btn.warning:hover {
            background-color: #d35400;
        }
        .btn.success {
            background-color: #27ae60;
        }
        .btn.success:hover {
            background-color: #2ecc71;
        }
        .slider-container {
            margin-top: 10px;
        }
        input[type="range"] {
            width: 100%;
        }
        .value-display {
            text-align: center;
            margin-top: 5px;
            font-weight: bold;
        }
        .log-entry {
            padding: 8px;
            margin-bottom: 5px;
            border-radius: 4px;
        }
        .log-entry:nth-child(odd) {
            background-color: #f9f9f9;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .status-good {
            background-color: #2ecc71;
        }
        .status-warning {
            background-color: #f39c12;
        }
        .status-critical {
            background-color: #e74c3c;
        }
        /* Heartbeat tab specific styles */
        .pulse-animation {
            animation: pulse 1s infinite;
            display: inline-block;
            color: #e74c3c;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.2); }
            100% { transform: scale(1); }
        }
        .heartbeat-controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1><i class="fas fa-project-diagram"></i> Echo System Dashboard</h1>
        <div>
            <span id="current-time"></span>
        </div>
    </div>
    
    <div class="container">
        <div class="tabs">
            <div class="tab active" data-tab="overview">Overview</div>
            <div class="tab" data-tab="resources">System Resources</div>
            <div class="tab" data-tab="heartbeat">Adaptive Heartbeat</div>
            <div class="tab" data-tab="logs">Activity Logs</div>
            <div class="tab" data-tab="network">Network</div>
            <div class="tab" data-tab="config">Configuration</div>
        </div>
        
        <!-- Overview Tab -->
        <div id="overview" class="tab-content active">
            <div class="status-container">
                <div class="status-card">
                    <h3>System Status</h3>
                    <p><strong>Status:</strong> <span id="system-status">Operational</span></p>
                    <p><strong>Uptime:</strong> <span id="system-uptime">Loading...</span></p>
                </div>
                <div class="status-card">
                    <h3>Resource Usage</h3>
                    <p><strong>CPU:</strong> <span id="cpu-usage">Loading...</span></p>
                    <p><strong>Memory:</strong> <span id="memory-usage">Loading...</span></p>
                    <p><strong>Disk:</strong> <span id="disk-usage">Loading...</span></p>
                </div>
                <div class="status-card">
                    <h3>Heartbeat Status</h3>
                    <p><strong>Rate:</strong> <span id="overview-heartbeat-rate">Loading...</span></p>
                    <p><strong>Mode:</strong> <span id="overview-heartbeat-mode">Normal</span></p>
                </div>
                <div class="status-card">
                    <h3>Recent Events</h3>
                    <div id="recent-events">Loading events...</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>System Overview</h3>
                <div class="chart">
                    <img src="/chart/system_overview" alt="System Overview" id="system-overview-chart">
                </div>
            </div>
        </div>
        
        <!-- Resources Tab -->
        <div id="resources" class="tab-content">
            <div class="status-container">
                <div class="status-card">
                    <h3>CPU</h3>
                    <div id="cpu-detail">
                        <p><strong>Usage:</strong> <span id="cpu-percent">Loading...</span></p>
                        <p><strong>Cores:</strong> <span id="cpu-cores">Loading...</span></p>
                    </div>
                </div>
                <div class="status-card">
                    <h3>Memory</h3>
                    <div id="memory-detail">
                        <p><strong>Used:</strong> <span id="memory-used">Loading...</span></p>
                        <p><strong>Available:</strong> <span id="memory-available">Loading...</span></p>
                        <p><strong>Total:</strong> <span id="memory-total">Loading...</span></p>
                    </div>
                </div>
                <div class="status-card">
                    <h3>Disk</h3>
                    <div id="disk-detail">
                        <p><strong>Used:</strong> <span id="disk-used">Loading...</span></p>
                        <p><strong>Free:</strong> <span id="disk-free">Loading...</span></p>
                        <p><strong>Total:</strong> <span id="disk-total">Loading...</span></p>
                    </div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>CPU History</h3>
                <div class="chart">
                    <img src="/chart/cpu_history" alt="CPU History" id="cpu-chart">
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Memory History</h3>
                <div class="chart">
                    <img src="/chart/memory_history" alt="Memory History" id="memory-chart">
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Process List</h3>
                <table id="process-table">
                    <thead>
                        <tr>
                            <th>PID</th>
                            <th>Name</th>
                            <th>CPU %</th>
                            <th>Memory %</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="process-body">
                        <tr>
                            <td colspan="5">Loading processes...</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Adaptive Heartbeat Tab -->
        <div id="heartbeat" class="tab-content">
            <div class="status-container">
                <div class="status-card">
                    <h3><i class="fas fa-heartbeat pulse-animation"></i> Current Heartbeat</h3>
                    <p><strong>Rate:</strong> <span id="heartbeat-rate">Loading...</span> BPM</p>
                    <p><strong>System Status:</strong> <span id="heartbeat-status">
                        <span class="status-indicator status-good"></span>Normal
                    </span></p>
                    <p><strong>Hyper Drive:</strong> <span id="hyper-drive-status">Inactive</span></p>
                </div>
                
                <div class="status-card">
                    <h3>System Health</h3>
                    <p><strong>CPU Usage:</strong> <span id="heartbeat-cpu">Loading...</span></p>
                    <p><strong>Last Assessment:</strong> <span id="last-assessment">Loading...</span></p>
                    <button id="force-assessment" class="btn">Force Assessment</button>
                </div>
                
                <div class="status-card">
                    <h3>Quick Actions</h3>
                    <button id="toggle-hyper-drive" class="btn warning">Toggle Hyper Drive</button>
                    <button id="restart-heartbeat" class="btn danger">Restart Heartbeat</button>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Heartbeat History</h3>
                <p>Showing heartbeat rate over time with hyper drive periods highlighted in yellow</p>
                <div class="chart">
                    <img src="/chart/heartbeat_history" alt="Heartbeat History" id="heartbeat-chart">
                </div>
            </div>
            
            <div class="control-panel">
                <div class="control-card">
                    <h3>Base Heartbeat Rate</h3>
                    <p>Adjust the base heartbeat rate (beats per minute)</p>
                    <div class="slider-container">
                        <input type="range" id="base-rate-slider" min="30" max="120" value="60">
                        <div class="value-display"><span id="base-rate-value">60</span> BPM</div>
                    </div>
                    <button id="update-base-rate" class="btn">Update</button>
                </div>
                
                <div class="control-card">
                    <h3>Hyper Drive Threshold</h3>
                    <p>Set CPU threshold for automatic Hyper Drive activation</p>
                    <div class="slider-container">
                        <input type="range" id="hyper-threshold-slider" min="60" max="95" value="90">
                        <div class="value-display"><span id="hyper-threshold-value">90</span>%</div>
                    </div>
                    <button id="update-hyper-threshold" class="btn">Update</button>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Recent Heartbeat Events</h3>
                <div id="heartbeat-events">
                    <p>Loading events...</p>
                </div>
            </div>
        </div>
        
        <!-- Logs Tab -->
        <div id="logs" class="tab-content">
            <div class="chart-container">
                <h3>System Log</h3>
                <div id="log-content">
                    <p>Loading logs...</p>
                </div>
            </div>
        </div>
        
        <!-- Network Tab -->
        <div id="network" class="tab-content">
            <div class="status-container">
                <div class="status-card">
                    <h3>Network Status</h3>
                    <p><strong>Status:</strong> <span id="network-status">Connected</span></p>
                </div>
                <div class="status-card">
                    <h3>Network Traffic</h3>
                    <p><strong>Sent:</strong> <span id="network-sent">Loading...</span></p>
                    <p><strong>Received:</strong> <span id="network-received">Loading...</span></p>
                </div>
                <div class="status-card">
                    <h3>Active Connections</h3>
                    <p><strong>Count:</strong> <span id="connection-count">Loading...</span></p>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Network History</h3>
                <div class="chart">
                    <img src="/chart/network_history" alt="Network History" id="network-chart">
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Connection List</h3>
                <table id="connection-table">
                    <thead>
                        <tr>
                            <th>Local Address</th>
                            <th>Remote Address</th>
                            <th>Status</th>
                            <th>Type</th>
                        </tr>
                    </thead>
                    <tbody id="connection-body">
                        <tr>
                            <td colspan="4">Loading connections...</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Configuration Tab -->
        <div id="config" class="tab-content">
            <div class="chart-container">
                <h3>System Configuration</h3>
                <table id="config-table">
                    <thead>
                        <tr>
                            <th>Setting</th>
                            <th>Value</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody id="config-body">
                        <tr>
                            <td colspan="3">Loading configuration...</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // Tab switching functionality
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', function() {
                // Remove active class from all tabs
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                // Add active class to clicked tab
                this.classList.add('active');
                document.getElementById(this.dataset.tab).classList.add('active');
                
                // Refresh charts on tab change
                updateCharts();
            });
        });
        
        // Current time update
        function updateCurrentTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = now.toLocaleString();
        }
        
        // Update charts and refresh data
        function updateCharts() {
            const activeTab = document.querySelector('.tab.active').dataset.tab;
            
            // Add timestamp to force refresh of images
            const timestamp = new Date().getTime();
            
            if (activeTab === 'overview' || activeTab === 'resources') {
                document.getElementById('cpu-chart').src = '/chart/cpu_history?' + timestamp;
                document.getElementById('memory-chart').src = '/chart/memory_history?' + timestamp;
                document.getElementById('system-overview-chart').src = '/chart/system_overview?' + timestamp;
            }
            
            if (activeTab === 'network') {
                document.getElementById('network-chart').src = '/chart/network_history?' + timestamp;
            }
            
            if (activeTab === 'heartbeat') {
                document.getElementById('heartbeat-chart').src = '/chart/heartbeat_history?' + timestamp;
            }
        }
        
        // Fetch and update system metrics
        function updateSystemMetrics() {
            fetch('/api/system_metrics')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('cpu-usage').textContent = data.cpu + '%';
                    document.getElementById('cpu-percent').textContent = data.cpu + '%';
                    document.getElementById('memory-usage').textContent = data.memory + '%';
                    document.getElementById('disk-usage').textContent = data.disk + '%';
                    document.getElementById('system-uptime').textContent = data.uptime;
                    document.getElementById('cpu-cores').textContent = data.cpu_cores;
                    document.getElementById('memory-used').textContent = data.memory_used;
                    document.getElementById('memory-available').textContent = data.memory_available;
                    document.getElementById('memory-total').textContent = data.memory_total;
                    document.getElementById('disk-used').textContent = data.disk_used;
                    document.getElementById('disk-free').textContent = data.disk_free;
                    document.getElementById('disk-total').textContent = data.disk_total;
                    
                    // Update process table
                    const processBody = document.getElementById('process-body');
                    processBody.innerHTML = '';
                    data.processes.forEach(process => {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${process.pid}</td>
                            <td>${process.name}</td>
                            <td>${process.cpu}%</td>
                            <td>${process.memory}%</td>
                            <td>${process.status}</td>
                        `;
                        processBody.appendChild(row);
                    });
                })
                .catch(error => console.error('Error fetching system metrics:', error));
        }
        
        // Fetch and update network metrics
        function updateNetworkMetrics() {
            fetch('/api/network_metrics')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('network-sent').textContent = data.sent;
                    document.getElementById('network-received').textContent = data.received;
                    document.getElementById('connection-count').textContent = data.connections;
                    
                    // Update connection table
                    const connectionBody = document.getElementById('connection-body');
                    connectionBody.innerHTML = '';
                    data.connection_list.forEach(conn => {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${conn.local_address}</td>
                            <td>${conn.remote_address}</td>
                            <td>${conn.status}</td>
                            <td>${conn.type}</td>
                        `;
                        connectionBody.appendChild(row);
                    });
                })
                .catch(error => console.error('Error fetching network metrics:', error));
        }
        
        // Fetch and update system logs
        function updateSystemLogs() {
            fetch('/api/system_logs')
                .then(response => response.json())
                .then(data => {
                    const logContent = document.getElementById('log-content');
                    logContent.innerHTML = '';
                    data.logs.forEach(log => {
                        const logEntry = document.createElement('div');
                        logEntry.className = 'log-entry';
                        logEntry.innerHTML = `<strong>${log.timestamp}</strong>: ${log.message}`;
                        logContent.appendChild(logEntry);
                    });
                })
                .catch(error => console.error('Error fetching system logs:', error));
        }
        
        // Fetch and update recent events
        function updateRecentEvents() {
            fetch('/api/recent_events')
                .then(response => response.json())
                .then(data => {
                    const eventsDiv = document.getElementById('recent-events');
                    eventsDiv.innerHTML = '';
                    data.events.forEach(event => {
                        const eventEntry = document.createElement('div');
                        eventEntry.className = 'log-entry';
                        eventEntry.innerHTML = `<strong>${event.timestamp}</strong>: ${event.message}`;
                        eventsDiv.appendChild(eventEntry);
                    });
                })
                .catch(error => console.error('Error fetching recent events:', error));
        }
        
        // Fetch and update heartbeat metrics
        function updateHeartbeatMetrics() {
            fetch('/api/heartbeat_metrics')
                .then(response => response.json())
                .then(data => {
                    // Update heartbeat tab
                    document.getElementById('heartbeat-rate').textContent = data.current_rate;
                    document.getElementById('heartbeat-cpu').textContent = data.cpu_usage + '%';
                    document.getElementById('hyper-drive-status').textContent = data.hyper_drive ? 'Active' : 'Inactive';
                    
                    // Update overview tab heartbeat status
                    document.getElementById('overview-heartbeat-rate').textContent = data.current_rate + ' BPM';
                    document.getElementById('overview-heartbeat-mode').textContent = data.hyper_drive ? 'Hyper Drive' : 'Normal';
                    
                    // Update heartbeat status indicator
                    const statusIndicator = document.getElementById('heartbeat-status');
                    if (data.health_status === 'Good') {
                        statusIndicator.innerHTML = '<span class="status-indicator status-good"></span>Normal';
                    } else if (data.health_status === 'Warning') {
                        statusIndicator.innerHTML = '<span class="status-indicator status-warning"></span>Elevated';
                    } else if (data.health_status === 'Critical') {
                        statusIndicator.innerHTML = '<span class="status-indicator status-critical"></span>Critical';
                    }
                    
                    // Update heartbeat events
                    const eventsDiv = document.getElementById('heartbeat-events');
                    eventsDiv.innerHTML = '';
                    data.recent_logs.forEach(log => {
                        const logEntry = document.createElement('div');
                        logEntry.className = 'log-entry';
                        logEntry.innerHTML = `<strong>${log.timestamp}</strong>: ${log.message}`;
                        eventsDiv.appendChild(logEntry);
                    });
                })
                .catch(error => console.error('Error fetching heartbeat metrics:', error));
        }
        
        // Setup heartbeat controls
        function setupHeartbeatControls() {
            // Base rate slider
            const baseRateSlider = document.getElementById('base-rate-slider');
            const baseRateValue = document.getElementById('base-rate-value');
            
            baseRateSlider.addEventListener('input', () => {
                baseRateValue.textContent = baseRateSlider.value;
            });
            
            document.getElementById('update-base-rate').addEventListener('click', () => {
                const value = baseRateSlider.value;
                fetch(`/api/update_heartbeat_rate?value=${value}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Base heartbeat rate updated!');
                        } else {
                            alert('Error: ' + data.message);
                        }
                    })
                    .catch(error => console.error('Error updating heartbeat rate:', error));
            });
            
            // Hyper drive threshold slider
            const hyperThresholdSlider = document.getElementById('hyper-threshold-slider');
            const hyperThresholdValue = document.getElementById('hyper-threshold-value');
            
            hyperThresholdSlider.addEventListener('input', () => {
                hyperThresholdValue.textContent = hyperThresholdSlider.value;
            });
            
            document.getElementById('update-hyper-threshold').addEventListener('click', () => {
                const value = hyperThresholdSlider.value;
                fetch(`/api/update_hyper_threshold?value=${value}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Hyper drive threshold updated!');
                        } else {
                            alert('Error: ' + data.message);
                        }
                    })
                    .catch(error => console.error('Error updating threshold:', error));
            });
            
            // Toggle hyper drive button
            document.getElementById('toggle-hyper-drive').addEventListener('click', () => {
                fetch('/api/toggle_hyper_drive')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert(data.message);
                            updateHeartbeatMetrics();
                        } else {
                            alert('Error: ' + data.message);
                        }
                    })
                    .catch(error => console.error('Error toggling hyper drive:', error));
            });
            
            // Force assessment button
            document.getElementById('force-assessment').addEventListener('click', () => {
                fetch('/api/force_heartbeat_assessment')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('System assessment completed!');
                            updateHeartbeatMetrics();
                        } else {
                            alert('Error: ' + data.message);
                        }
                    })
                    .catch(error => console.error('Error forcing assessment:', error));
            });
            
            // Restart heartbeat button
            document.getElementById('restart-heartbeat').addEventListener('click', () => {
                if (confirm('Are you sure you want to restart the heartbeat system?')) {
                    fetch('/api/restart_heartbeat')
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                alert('Heartbeat system restarted!');
                                updateHeartbeatMetrics();
                            } else {
                                alert('Error: ' + data.message);
                            }
                        })
                        .catch(error => console.error('Error restarting heartbeat:', error));
                }
            });
        }
        
        // Initialize everything
        function initialize() {
            updateCurrentTime();
            updateSystemMetrics();
            updateNetworkMetrics();
            updateSystemLogs();
            updateRecentEvents();
            updateHeartbeatMetrics();
            updateCharts();
            setupHeartbeatControls();
            
            // Update time every second
            setInterval(updateCurrentTime, 1000);
            
            // Update metrics periodically
            setInterval(updateSystemMetrics, 5000);
            setInterval(updateNetworkMetrics, 5000);
            setInterval(updateSystemLogs, 10000);
            setInterval(updateRecentEvents, 5000);
            setInterval(updateHeartbeatMetrics, 3000);
            setInterval(updateCharts, 30000);
        }
        
        // Run initialization when page loads
        window.addEventListener('load', initialize);
    </script>
</body>
</html>
'''

def parse_arguments():
    parser = argparse.ArgumentParser(description="Web-based GUI Dashboard for Deep Tree Echo")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the web server on (default: 8080)")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    parser.add_argument("--no-activity", action="store_true", help="Disable activity monitoring system")
    return parser.parse_args()

def get_system_metrics():
    """Get current system metrics (CPU, memory, disk)"""
    try:
        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        
        # Try to get echo metrics if Deep Tree Echo is available
        echo_metrics = {}
        try:
            from deep_tree_echo import DeepTreeEcho
            if hasattr(DeepTreeEcho, 'get_instance'):
                # Singleton pattern
                echo = DeepTreeEcho.get_instance()
                echo_metrics = echo.analyze_echo_patterns()
        except Exception as e:
            logger.warning(f"Error getting echo metrics: {e}")
            
        return {
            'cpu': cpu,
            'memory': memory,
            'disk': disk,
            'echo_metrics': echo_metrics
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return {'cpu': 0, 'memory': 0, 'disk': 0}

def get_memory_stats():
    """Get memory system statistics"""
    try:
        if memory is None:
            return {'node_count': 0, 'edge_count': 0, 'avg_salience': 0.0}
        
        return memory.generate_statistics()
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return {'error': str(e)}

def get_recent_logs(max_logs=50):
    """Get recent activity logs from all components"""
    logs = []
    try:
        logs_dir = Path('activity_logs')
        if not logs_dir.exists():
            return {'logs': []}
            
        for component in logs_dir.iterdir():
            if component.is_dir():
                activity_file = component / 'activity.json'
                if activity_file.exists():
                    try:
                        with open(activity_file) as f:
                            component_logs = json.load(f)
                            logs.extend(component_logs)
                    except Exception as e:
                        logger.error(f"Error reading logs from {activity_file}: {e}")
        
        # Sort by timestamp, newest first
        logs.sort(key=lambda x: x.get('time', 0), reverse=True)
        
        # Limit to max_logs
        logs = logs[:max_logs]
        
        return {'logs': logs}
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return {'logs': []}

def generate_system_health_chart():
    """Generate chart showing system health over time"""
    try:
        # Placeholder history data for visualization
        history = {
            'timestamps': [time.time() - i * 60 for i in range(30, 0, -1)],
            'cpu': [psutil.cpu_percent() * 0.8 + 10 * np.sin(i/3) for i in range(30)],
            'memory': [psutil.virtual_memory().percent * 0.7 + 15 * np.sin(i/5 + 2) for i in range(30)]
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#272741')
        ax.set_facecolor('#323250')
        
        # Format timestamps as readable time
        labels = [time.strftime("%H:%M", time.localtime(t)) for t in history['timestamps']]
        
        # Plot data
        ax.plot(labels, history['cpu'], 'o-', color='#4f9cff', label='CPU Usage (%)')
        ax.plot(labels, history['memory'], 's-', color='#ff6e4a', label='Memory Usage (%)')
        
        # Set labels and title
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('Usage (%)', color='white')
        ax.set_title('System Resource Usage Over Time', color='white', fontsize=14)
        
        # Customize grid and ticks
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='x', colors='white', rotation=45)
        ax.tick_params(axis='y', colors='white')
        ax.set_ylim(0, 100)
        
        # Customize legend
        ax.legend(loc='upper left', facecolor='#272741', framealpha=0.8, labelcolor='white')
        
        plt.tight_layout()
        
        # Convert plot to image
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png', dpi=100)
        img_data.seek(0)
        plt.close(fig)
        
        return img_data
    except Exception as e:
        logger.error(f"Error generating system health chart: {e}")
        # Return a simple error image
        return generate_error_image("Error generating system health chart")

def generate_echo_history_chart():
    """Generate chart showing echo patterns over time"""
    try:
        # Placeholder echo history data
        history = {
            'timestamps': [time.time() - i * 60 for i in range(30, 0, -1)],
            'avg_echo': [0.4 + 0.2 * np.sin(i/5) for i in range(30)],
            'max_echo': [0.7 + 0.15 * np.sin(i/4 + 1) for i in range(30)],
            'resonant_nodes': [10 + 5 * np.sin(i/6 + 2) for i in range(30)]
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#272741')
        ax.set_facecolor('#323250')
        
        # Format timestamps as readable time
        labels = [time.strftime("%H:%M", time.localtime(t)) for t in history['timestamps']]
        
        # Plot echo values
        ax.plot(labels, history['avg_echo'], 'o-', color='#4f9cff', label='Avg Echo')
        ax.plot(labels, history['max_echo'], 's-', color='#9c4fff', label='Max Echo')
        
        # Create secondary y-axis for resonant nodes
        ax2 = ax.twinx()
        ax2.plot(labels, history['resonant_nodes'], '^-', color='#ff6e4a', label='Resonant Nodes')
        
        # Set labels and title
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('Echo Value', color='#4f9cff')
        ax2.set_ylabel('Node Count', color='#ff6e4a')
        ax.set_title('Echo Patterns Over Time', color='white', fontsize=14)
        
        # Customize ticks and grid
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='x', colors='white', rotation=45)
        ax.tick_params(axis='y', colors='#4f9cff')
        ax2.tick_params(axis='y', colors='#ff6e4a')
        
        # Customize limits
        ax.set_ylim(0, 1)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
                 facecolor='#272741', framealpha=0.8, labelcolor='white')
        
        plt.tight_layout()
        
        # Convert plot to image
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png', dpi=100)
        img_data.seek(0)
        plt.close(fig)
        
        return img_data
    except Exception as e:
        logger.error(f"Error generating echo history chart: {e}")
        return generate_error_image("Error generating echo history chart")

def generate_memory_graph():
    """Generate visualization of the memory system"""
    try:
        import networkx as nx
        
        # Create a sample memory graph if we don't have real data
        G = nx.DiGraph()
        
        # If we have a real memory system, use its data
        if memory is not None:
            # Add nodes from memory system (limit to 100 for performance)
            nodes = list(memory.nodes.items())[:100]
            for node_id, node in nodes:
                G.add_node(node_id, 
                         content=node.content[:50],
                         salience=getattr(node, 'salience', 0.5),
                         memory_type=str(getattr(node, 'memory_type', 'unknown')),
                         size=300 + 700 * getattr(node, 'salience', 0.5))
                
            # Add edges between these nodes
            for edge in memory.edges:
                if edge.from_id in G and edge.to_id in G:
                    G.add_edge(edge.from_id, edge.to_id, 
                             weight=edge.weight)
        else:
            # Generate a random memory graph for demonstration
            for i in range(30):
                node_id = f"node_{i}"
                memory_types = ['episodic', 'semantic', 'procedural', 'working']
                G.add_node(node_id, 
                         content=f"Memory node {i}",
                         salience=np.random.uniform(0.3, 0.9),
                         memory_type=np.random.choice(memory_types),
                         size=300 + 700 * np.random.uniform(0.3, 0.9))
                
            # Add some random edges
            for i in range(50):
                from_id = f"node_{np.random.randint(0, 30)}"
                to_id = f"node_{np.random.randint(0, 30)}"
                if from_id != to_id:
                    G.add_edge(from_id, to_id, weight=np.random.uniform(0.3, 0.9))
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#272741')
        ax.set_facecolor('#272741')
        
        # Define node positions using a layout algorithm
        pos = nx.spring_layout(G)
        
        # Get node attributes for visualization
        node_sizes = [data.get('size', 300) for _, data in G.nodes(data=True)]
        
        # Color nodes by memory type
        memory_types = [data.get('memory_type', 'unknown') for _, data in G.nodes(data=True)]
        unique_types = list(set(memory_types))
        color_map = plt.cm.get_cmap('tab10', len(unique_types))
        type_to_color = {t: color_map(i) for i, t in enumerate(unique_types)}
        node_colors = [type_to_color[t] for t in memory_types]
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(G, pos, 
                                     node_size=node_sizes, 
                                     node_color=node_colors, 
                                     alpha=0.8, ax=ax)
        
        # Draw edges with color based on weight
        edge_weights = [G[u][v].get('weight', 0.5) for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, 
                             edge_color=edge_weights,
                             edge_cmap=plt.cm.Blues,
                             width=1.5,
                             alpha=0.6,
                             arrows=True,
                             ax=ax,
                             arrowsize=10)
        
        # Create legend for memory types
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    label=t, markerfacecolor=type_to_color[t], markersize=8) 
                         for t in unique_types]
        ax.legend(handles=legend_elements, title="Memory Types", 
                 loc="upper right", framealpha=0.8,
                 facecolor='#272741', labelcolor='white')
        
        # Add title and remove axis
        ax.set_title("Memory Hypergraph Visualization", color='white', fontsize=16)
        ax.set_axis_off()
        
        plt.tight_layout()
        
        # Convert plot to image
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png', dpi=100)
        img_data.seek(0)
        plt.close(fig)
        
        return img_data
    except Exception as e:
        logger.error(f"Error generating memory graph: {e}")
        return generate_error_image("Error generating memory graph")

def generate_echo_network():
    """Generate visualization of the echo network"""
    try:
        import networkx as nx
        
        # Create a network representation of the echo tree
        G = nx.DiGraph()
        
        # Try to get the actual echo tree
        try:
            from deep_tree_echo import DeepTreeEcho
            echo = DeepTreeEcho() if not hasattr(DeepTreeEcho, 'get_instance') else DeepTreeEcho.get_instance()
            root_node = echo.root
            
            # If we have a real tree, use it
            if root_node:
                # Helper to add nodes recursively
                def add_node_to_graph(node, parent_id=None):
                    node_id = id(node)
                    G.add_node(node_id, 
                             echo_value=getattr(node, 'echo_value', 0.5),
                             content=getattr(node, 'content', 'Unknown'),
                             size=300 + 1000 * getattr(node, 'echo_value', 0.5))
                    
                    if parent_id is not None:
                        G.add_edge(parent_id, node_id)
                    
                    for child in getattr(node, 'children', []):
                        add_node_to_graph(child, node_id)
                
                # Add all nodes starting from root
                add_node_to_graph(root_node)
        except Exception as e:
            logger.warning(f"Could not get real echo tree, generating demo: {e}")
            
            # If no real tree, generate a demo one
            if len(G.nodes()) == 0:
                # Create a demo tree structure
                for i in range(20):
                    G.add_node(i, 
                             echo_value=np.random.uniform(0.2, 0.9),
                             content=f"Node {i}",
                             size=300 + 1000 * np.random.uniform(0.2, 0.9))
                
                # Create a tree-like structure
                for i in range(1, 20):
                    parent = (i - 1) // 3  # Simple formula to create a tree
                    G.add_edge(parent, i)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#272741')
        ax.set_facecolor('#272741')
        
        if not G.nodes():
            ax.text(0.5, 0.5, "No echo network data available", ha='center', va='center', color='white', fontsize=14)
            ax.set_axis_off()
        else:
            # Define node positions using a layout algorithm
            pos = nx.kamada_kawai_layout(G)
            
            # Get node attributes for visualization
            node_sizes = [data.get('size', 300) for _, data in G.nodes(data=True)]
            echo_values = [data.get('echo_value', 0.5) for _, data in G.nodes(data=True)]
            
            # Create a colormap
            cmap = plt.cm.viridis
            norm = plt.Normalize(vmin=0, vmax=1)
            
            # Draw nodes
            nodes = nx.draw_networkx_nodes(G, pos, 
                                         node_size=node_sizes, 
                                         node_color=echo_values, 
                                         cmap=cmap,
                                         alpha=0.9, ax=ax)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, 
                                 edge_color='gray',
                                 alpha=0.5, 
                                 arrows=True,
                                 arrowsize=10,
                                 ax=ax)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, label='Echo Value')
            cbar.ax.yaxis.label.set_color('white')
            cbar.ax.tick_params(colors='white')
            
            # Add title and remove axis
            ax.set_title("Deep Tree Echo Visualization", color='white', fontsize=16)
            ax.set_axis_off()
        
        plt.tight_layout()
        
        # Convert plot to image
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png', dpi=100)
        img_data.seek(0)
        plt.close(fig)
        
        return img_data
    except Exception as e:
        logger.error(f"Error generating echo network: {e}")
        return generate_error_image("Error generating echo network visualization")

def generate_error_image(error_text):
    """Generate a simple error image with text"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#272741')
    ax.set_facecolor('#272741')
    
    ax.text(0.5, 0.5, error_text, ha='center', va='center', color='white', fontsize=14)
    ax.set_axis_off()
    
    plt.tight_layout()
    
    # Convert plot to image
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=100)
    img_data.seek(0)
    plt.close(fig)
    
    return img_data

@app.route('/api/heartbeat_metrics')
def heartbeat_metrics():
    """Return the current metrics from the adaptive heartbeat system"""
    current_rate = heartbeat_system.current_heartbeat_rate
    hyper_drive = heartbeat_system.hyper_drive_active
    cpu_usage = heartbeat_system.last_cpu_usage
    
    # Determine health status based on heartbeat and CPU usage
    if hyper_drive:
        health_status = "Warning"
    elif cpu_usage > 80:
        health_status = "Warning"
    elif cpu_usage > 95:
        health_status = "Critical"
    else:
        health_status = "Good"
    
    # Get recent logs (limited to last 10)
    recent_logs = heartbeat_logs[-10:] if heartbeat_logs else []
    
    return jsonify({
        'current_rate': current_rate,
        'hyper_drive': hyper_drive,
        'cpu_usage': cpu_usage,
        'health_status': health_status,
        'recent_logs': recent_logs
    })

@app.route('/api/update_heartbeat_rate')
def update_heartbeat_rate():
    """Update the base heartbeat rate"""
    try:
        value = int(request.args.get('value', 60))
        heartbeat_system.base_heartbeat_rate = value
        log_heartbeat_event(f"Base heartbeat rate updated to {value} BPM")
        return jsonify({'success': True, 'message': f'Heartbeat rate updated to {value}'})
    except Exception as e:
        logger.error(f"Error updating heartbeat rate: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/update_hyper_threshold')
def update_hyper_threshold():
    """Update the hyper drive threshold"""
    try:
        value = int(request.args.get('value', 90))
        heartbeat_system.hyper_drive_threshold = value
        log_heartbeat_event(f"Hyper drive threshold updated to {value}%")
        return jsonify({'success': True, 'message': f'Threshold updated to {value}%'})
    except Exception as e:
        logger.error(f"Error updating hyper drive threshold: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/toggle_hyper_drive')
def toggle_hyper_drive():
    """Toggle the hyper drive mode"""
    try:
        new_state = not heartbeat_system.hyper_drive_active
        heartbeat_system.hyper_drive_active = new_state
        log_heartbeat_event(f"Hyper drive manually {'activated' if new_state else 'deactivated'}")
        return jsonify({'success': True, 'message': f'Hyper drive {"activated" if new_state else "deactivated"}'})
    except Exception as e:
        logger.error(f"Error toggling hyper drive: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/force_heartbeat_assessment')
def force_heartbeat_assessment():
    """Force the heartbeat system to perform a full assessment"""
    try:
        heartbeat_system.assess_system_state(force=True)
        log_heartbeat_event("Manual system assessment triggered")
        return jsonify({'success': True, 'message': 'Assessment completed'})
    except Exception as e:
        logger.error(f"Error during system assessment: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/restart_heartbeat')
def restart_heartbeat():
    """Restart the heartbeat system"""
    try:
        # Stop the current thread if it exists
        global heartbeat_thread
        if heartbeat_thread and heartbeat_thread.is_alive():
            heartbeat_system.stop_heartbeat()
            heartbeat_thread.join(timeout=2.0)
        
        # Reset and restart
        heartbeat_system.reset()
        start_heartbeat_thread()
        
        log_heartbeat_event("Heartbeat system restarted")
        return jsonify({'success': True, 'message': 'Heartbeat system restarted'})
    except Exception as e:
        logger.error(f"Error restarting heartbeat system: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/chart/heartbeat_history')
def heartbeat_history_chart():
    """Generate a chart showing heartbeat rate history"""
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(system_history['timestamps'][-50:], system_history['heartbeat'][-50:], 'r-', linewidth=2)
        
        # Add visual indicators for hyper drive periods if available
        hyper_periods = heartbeat_system.get_hyper_drive_periods()
        if hyper_periods:
            for period in hyper_periods[-5:]:  # Show last 5 periods
                start, end = period
                if end == 0:  # Still active
                    end = datetime.now()
                plt.axvspan(start, end, color='yellow', alpha=0.3)
        
        plt.title('Heartbeat Rate History')
        plt.xlabel('Time')
        plt.ylabel('Rate (BPM)')
        plt.grid(True)
        plt.tight_layout()
        
        # Convert plot to PNG image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        
        return Response(img.getvalue(), content_type='image/png')
    except Exception as e:
        logger.error(f"Error generating heartbeat chart: {e}")
        # Return a simple error image
        return Response(generate_error_image("Error generating chart"), content_type='image/png')

def log_heartbeat_event(message):
    """Log a heartbeat event"""
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'message': message
    }
    heartbeat_logs.append(log_entry)
    # Keep only the last 100 logs
    if len(heartbeat_logs) > 100:
        heartbeat_logs.pop(0)
    logger.info(f"Heartbeat event: {message}")

# Update the update_metrics function to include heartbeat data
def update_metrics():
    """Update system metrics for historical tracking"""
    while True:
        try:
            # Get current system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Get network stats (simplified)
            net_io = psutil.net_io_counters()
            net_usage = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # Convert to MB
            
            # Get current timestamp
            current_time = datetime.now()
            
            # Store in history
            system_history['cpu'].append(cpu_percent)
            system_history['memory'].append(memory_percent)
            system_history['disk'].append(disk_percent)
            system_history['network'].append(net_usage)
            system_history['timestamps'].append(current_time)
            
            # Add heartbeat data
            system_history['heartbeat'].append(heartbeat_system.current_heartbeat_rate)
            
            # Keep only the last 1000 data points to prevent memory issues
            if len(system_history['cpu']) > 1000:
                for key in system_history:
                    system_history[key] = system_history[key][-1000:]
            
            # Sleep for a bit
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error in update_metrics: {e}")
            time.sleep(5)  # Wait a bit longer if there was an error

# Flask routes
@app.route('/')
def index():
    return html_template

@app.route('/api/system_metrics')
def api_system_metrics():
    return jsonify(get_system_metrics())

@app.route('/api/memory_stats')
def api_memory_stats():
    return jsonify({'stats': get_memory_stats()})

@app.route('/api/recent_logs')
def api_recent_logs():
    return jsonify(get_recent_logs())

@app.route('/api/process_info')
def api_process_info():
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            processes.append({
                'pid': proc.info['pid'],
                'name': proc.info['name'],
                'cpu_percent': proc.info['cpu_percent'],
                'memory_percent': proc.info['memory_percent']
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Sort by CPU usage, highest first
    processes = sorted(processes, key=lambda p: p['cpu_percent'], reverse=True)[:10]
    return jsonify({'processes': processes})

@app.route('/api/tasks')
def api_tasks():
    try:
        tasks = []
        if activity_regulator:
            # Convert queue to list for viewing
            queue_copy = []
            for task in list(activity_regulator.task_queue.queue):
                queue_copy.append({
                    'task_id': task.task_id,
                    'priority': task.priority.name if hasattr(task.priority, 'name') else str(task.priority),
                    'scheduled_time': task.scheduled_time
                })
            tasks = queue_copy
        return jsonify({'tasks': tasks})
    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        return jsonify({'tasks': [], 'error': str(e)})

@app.route('/api/add_task')
def api_add_task():
    try:
        task_id = request.args.get('task_id', '')
        if not task_id or not activity_regulator:
            return jsonify({'success': False, 'error': 'Invalid task ID or activity regulator not available'})
        
        from activity_regulation import TaskPriority
        activity_regulator.add_task(
            task_id=task_id,
            callback=lambda: print(f"Executing {task_id}"),
            priority=TaskPriority.MEDIUM
        )
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error adding task: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_echo_threshold')
def api_update_echo_threshold():
    try:
        value = float(request.args.get('value', 0.75))
        
        try:
            from deep_tree_echo import DeepTreeEcho
            echo = DeepTreeEcho() if not hasattr(DeepTreeEcho, 'get_instance') else DeepTreeEcho.get_instance()
            echo.echo_threshold = value
        except Exception as e:
            logger.warning(f"Could not update echo threshold: {e}")
            
        return jsonify({'success': True, 'value': value})
    except Exception as e:
        logger.error(f"Error updating threshold: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/inject_random_echo')
def api_inject_random_echo():
    try:
        try:
            from deep_tree_echo import DeepTreeEcho
            echo = DeepTreeEcho() if not hasattr(DeepTreeEcho, 'get_instance') else DeepTreeEcho.get_instance()
            
            # Find random nodes to use
            all_nodes = []
            
            def collect_nodes(node):
                if node:
                    all_nodes.append(node)
                    for child in getattr(node, 'children', []):
                        collect_nodes(child)
            
            # Collect all nodes
            collect_nodes(echo.root)
            
            if len(all_nodes) > 1:
                import random
                source = random.choice(all_nodes)
                target = random.choice(all_nodes)
                while source == target:
                    target = random.choice(all_nodes)
                    
                strength = random.uniform(0.3, 0.9)
                echo.inject_echo(source, target, strength)
                return jsonify({'success': True})
        except Exception as e:
            logger.warning(f"Could not inject random echo: {e}")
            
        return jsonify({'success': False, 'error': 'Echo system not available'})
    except Exception as e:
        logger.error(f"Error in inject_random_echo: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/propagate_echoes')
def api_propagate_echoes():
    try:
        try:
            from deep_tree_echo import DeepTreeEcho
            echo = DeepTreeEcho() if not hasattr(DeepTreeEcho, 'get_instance') else DeepTreeEcho.get_instance()
            echo.propagate_echoes()
            return jsonify({'success': True})
        except Exception as e:
            logger.warning(f"Could not propagate echoes: {e}")
            
        return jsonify({'success': False, 'error': 'Echo system not available'})
    except Exception as e:
        logger.error(f"Error in propagate_echoes: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/prune_weak_echoes')
def api_prune_weak_echoes():
    try:
        try:
            from deep_tree_echo import DeepTreeEcho
            echo = DeepTreeEcho() if not hasattr(DeepTreeEcho, 'get_instance') else DeepTreeEcho.get_instance()
            echo.prune_weak_echoes()
            return jsonify({'success': True})
        except Exception as e:
            logger.warning(f"Could not prune weak echoes: {e}")
            
        return jsonify({'success': False, 'error': 'Echo system not available'})
    except Exception as e:
        logger.error(f"Error in prune_weak_echoes: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/chart/system_health')
def chart_system_health():
    img_data = generate_system_health_chart()
    return Response(img_data.getvalue(), mimetype='image/png')

@app.route('/chart/echo_history')
def chart_echo_history():
    img_data = generate_echo_history_chart()
    return Response(img_data.getvalue(), mimetype='image/png')

@app.route('/chart/memory_graph')
def chart_memory_graph():
    img_data = generate_memory_graph()
    return Response(img_data.getvalue(), mimetype='image/png')

@app.route('/chart/echo_network')
def chart_echo_network():
    img_data = generate_echo_network()
    return Response(img_data.getvalue(), mimetype='image/png')

def main():
    global memory, activity_regulator
    
    args = parse_arguments()
    
    try:
        # Initialize memory system
        logger.info("Initializing memory system")
        memory = HypergraphMemory(storage_dir="echo_memory")
        
        # Only initialize activity regulator if not disabled
        if not args.no_activity:
            logger.info("Initializing activity regulator")
            from activity_regulation import ActivityRegulator
            activity_regulator = ActivityRegulator()

        # Start the heartbeat system
        start_heartbeat_thread()
        
        # Start the metrics update thread
        metrics_thread = threading.Thread(target=update_metrics, daemon=True)
        metrics_thread.start()

        # Print server information
        print("\n" + "="*80)
        print(f"Deep Tree Echo Dashboard Web Server")
        print(f"Running on http://localhost:{args.port}")
        print(f"Also try your forwarded port URLs:")
        print(f"- http://127.0.0.1:{args.port}")
        print(f"- http://localhost:{args.port}")
        print("="*80 + "\n")

        # Start Flask server
        app.run(host='0.0.0.0', port=args.port, debug=args.debug)
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())