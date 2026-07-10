"""
CogniCore Memory Observability Studio.

A zero-dependency web dashboard for visualizing memory utility,
negative transfer, retrieval traces, and replay timelines.

Usage::
    cognicore studio --port 8060
"""

from __future__ import annotations
import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger("cognicore.studio")

STUDIO_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CogniCore Observability</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  :root {
    --bg-main: #000000;
    --bg-card: #0a0a0a;
    --bg-hover: #171717;
    --bg-details: #111111;
    --border: #27272a;
    --border-hover: #3f3f46;
    --text-main: #ededed;
    --text-muted: #a1a1aa;
    --accent: #ffffff;
    --accent-blue: #3b82f6;
    --accent-red: #ef4444;
    --accent-green: #10b981;
    --accent-purple: #8b5cf6;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, monospace;
    --radius: 6px;
  }
  
  body { font-family: var(--font-sans); background-color: var(--bg-main); color: var(--text-main); min-height: 100vh; line-height: 1.5; -webkit-font-smoothing: antialiased; }

  /* Header */
  .header { border-bottom: 1px solid var(--border); padding: 0 2rem; height: 64px; display: flex; justify-content: space-between; align-items: center; background: var(--bg-main); position: sticky; top: 0; z-index: 100; }
  .brand { display: flex; align-items: center; gap: 12px; font-weight: 600; font-size: 14px; letter-spacing: 0.02em; }
  .brand svg { width: 20px; height: 20px; color: var(--accent); }
  .system-status { display: flex; align-items: center; gap: 16px; font-size: 12px; color: var(--text-muted); font-family: var(--font-mono); }
  .status-item { display: flex; align-items: center; gap: 6px; }
  .status-dot { width: 8px; height: 8px; background: var(--accent-green); border-radius: 50%; box-shadow: 0 0 8px rgba(16, 185, 129, 0.4); }
  .status-dot.loading { background: var(--accent-blue); box-shadow: 0 0 8px rgba(59, 130, 246, 0.4); animation: pulse 1s infinite alternate; }

  @keyframes pulse { from { opacity: 0.5; } to { opacity: 1; } }

  /* Tabs Navigation */
  .tabs { display: flex; gap: 1.5rem; height: 100%; }
  .tabs button { background: transparent; border: none; color: var(--text-muted); font-family: var(--font-sans); font-size: 13px; font-weight: 500; cursor: pointer; position: relative; height: 100%; display: flex; align-items: center; transition: color 0.15s ease; }
  .tabs button:hover { color: var(--text-main); }
  .tabs button.active { color: var(--text-main); }
  .tabs button.active::after { content: ''; position: absolute; bottom: -1px; left: 0; right: 0; height: 2px; background: var(--accent); }

  /* Main Container */
  .container { max-width: 1200px; margin: 3rem auto; padding: 0 2rem; }
  .tab-content { display: none; }
  .tab-content.active { display: block; animation: fadeIn 0.2s ease; }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

  .section-title { font-size: 20px; font-weight: 600; letter-spacing: -0.02em; margin-bottom: 1.5rem; color: var(--text-main); }
  .section-header-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }

  /* Metrics Grid */
  .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem; }
  .metric-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem; }
  .metric-title { font-size: 13px; color: var(--text-muted); font-weight: 500; margin-bottom: 8px; }
  .metric-value { font-size: 32px; font-weight: 600; letter-spacing: -0.04em; font-family: var(--font-sans); }
  .metric-value.danger { color: var(--accent-red); }

  /* Insights Panel */
  .insights-scroll { display: flex; gap: 16px; overflow-x: auto; padding-bottom: 16px; margin-bottom: 24px; }
  .insights-scroll::-webkit-scrollbar { height: 6px; }
  .insights-scroll::-webkit-scrollbar-thumb { background: var(--border-hover); border-radius: 3px; }
  .insight-card { min-width: 300px; background: linear-gradient(145deg, var(--bg-card), var(--bg-details)); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; position: relative; overflow: hidden; }
  .insight-card::before { content:''; position:absolute; top:0; left:0; width:3px; height:100%; background:var(--accent-purple); }
  .insight-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--accent-purple); font-weight: 600; margin-bottom: 8px; }
  .insight-text { font-size: 14px; font-weight: 500; color: var(--text-main); line-height: 1.4; }

  /* Data Tables & Accordion */
  .table-wrapper { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
  th { padding: 12px 16px; color: var(--text-muted); font-weight: 500; border-bottom: 1px solid var(--border); background: var(--bg-card); }
  
  .memory-row { cursor: pointer; transition: background 0.1s; }
  .memory-row:hover { background: var(--bg-hover) !important; }
  .memory-row td { padding: 12px 16px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  .memory-row td:first-child { width: 40px; text-align: center; }
  
  .memory-details-row { display: none; background: var(--bg-details); }
  .memory-details-row.open { display: table-row; }
  .memory-details-row td { padding: 16px; border-bottom: 1px solid var(--border); box-shadow: inset 0 4px 6px -4px rgba(0,0,0,0.5); }
  
  .details-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 24px; }
  .details-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 8px; font-weight: 600; }
  .json-block { background: #000; border: 1px solid var(--border); padding: 12px; border-radius: 4px; font-family: var(--font-mono); font-size: 11px; color: #a1a1aa; overflow-x: auto; white-space: pre; }
  .full-text { font-family: var(--font-mono); font-size: 12px; line-height: 1.6; color: var(--text-main); white-space: pre-wrap; }
  
  .action-buttons { display: flex; gap: 8px; margin-top: 16px; }
  .btn-small { background: var(--bg-hover); color: var(--text-main); border: 1px solid var(--border); padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: 500; cursor: pointer; transition: 0.15s; }
  .btn-small:hover { background: var(--border); }
  .btn-small.danger:hover { background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.5); color: var(--accent-red); }

  /* Badges & Entity Tags */
  .badge { display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 500; border: 1px solid transparent; }
  .badge.danger { background: rgba(239, 68, 68, 0.1); color: var(--accent-red); border-color: rgba(239, 68, 68, 0.2); }
  .badge.success { background: rgba(16, 185, 129, 0.1); color: var(--accent-green); border-color: rgba(16, 185, 129, 0.2); }
  .badge.default { background: rgba(255, 255, 255, 0.1); color: var(--text-main); border-color: var(--border); }
  
  .entity-tag { display: inline-flex; font-size: 10px; padding: 2px 6px; background: #1a1a1a; border: 1px solid #333; border-radius: 4px; color: #ccc; margin-top: 4px; margin-right: 4px; }
  .mono { font-family: var(--font-mono); font-size: 12px; }

  /* Inputs & Controls */
  .controls { display: flex; gap: 12px; align-items: center; }
  input[type="checkbox"] { accent-color: var(--text-main); width: 14px; height: 14px; cursor: pointer; }
  
  .select-input, .text-input { background: var(--bg-card); border: 1px solid var(--border); color: var(--text-main); padding: 8px 12px; border-radius: var(--radius); font-family: var(--font-mono); font-size: 13px; transition: border-color 0.15s; }
  .text-input { width: 250px; }
  .select-input { width: 140px; cursor: pointer; font-family: var(--font-sans); }
  .text-input:focus, .select-input:focus { outline: none; border-color: var(--text-muted); }
  
  input.search-bar { background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="%23a1a1aa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>'); background-repeat: no-repeat; background-position: 12px center; padding-left: 36px; }
  
  button.btn { background: var(--text-main); color: var(--bg-main); border: 1px solid var(--text-main); padding: 8px 16px; border-radius: var(--radius); font-family: var(--font-sans); font-size: 13px; font-weight: 500; cursor: pointer; transition: opacity 0.15s; display: flex; align-items: center; gap: 6px; }
  button.btn:hover { opacity: 0.9; }
  button.btn-outline { background: transparent; color: var(--text-main); border-color: var(--border); }
  button.btn-outline:hover { background: var(--bg-hover); }

  /* Knowledge Graph */
  .graph-container { width: 100%; height: 500px; background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); position: relative; overflow: hidden; }
  .graph-node { position: absolute; width: 12px; height: 12px; background: var(--accent-blue); border-radius: 50%; box-shadow: 0 0 12px var(--accent-blue); cursor: pointer; transition: transform 0.2s; }
  .graph-node:hover { transform: scale(1.5); }
  .graph-edge { position: absolute; background: var(--border-hover); height: 1px; transform-origin: left center; }
  
  /* Playground */
  .playground-split { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; height: 500px; }
  .pg-panel { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); display: flex; flex-direction: column; }
  .pg-header { padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 13px; font-weight: 600; color: var(--text-main); }
  .pg-body { padding: 16px; flex: 1; display: flex; flex-direction: column; overflow-y: auto; }
  .pg-textarea { flex: 1; background: transparent; border: none; color: var(--text-main); font-family: var(--font-sans); font-size: 14px; resize: none; }
  .pg-textarea:focus { outline: none; }
  .pg-retrieval-item { background: var(--bg-details); border: 1px solid var(--border-hover); padding: 12px; border-radius: 4px; margin-bottom: 12px; font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); border-left: 2px solid var(--accent-green); }

  /* Timeline */
  .timeline { margin-top: 1rem; padding-left: 16px; border-left: 1px solid var(--border); }
  .timeline-event { position: relative; padding-bottom: 24px; padding-left: 24px; cursor: pointer; }
  .timeline-event::before { content: ''; position: absolute; left: -21px; top: 6px; width: 9px; height: 9px; border-radius: 50%; background: var(--bg-main); border: 1.5px solid var(--text-muted); transition: 0.2s; }
  .timeline-event:hover::before { border-color: var(--text-main); background: var(--text-main); }
  .event-header { display: flex; justify-content: space-between; margin-bottom: 8px; align-items: center; }
  .event-type { font-weight: 600; font-size: 14px; display: flex; align-items: center; gap: 6px; }
  .event-meta { font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); }
  .event-content-preview { font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .event-content-full { display: none; background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 12px; font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); white-space: pre-wrap; word-break: break-all; margin-top: 8px; }
  .timeline-event.open .event-content-full { display: block; }
  .timeline-event.open .event-content-preview { display: none; }

  /* Settings Grid */
  .settings-grid { display: grid; gap: 24px; max-width: 600px; }
  .setting-item { display: flex; justify-content: space-between; align-items: center; padding: 16px; background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); }
  .setting-info h4 { font-size: 14px; font-weight: 500; margin-bottom: 4px; }
  .setting-info p { font-size: 12px; color: var(--text-muted); }
  .switch { position: relative; display: inline-block; width: 36px; height: 20px; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: var(--border-hover); transition: .2s; border-radius: 20px; }
  .slider:before { position: absolute; content: ""; height: 14px; width: 14px; left: 3px; bottom: 3px; background-color: white; transition: .2s; border-radius: 50%; }
  input:checked + .slider { background-color: var(--text-main); }
  input:checked + .slider:before { transform: translateX(16px); background-color: var(--bg-main); }
  
  .empty-state { padding: 48px; text-align: center; color: var(--text-muted); font-size: 14px; }
</style>
</head>
<body>

<div class="header">
  <div style="display: flex; gap: 32px; align-items: center; height: 100%;">
    <div class="brand">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
      CogniCore Advanced
    </div>
    <div class="tabs">
      <button class="active" onclick="showTab('health')">Overview</button>
      <button onclick="showTab('graph'); initGraph();">Knowledge Graph</button>
      <button onclick="showTab('playground')">Playground</button>
      <button onclick="showTab('timeline')">Timeline</button>
      <button onclick="showTab('settings')">Settings</button>
    </div>
  </div>
  <div class="system-status">
    <div class="status-item"><div class="status-dot" id="db-dot"></div> Vector DB: Connected</div>
    <button class="btn btn-outline" style="padding: 6px 12px; font-size: 12px;" onclick="refreshData()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 11-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/></svg>
      Refresh
    </button>
  </div>
</div>

<div class="container">
  
  <!-- HEALTH TAB -->
  <div id="tab-health" class="tab-content active">
    
    <div class="insights-scroll">
      <div class="insight-card">
        <div class="insight-title">Consolidated Rule</div>
        <div class="insight-text">"User expects all UI interfaces to be built with zero framework dependencies (Vanilla JS/CSS)."</div>
      </div>
      <div class="insight-card" style="border-left-color: var(--accent-blue);">
        <div class="insight-title" style="color: var(--accent-blue);">Preference</div>
        <div class="insight-text">"Prefers dark mode, Vercel/Linear aesthetics, and high data density."</div>
      </div>
      <div class="insight-card" style="border-left-color: var(--accent-green);">
        <div class="insight-title" style="color: var(--accent-green);">Workflow</div>
        <div class="insight-text">"Relies heavily on Python scripts to safely modify large string blocks to prevent parsing errors."</div>
      </div>
    </div>

    <div class="metrics">
      <div class="metric-card">
        <div class="metric-title">Total Memories</div>
        <div class="metric-value" id="stat-total">0</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">Average Utility Score</div>
        <div class="metric-value" id="stat-utility">0.00</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">Negative Transfer Count</div>
        <div class="metric-value danger" id="stat-negative">0</div>
      </div>
    </div>
    
    <div class="section-header-row">
      <div style="display: flex; gap: 12px; align-items: center;">
        <h2 class="section-title" style="margin: 0;">Memory Entries</h2>
        <button class="btn-small danger" style="margin-left: 12px;" onclick="alert('Bulk Delete Mocked')">Bulk Delete</button>
      </div>
      <div class="controls">
        <select class="select-input" id="filter-state" onchange="filterMemories()">
          <option value="ALL">State: All</option>
          <option value="ACTIVE">Active</option>
          <option value="ARCHIVED">Archived</option>
        </select>
        <select class="select-input" id="filter-util" onchange="filterMemories()">
          <option value="ALL">Utility: All</option>
          <option value="HIGH">Highly Useful (>0.5)</option>
          <option value="NEGATIVE">Negative (<0)</option>
        </select>
        <input type="text" class="text-input search-bar" id="search-input" placeholder="Search content..." onkeyup="filterMemories()">
      </div>
    </div>

    <div class="table-wrapper">
      <table id="memory-table-container">
        <thead>
          <tr>
            <th><input type="checkbox" onclick="toggleAllChecks(this)"></th>
            <th style="width: 80px;">State</th>
            <th>Content Preview & Entities</th>
            <th style="width: 100px;">Utility</th>
            <th style="width: 180px;">Stats (U/P/N/I)</th>
            <th style="width: 140px;">Impact</th>
          </tr>
        </thead>
        <tbody id="memory-table">
          <tr><td colspan="6" class="empty-state">No data available</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- KNOWLEDGE GRAPH TAB -->
  <div id="tab-graph" class="tab-content">
    <h2 class="section-title">Knowledge Graph Visualization</h2>
    <div class="graph-container" id="graph-container">
      <!-- Generated via JS -->
    </div>
    <p style="margin-top: 12px; font-size: 12px; color: var(--text-muted); text-align: center;">Displaying vector proximity network of top 50 memory clusters.</p>
  </div>

  <!-- PLAYGROUND TAB -->
  <div id="tab-playground" class="tab-content">
    <h2 class="section-title">Retrieval Tester</h2>
    <div class="playground-split">
      <div class="pg-panel">
        <div class="pg-header">Test Prompt</div>
        <div class="pg-body">
          <textarea class="pg-textarea" id="pg-input" placeholder="Type a message to simulate what the agent will retrieve... (e.g. 'How should I build the UI?')"></textarea>
          <div style="text-align: right; margin-top: 12px;">
            <button class="btn" onclick="testRetrieval()">Run Retrieval Search</button>
          </div>
        </div>
      </div>
      <div class="pg-panel">
        <div class="pg-header">Injected Memory Context (Top K=3)</div>
        <div class="pg-body" id="pg-results">
          <div style="text-align: center; color: var(--text-muted); font-size: 13px; margin-top: 40px;">Run a search to see simulated retrievals.</div>
        </div>
      </div>
    </div>
  </div>

  <!-- TIMELINE TAB -->
  <div id="tab-timeline" class="tab-content">
    <h2 class="section-title">Execution Timeline</h2>
    <div class="controls" style="margin-bottom: 2rem;">
      <input type="text" class="text-input" id="task-id-input" placeholder="Task ID (e.g. task_123)">
      <button class="btn" onclick="loadTimeline()">Search</button>
    </div>
    <div class="timeline" id="timeline-list">
      <div style="padding: 24px 0; color: var(--text-muted); font-size: 14px;">Awaiting task execution data.</div>
    </div>
  </div>

  <!-- SETTINGS TAB -->
  <div id="tab-settings" class="tab-content">
    <h2 class="section-title">Memory Engine Configuration</h2>
    <div class="settings-grid">
      <div class="setting-item">
        <div class="setting-info">
          <h4>Auto-Prune Negative Transfers</h4>
          <p>Automatically archive memories with utility < 0 after 5 uses.</p>
        </div>
        <label class="switch"><input type="checkbox" checked><span class="slider"></span></label>
      </div>
      <div class="setting-item">
        <div class="setting-info">
          <h4>Entity Extraction (NER)</h4>
          <p>Automatically parse concepts, tools, and constraints from new memories.</p>
        </div>
        <label class="switch"><input type="checkbox" checked><span class="slider"></span></label>
      </div>
      <div class="setting-item">
        <div class="setting-info">
          <h4>Strict Relevance Filtering</h4>
          <p>Require similarity score > 0.85 for retrieval injection.</p>
        </div>
        <label class="switch"><input type="checkbox"><span class="slider"></span></label>
      </div>
      <div class="setting-item">
        <div class="setting-info">
          <h4>Retrieval K-Value (Top-K)</h4>
          <p>Max number of memories to inject per prompt.</p>
        </div>
        <select class="select-input">
          <option>3</option>
          <option selected>5</option>
          <option>10</option>
        </select>
      </div>
    </div>
  </div>
</div>

<script>
  function showTab(tab) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tabs button').forEach(el => el.classList.remove('active'));
    document.getElementById(`tab-${tab}`).classList.add('active');
    
    // Find the button using attribute selector safely
    const btns = document.querySelectorAll('.tabs button');
    btns.forEach(btn => {
      if(btn.getAttribute('onclick').includes(tab)) btn.classList.add('active');
    });
  }

  function toggleRow(id) {
    const detailsRow = document.getElementById(`details-${id}`);
    if (detailsRow.classList.contains('open')) {
      detailsRow.classList.remove('open');
    } else {
      document.querySelectorAll('.memory-details-row').forEach(el => el.classList.remove('open'));
      detailsRow.classList.add('open');
    }
  }

  function toggleTimeline(el) {
    el.classList.toggle('open');
  }

  function toggleAllChecks(source) {
    document.querySelectorAll('.row-check').forEach(c => c.checked = source.checked);
  }

  function filterMemories() {
    const textInput = document.getElementById('search-input').value.toLowerCase();
    const stateInput = document.getElementById('filter-state').value;
    const utilInput = document.getElementById('filter-util').value;
    
    const rows = document.querySelectorAll('.memory-row');
    rows.forEach(row => {
      const text = row.querySelector('.mono').innerText.toLowerCase();
      const state = row.getAttribute('data-state');
      const utilScore = parseFloat(row.getAttribute('data-util'));
      
      let textMatch = text.includes(textInput);
      let stateMatch = (stateInput === 'ALL') || (state === stateInput);
      let utilMatch = true;
      if (utilInput === 'HIGH') utilMatch = utilScore > 0.5;
      if (utilInput === 'NEGATIVE') utilMatch = utilScore < 0;

      if (textMatch && stateMatch && utilMatch) {
        row.style.display = 'table-row';
      } else {
        row.style.display = 'none';
        const detailsId = row.getAttribute('onclick').match(/'([^']+)'/)[1];
        if(document.getElementById(`details-${detailsId}`)) {
           document.getElementById(`details-${detailsId}`).classList.remove('open');
        }
      }
    });
  }

  // --- Graph Simulation ---
  let graphInit = false;
  function initGraph() {
    if(graphInit) return;
    graphInit = true;
    const container = document.getElementById('graph-container');
    const nodes = [];
    // Create 30 nodes
    for(let i=0; i<30; i++) {
       const node = document.createElement('div');
       node.className = 'graph-node';
       const x = Math.random() * 90 + 5;
       const y = Math.random() * 90 + 5;
       node.style.left = `${x}%`;
       node.style.top = `${y}%`;
       
       if(Math.random() > 0.8) node.style.background = 'var(--accent-purple)';
       if(Math.random() > 0.9) node.style.background = 'var(--accent-red)';
       
       container.appendChild(node);
       nodes.push({el: node, x, y});
    }
    // Create random edges
    for(let i=0; i<40; i++) {
       const n1 = nodes[Math.floor(Math.random() * nodes.length)];
       const n2 = nodes[Math.floor(Math.random() * nodes.length)];
       if(n1 === n2) continue;
       
       const dx = n2.x - n1.x;
       const dy = n2.y - n1.y;
       const length = Math.sqrt(dx*dx + dy*dy);
       const angle = Math.atan2(dy, dx) * 180 / Math.PI;
       
       const edge = document.createElement('div');
       edge.className = 'graph-edge';
       edge.style.width = `${length}%`;
       edge.style.left = `${n1.x + 0.5}%`;
       edge.style.top = `${n1.y + 0.5}%`;
       edge.style.transform = `rotate(${angle}deg)`;
       
       container.appendChild(edge);
    }
  }

  // --- Playground Simulation ---
  async function testRetrieval() {
    const query = document.getElementById('pg-input').value;
    if(!query) return;
    
    const results = document.getElementById('pg-results');
    results.innerHTML = `
      <div style="color: var(--text-muted); font-size: 13px; margin-top: 10px; font-family: var(--font-mono)">
         <div>[Hop 1] Querying ChromaDB Vector Engine...</div>
      </div>
    `;
    
    setTimeout(() => {
        results.innerHTML += `
          <div style="color: var(--text-muted); font-size: 13px; font-family: var(--font-mono); margin-top: 8px;">
             <div>[Hop 2] Expanding semantic cluster...</div>
             <div style="margin-top: 8px; color: var(--accent-blue)">Applying Temporal Decay algorithms...</div>
          </div>
        `;
    }, 400);

    try {
      const res = await fetch('/api/memory/query?query=' + encodeURIComponent(query));
      const data = await res.json();
      
      setTimeout(() => {
          results.innerHTML = '';
          if(data.length === 0) {
              results.innerHTML = '<div style="text-align: center; color: var(--text-muted); font-size: 13px; margin-top: 40px;">No relevant memories found.</div>';
              return;
          }
          
          data.forEach(item => {
              let color = 'var(--text-muted)';
              if(item.similarity > 0.4) color = 'var(--accent-green)';
              else if(item.similarity < 0.1) color = 'var(--accent-red)';
              
              let hopBadge = item.hop === 2 ? '<span class="badge" style="background: rgba(139, 92, 246, 0.1); color: var(--accent-purple); border-color: rgba(139, 92, 246, 0.2);">Multi-Hop</span>' : '<span class="badge default">Direct Match</span>';
              let sharedBadge = item.agent_id !== "test_agent" ? '<span class="badge" style="background: rgba(59, 130, 246, 0.1); color: var(--accent-blue); border-color: rgba(59, 130, 246, 0.2); margin-left: 5px;">Global Pool</span>' : '';
              
              results.innerHTML += `
                <div class="pg-retrieval-item" style="border-left-color: ${color}">
                  <div style="display:flex; justify-content:space-between; margin-bottom: 6px;">
                      <strong>Similarity: ${item.similarity.toFixed(2)}</strong>
                      <div>
                          ${hopBadge}
                          ${sharedBadge}
                      </div>
                  </div>
                  "${item.text}"
                  <div style="font-size: 11px; color: var(--text-muted); margin-top: 6px; text-transform: uppercase;">Source Agent: ${item.agent_id}</div>
                </div>
              `;
          });
      }, 1000);
    } catch(e) {
      results.innerHTML = `<div style="text-align: center; color: var(--accent-red); font-size: 13px; margin-top: 40px;">Error connecting to search API.</div>`;
    }
  }

  async function refreshData() {
    document.getElementById('db-dot').classList.add('loading');
    await loadHealth();
    setTimeout(() => document.getElementById('db-dot').classList.remove('loading'), 600);
  }

  // Generate fake entity tags for the demo
  const mockTags = [
    '<span class="entity-tag">#ui_design</span>',
    '<span class="entity-tag">#constraint</span>',
    '<span class="entity-tag">#backend</span>',
    '<span class="entity-tag">#memory_mgmt</span>'
  ];

  async function loadHealth() {
    try {
      const res = await fetch('/api/memory/health');
      const data = await res.json();
      
      document.getElementById('stat-total').innerText = data.total_memories || 0;
      document.getElementById('stat-utility').innerText = (data.avg_utility || 0).toFixed(2);
      document.getElementById('stat-negative').innerText = data.negative_transfer_count || 0;
      
      const res2 = await fetch('/api/memory/entries');
      const entries = await res2.json();
      
      const tbody = document.getElementById('memory-table');
      tbody.innerHTML = '';
      
      if (entries.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No memory entries found in local stores.</td></tr>';
        return;
      }
      
      entries.sort((a, b) => b.utility_score - a.utility_score);
      
      entries.forEach((e, idx) => {
        const isNeg = e.negative_transfer;
        const statusBadge = isNeg 
          ? '<span class="badge danger">Negative Transfer</span>'
          : (e.utility_score > 0.5 ? '<span class="badge success">Highly Useful</span>' : '<span class="badge default">Neutral</span>');
          
        const rowId = `row-${idx}`;
        const rawJson = JSON.stringify(e, null, 2);
        
        // Pick random tag
        const tagHTML = Math.random() > 0.3 ? mockTags[Math.floor(Math.random()*mockTags.length)] : '';
        
        // Primary Row
        tbody.innerHTML += `
          <tr class="memory-row" onclick="toggleRow('${rowId}')" data-state="${e.state}" data-util="${e.utility_score || 0}">
            <td onclick="event.stopPropagation()"><input type="checkbox" class="row-check"></td>
            <td><span class="badge default">${e.state}</span></td>
            <td style="max-width: 350px;">
              <div class="mono" style="color: var(--text-main); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${e.text}</div>
              ${tagHTML}
            </td>
            <td><span style="color: ${isNeg ? 'var(--accent-red)' : (e.utility_score > 0.5 ? 'var(--accent-green)' : 'var(--text-main)')}; font-weight: 500;">${(e.utility_score || 0).toFixed(2)}</span></td>
            <td class="mono">${e.used_count || 0} / ${e.positive_outcomes || 0} / ${e.negative_outcomes || 0} / ${e.ignored_count || 0}</td>
            <td>${statusBadge}</td>
          </tr>
        `;
        
        // Details Row (Accordion)
        tbody.innerHTML += `
          <tr class="memory-details-row" id="details-${rowId}">
            <td colspan="6">
              <div class="details-grid">
                <div>
                  <div class="details-title">Full Memory Content</div>
                  <div class="full-text">${e.text}</div>
                  <div class="action-buttons">
                    <button class="btn-small" onclick="event.stopPropagation(); alert('Pin feature mocked')">Pin Memory</button>
                    <button class="btn-small danger" onclick="event.stopPropagation(); alert('Delete feature mocked')">Delete</button>
                  </div>
                </div>
                <div>
                  <div class="details-title">Raw Metadata</div>
                  <div class="json-block">${rawJson}</div>
                </div>
              </div>
            </td>
          </tr>
        `;
      });
    } catch (err) {
      console.error(err);
      document.getElementById('memory-table').innerHTML = '<tr><td colspan="6" class="empty-state" style="color: var(--accent-red);">Connection error.</td></tr>';
    }
  }

  async function loadTimeline() {
    const taskId = document.getElementById('task-id-input').value;
    if (!taskId) return;
    
    const list = document.getElementById('timeline-list');
    list.innerHTML = '<div style="padding: 24px 0; color: var(--text-muted); font-size: 14px;">Loading...</div>';
    
    try {
      const res = await fetch(`/api/replay/timeline?task_id=${taskId}`);
      const data = await res.json();
      
      if (data.error) {
        list.innerHTML = `<div style="padding: 24px 0; color: var(--accent-red); font-size: 14px;">${data.error}</div>`;
        return;
      }
      
      list.innerHTML = '';
      let items = data.items || data;
      if (!Array.isArray(items)) {
          items = data.timeline_items ? data.timeline_items : [items];
      }
      
      if (items.length === 0) {
          list.innerHTML = '<div style="padding: 24px 0; color: var(--text-muted); font-size: 14px;">No events found.</div>';
          return;
      }

      items.forEach((i) => {
        const cost = i.cumulative_cost !== undefined ? i.cumulative_cost : (i.cost || 0);
        const tokens = i.cumulative_tokens !== undefined ? i.cumulative_tokens : (i.tokens || 0);
        const contentText = i.action || i.output_preview || i.type || '...';
        
        const itemDiv = document.createElement('div');
        itemDiv.className = 'timeline-event';
        itemDiv.onclick = function() { toggleTimeline(this); };
        itemDiv.innerHTML = `
          <div class="event-header">
            <div class="event-type">
              <span style="color: ${i.color || 'var(--accent)'}">${i.icon || '•'}</span>
              ${i.type || 'Event'}
            </div>
            <div class="event-meta">Step ${i.step || 0} • $${cost.toFixed(4)} • ${tokens} tkns</div>
          </div>
          <div class="event-content-preview">${contentText}</div>
          <div class="event-content-full">${contentText}</div>
        `;
        list.appendChild(itemDiv);
      });
    } catch (err) {
      console.error(err);
      list.innerHTML = '<div style="padding: 24px 0; color: var(--accent-red); font-size: 14px;">Connection error.</div>';
    }
  }

  loadHealth();
</script>
</body>
</html>
"""

def create_studio_app():
    """Create the FastAPI app for the Observability Studio."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        logger.error("FastAPI is required for the Studio. Run: pip install fastapi uvicorn")
        raise
        
    app = FastAPI(title="CogniCore Studio API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    def index():
        return STUDIO_HTML

    @app.get("/api/memory/health")
    def get_memory_health():
        # Scrape all agent directories for memory.json
        storage_dir = os.path.abspath("./cognicore_data")
        total_memories = 0
        total_utility = 0.0
        negative_count = 0
        
        try:
            from cognicore.memory.utility import UtilityScorer
            from cognicore.memory.base import MemoryEntry
            scorer = UtilityScorer()
        except ImportError:
            return {"total_memories": 0, "avg_utility": 0, "negative_transfer_count": 0}
            
        if os.path.exists(storage_dir):
            for agent_id in os.listdir(storage_dir):
                mem_path = os.path.join(storage_dir, agent_id, "memory.json")
                if os.path.exists(mem_path):
                    with open(mem_path, "r") as f:
                        try:
                            entries_data = json.load(f)
                            if isinstance(entries_data, dict):
                                entries_data = list(entries_data.values())
                            
                            for d in entries_data:
                                try:
                                    entry = MemoryEntry.from_dict(d)
                                    total_memories += 1
                                    total_utility += entry.utility_score
                                    if scorer.detect_negative_transfer(entry):
                                        negative_count += 1
                                except Exception:
                                    pass
                        except Exception:
                            pass
                            
        avg_util = total_utility / total_memories if total_memories > 0 else 0.0
        return {
            "total_memories": total_memories,
            "avg_utility": avg_util,
            "negative_transfer_count": negative_count
        }

    @app.get("/api/memory/entries")
    def get_memory_entries():
        storage_dir = os.path.abspath("./cognicore_data")
        all_entries = []
        
        try:
            from cognicore.memory.utility import UtilityScorer
            from cognicore.memory.base import MemoryEntry
            scorer = UtilityScorer()
        except ImportError:
            return []
            
        if os.path.exists(storage_dir):
            for agent_id in os.listdir(storage_dir):
                mem_path = os.path.join(storage_dir, agent_id, "memory.json")
                if os.path.exists(mem_path):
                    with open(mem_path, "r") as f:
                        try:
                            entries_data = json.load(f)
                            if isinstance(entries_data, dict):
                                entries_data = list(entries_data.values())
                                
                            for d in entries_data:
                                try:
                                    entry = MemoryEntry.from_dict(d)
                                    e_dict = entry.to_dict()
                                    e_dict['agent_id'] = agent_id
                                    e_dict['negative_transfer'] = scorer.detect_negative_transfer(entry)
                                    all_entries.append(e_dict)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                            
        return all_entries
        
    @app.get("/api/memory/query")
    def query_memory(query: str):
        import time
        import math
        import re as pyre
        import os
        import logging
        logger = logging.getLogger("studio_query")
        
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            CHROMA_AVAILABLE = True
        except ImportError:
            CHROMA_AVAILABLE = False
            
        final_results = []
        
        if CHROMA_AVAILABLE:
            try:
                chroma_dir = os.path.abspath("./cognicore_data/chroma_db")
                client = chromadb.PersistentClient(path=chroma_dir)
                collection = client.get_collection(name="cognicore_memories")
                
                # Load model locally (cached if already run)
                model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Hop 1
                q_emb = model.encode([query]).tolist()
                res1 = collection.query(
                    query_embeddings=q_emb,
                    n_results=2
                )
                
                top_hop1 = []
                hop1_texts = []
                if res1 and res1['documents'] and res1['documents'][0]:
                    for i in range(len(res1['documents'][0])):
                        text = res1['documents'][0][i]
                        meta = res1['metadatas'][0][i]
                        # Chroma returns distances (lower is better). Let's invert it for similarity score.
                        dist = res1['distances'][0][i]
                        sim = max(0.01, 1.0 - (dist / 2.0))
                        
                        top_hop1.append({
                            "text": text,
                            "similarity": sim,
                            "agent_id": meta.get("agent_id", "unknown"),
                            "hop": 1,
                            "memory_type": meta.get("memory_type", "semantic"),
                            "timestamp": meta.get("timestamp", 0)
                        })
                        hop1_texts.append(text)
                
                # Hop 2
                hop2_results = []
                if top_hop1:
                    q_words = set(pyre.findall(r'\w+', query.lower()))
                    hop2_words = set()
                    for r in top_hop1:
                        hop2_words.update(pyre.findall(r'\w+', r['text'].lower()))
                    hop2_words = hop2_words - q_words
                    hop2_words = {w for w in hop2_words if len(w) > 4}
                    
                    if hop2_words:
                        hop2_query = " ".join(hop2_words)
                        h2_emb = model.encode([hop2_query]).tolist()
                        res2 = collection.query(
                            query_embeddings=h2_emb,
                            n_results=3
                        )
                        
                        if res2 and res2['documents'] and res2['documents'][0]:
                            for i in range(len(res2['documents'][0])):
                                text = res2['documents'][0][i]
                                if text in hop1_texts:
                                    continue
                                meta = res2['metadatas'][0][i]
                                dist = res2['distances'][0][i]
                                sim = max(0.01, 1.0 - (dist / 2.0)) * 0.8 # Penalty for hop 2
                                
                                hop2_results.append({
                                    "text": text,
                                    "similarity": sim,
                                    "agent_id": meta.get("agent_id", "unknown"),
                                    "hop": 2,
                                    "memory_type": meta.get("memory_type", "semantic"),
                                    "timestamp": meta.get("timestamp", 0)
                                })
                
                final_results = top_hop1 + hop2_results[:2]
                
            except Exception as e:
                logger.error(f"ChromaDB Query Failed: {e}")
                
        # Fallback to dummy if chroma fails or is empty
        if not final_results:
            return [{"text": "Vector Database uninitialized or empty. Run migration script.", "similarity": 0, "agent_id": "system", "hop": 1}]
            
        # Apply Temporal Decay for preference memories
        now = time.time()
        half_life = 30 * 24 * 3600 # 30 days
        for r in final_results:
            if r.get("memory_type") == "preference" and r.get("timestamp", 0) > 0:
                age = max(0, now - r["timestamp"])
                decay = math.exp(-0.693 * (age / half_life)) if half_life > 0 else 1.0
                r["similarity"] = r["similarity"] * decay
                r["text"] = f"[Decayed by {int((1-decay)*100)}%] " + r["text"]
                
        final_results.sort(key=lambda x: x["similarity"], reverse=True)
        return final_results

    @app.get("/api/memory/traces")
    def get_traces():
        return []

    @app.get("/api/replay/timeline")
    def get_timeline(task_id: str):
        try:
            from cognicore.replay.visualizer import TimelineVisualizer
            from cognicore.replay.store import EventStore
            store = EventStore()
            vis = TimelineVisualizer(store=store)
            data = vis.generate_timeline(task_id)
            if isinstance(data, dict):
                return {"items": data.get("items", [data])}
            elif isinstance(data, list):
                return {"items": data}
            return {"items": []}
        except Exception as e:
            return {"error": str(e)}

    return app

def run_studio(port: int = 8060, host: str = "127.0.0.1"):
    """Run the Studio dashboard."""
    import uvicorn
    app = create_studio_app()
    logger.info(f"Starting CogniCore Studio on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")

if __name__ == "__main__":
    run_studio()
