// skill-bridge/extension.js
//
// Pseudo-MCP bridge: exposes a local HTTP server so that skill.py (or any
// local tool) can call GitHub Copilot's Language Model API without any
// direct API keys, MCP servers, or external network calls beyond what VS Code
// already handles for you.
//
// API surface
// -----------
// POST /chat
//   Body:  { "messages": [{"role": "user"|"assistant", "content": "..."}],
//             "model_family": "gpt-4o" (optional),
//             "system": "..."          (optional system prompt) }
//   Reply: { "response": "...", "model": "<model name used>" }
//         | { "error": "..." }          (HTTP 500)
//
// GET /status
//   Reply: { "running": true, "port": 7777, "models": ["gpt-4o", ...] }
//
// POST /list-models
//   Reply: { "models": [{"id": "...", "vendor": "...", "family": "...", "version": "..."}] }
//
// All endpoints bind to 127.0.0.1 only — no external exposure.

"use strict";
const vscode = require("vscode");
const http   = require("http");
const cp     = require("child_process");
const path   = require("path");
const fs     = require("fs");

let server      = null;
let statusBarItem = null;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Safely read the full body of an IncomingMessage, resolve to string. */
function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data",  chunk => { data += chunk; });
    req.on("end",   ()    => resolve(data));
    req.on("error", err   => reject(err));
  });
}

/** Send a JSON response. */
function jsonReply(res, status, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(status, {
    "Content-Type":   "application/json",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

/** Select the best available Copilot model. */
async function selectModel(family) {
  const preferred = family || vscode.workspace.getConfiguration("skillBridge").get("modelFamily", "gpt-4o");
  // Try exact family match first, then any copilot model
  for (const selector of [
    { vendor: "copilot", family: preferred },
    { vendor: "copilot" },
  ]) {
    const models = await vscode.lm.selectChatModels(selector);
    if (models.length > 0) return models[0];
  }
  return null;
}

/** Convert our wire format messages to VS Code LM messages. */
function toVscodeLmMessages(messages, system) {
  const result = [];
  if (system) {
    // Prepend system text as a User turn — Copilot LM API only supports User/Assistant
    result.push(vscode.LanguageModelChatMessage.User(`[System]\n${system}`));
  }
  for (const m of messages) {
    if (m.role === "assistant") {
      result.push(vscode.LanguageModelChatMessage.Assistant(m.content));
    } else {
      result.push(vscode.LanguageModelChatMessage.User(m.content));
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Request router
// ---------------------------------------------------------------------------

async function handleRequest(req, res) {
  const { method, url } = req;

  // Health / status
  if (method === "GET" && url === "/status") {
    const models = await vscode.lm.selectChatModels({ vendor: "copilot" });
    return jsonReply(res, 200, {
      running: true,
      port:    vscode.workspace.getConfiguration("skillBridge").get("port", 7777),
      models:  models.map(m => m.id || m.name),
    });
  }

  // List available models
  if (method === "POST" && url === "/list-models") {
    const models = await vscode.lm.selectChatModels({ vendor: "copilot" });
    return jsonReply(res, 200, {
      models: models.map(m => ({
        id:      m.id      || m.name,
        vendor:  m.vendor  || "copilot",
        family:  m.family  || "unknown",
        version: m.version || "unknown",
      })),
    });
  }

  // Main chat endpoint
  if (method === "POST" && url === "/chat") {
    let payload;
    try {
      payload = JSON.parse(await readBody(req));
    } catch (e) {
      return jsonReply(res, 400, { error: `Invalid JSON: ${e.message}` });
    }

    const { messages, model_family, system } = payload;
    if (!Array.isArray(messages) || messages.length === 0) {
      return jsonReply(res, 400, { error: "messages must be a non-empty array" });
    }

    let model;
    try {
      model = await selectModel(model_family);
    } catch (e) {
      return jsonReply(res, 500, { error: `Model selection failed: ${e.message}` });
    }
    if (!model) {
      return jsonReply(res, 503, {
        error: "No Copilot language model available. Make sure GitHub Copilot is signed in and active.",
      });
    }

    const lmMessages = toVscodeLmMessages(messages, system);
    const cts        = new vscode.CancellationTokenSource();
    const maxTokens  = vscode.workspace.getConfiguration("skillBridge").get("maxTokens", 4096);

    let responseText = "";
    try {
      const response = await model.sendRequest(lmMessages, { maxTokens }, cts.token);
      for await (const chunk of response.stream) {
        if (chunk instanceof vscode.LanguageModelTextPart) {
          responseText += chunk.value;
        }
      }
    } catch (e) {
      cts.dispose();
      return jsonReply(res, 500, { error: `LM request failed: ${e.message}` });
    }
    cts.dispose();
    return jsonReply(res, 200, { response: responseText, model: model.id || model.name });
  }

  return jsonReply(res, 404, { error: `Unknown endpoint: ${method} ${url}` });
}

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

function startServer(context) {
  const port = vscode.workspace.getConfiguration("skillBridge").get("port", 7777);

  if (server) {
    server.close();
    server = null;
  }

  server = http.createServer(async (req, res) => {
    // Only accept loopback connections — belt-and-suspenders security
    const remote = req.socket.remoteAddress || "";
    if (remote !== "127.0.0.1" && remote !== "::1" && remote !== "::ffff:127.0.0.1") {
      res.writeHead(403);
      res.end("Forbidden: only localhost connections accepted");
      return;
    }
    try {
      await handleRequest(req, res);
    } catch (err) {
      console.error("[skill-bridge] Unhandled error:", err);
      try { jsonReply(res, 500, { error: String(err) }); } catch (_) { /* already sent */ }
    }
  });

  server.on("error", err => {
    vscode.window.showErrorMessage(`Skill Bridge failed to start on port ${port}: ${err.message}`);
    updateStatusBar(false, port);
  });

  server.listen(port, "127.0.0.1", () => {
    console.log(`[skill-bridge] Listening on http://127.0.0.1:${port}`);
    updateStatusBar(true, port);
  });

  if (context) {
    context.subscriptions.push({ dispose: () => { if (server) { server.close(); server = null; } } });
  }
}

function updateStatusBar(running, port) {
  if (!statusBarItem) return;
  if (running) {
    statusBarItem.text     = `$(broadcast) Skill Bridge :${port}`;
    statusBarItem.tooltip  = `skill-bridge is running on http://127.0.0.1:${port}`;
    statusBarItem.color    = undefined;
  } else {
    statusBarItem.text    = `$(circle-slash) Skill Bridge (stopped)`;
    statusBarItem.tooltip = "skill-bridge is not running";
    statusBarItem.color   = new vscode.ThemeColor("statusBarItem.warningForeground");
  }
}

// ---------------------------------------------------------------------------
// Pipeline FSM
// ---------------------------------------------------------------------------

/**
 * Runs a pipeline.json definition as a state machine.
 * Lives entirely in the extension host — survives webview reloads.
 *
 * Step states: 'idle' | 'running' | 'success' | 'failure' | 'skipped'
 *
 * Output piping: if a step sets captureOutput:true, its stdout is stored in
 * this.prevOutput and injected as the PREV_OUTPUT env var into the next step.
 */
class PipelineFsm {
  constructor(pipeline, postMessage) {
    this.pipeline      = pipeline;
    this.postMessage   = postMessage;
    this.states        = {};
    this.activeProcess = null;
    this.prevOutput    = "";
    this._stopped      = false;
    for (const id of Object.keys(pipeline.steps)) {
      this.states[id] = "idle";
    }
  }

  async run() {
    this._stopped   = false;
    this.prevOutput = "";
    for (const id of Object.keys(this.pipeline.steps)) {
      this.states[id] = "idle";
    }
    this._broadcast();
    await this._runStep(this.pipeline.entry);
  }

  stop() {
    this._stopped = true;
    if (this.activeProcess) {
      try { this.activeProcess.kill(); } catch (_) {}
      this.activeProcess = null;
    }
  }

  reset() {
    this.stop();
    for (const id of Object.keys(this.pipeline.steps)) {
      this.states[id] = "idle";
    }
    this.prevOutput = "";
    this._broadcast();
  }

  _broadcast() {
    this.postMessage({ type: "fsm-state", states: { ...this.states } });
  }

  async _runStep(stepId) {
    if (!stepId || !this.pipeline.steps[stepId] || this._stopped) return;
    const step = this.pipeline.steps[stepId];
    this.states[stepId] = "running";
    this._broadcast();
    this.postMessage({ type: "step-start", stepId, label: step.label || stepId });

    const exitCode = await this._spawnStep(stepId, step);

    if (this._stopped) return;

    if (exitCode === 0) {
      this.states[stepId] = "success";
      this._broadcast();
      if (step.onSuccess) await this._runStep(step.onSuccess);
    } else {
      this.states[stepId] = "failure";
      this._broadcast();
      if (step.onFailure) await this._runStep(step.onFailure);
    }
  }

  _spawnStep(stepId, step) {
    return new Promise(resolve => {
      const wsFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || process.cwd();
      const cwd = (step.workingDir || wsFolder).replace(/\$\{workspaceFolder\}/g, wsFolder);
      const env = { ...process.env, PREV_OUTPUT: this.prevOutput };

      let proc;
      try {
        proc = cp.spawn(step.command, step.args || [], {
          cwd,
          env,
          stdio: ["ignore", "pipe", "pipe"],
        });
      } catch (err) {
        this.postMessage({ type: "step-output", stepId, text: `\nFailed to start process: ${err.message}\n` });
        this.postMessage({ type: "step-done",   stepId, exitCode: 1 });
        return resolve(1);
      }

      this.activeProcess = proc;
      let outputBuf = "";

      const onData = data => {
        const text = data.toString();
        outputBuf += text;
        this.postMessage({ type: "step-output", stepId, text });
      };
      proc.stdout.on("data", onData);
      proc.stderr.on("data", onData);

      proc.on("close", code => {
        this.activeProcess = null;
        if (step.captureOutput) this.prevOutput = outputBuf.trim();
        this.postMessage({ type: "step-done", stepId, exitCode: code ?? 1 });
        resolve(code ?? 1);
      });

      proc.on("error", err => {
        this.activeProcess = null;
        this.postMessage({ type: "step-output", stepId, text: `\nProcess error: ${err.message}\n` });
        this.postMessage({ type: "step-done",   stepId, exitCode: 1 });
        resolve(1);
      });
    });
  }
}

// ---------------------------------------------------------------------------
// Pipeline panel (webview)
// ---------------------------------------------------------------------------

let pipelinePanel = null;
let pipelineFsm   = null;

function getNonce() {
  let n = "";
  const c = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  for (let i = 0; i < 32; i++) n += c[Math.floor(Math.random() * c.length)];
  return n;
}

function getPipelineWebviewHtml(nonce) {
  return /* html */`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pipeline</title>
  <style>
    :root {
      --bg:         var(--vscode-editor-background);
      --fg:         var(--vscode-editor-foreground);
      --border:     var(--vscode-panel-border, #454545);
      --sidebar-bg: var(--vscode-sideBar-background, var(--vscode-editor-background));
      --btn-bg:     var(--vscode-button-background, #0e639c);
      --btn-fg:     var(--vscode-button-foreground, #fff);
      --btn-hover:  var(--vscode-button-hoverBackground, #1177bb);
      --btn-stop:   var(--vscode-statusBarItem-errorBackground, #c42b1c);
      --c-success:  var(--vscode-terminal-ansiGreen,  #4ec9b0);
      --c-failure:  var(--vscode-terminal-ansiRed,    #f44747);
      --c-running:  var(--vscode-terminal-ansiYellow, #cca700);
      --c-idle:     var(--vscode-descriptionForeground, #888);
      --term-bg:    var(--vscode-terminal-background, #1e1e1e);
      --term-fg:    var(--vscode-terminal-foreground, #cccccc);
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--fg);
           font-family: var(--vscode-font-family, system-ui);
           font-size: var(--vscode-font-size, 13px);
           height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

    /* ── Header ─────────────────────────────────────────────────────────── */
    #header { display: flex; align-items: center; gap: 6px; padding: 6px 10px;
              border-bottom: 1px solid var(--border); background: var(--sidebar-bg);
              flex-shrink: 0; min-height: 36px; }
    #pipeline-name { font-weight: 600; font-size: 13px; flex: 1;
                     white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    button { background: var(--btn-bg); color: var(--btn-fg); border: none;
             padding: 3px 10px; border-radius: 2px; cursor: pointer; font-size: 12px;
             line-height: 20px; }
    button:hover:not(:disabled) { background: var(--btn-hover); }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    #btn-stop { background: var(--btn-stop); }
    #run-status { font-size: 11px; color: var(--c-idle); margin-left: 4px; }

    /* ── Main layout ─────────────────────────────────────────────────────── */
    #main { display: flex; flex: 1; overflow: hidden; }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    #sidebar { width: 210px; flex-shrink: 0; border-right: 1px solid var(--border);
               background: var(--sidebar-bg); overflow-y: auto; padding: 8px 0;
               display: flex; flex-direction: column; gap: 0; }
    .step-row { display: flex; flex-direction: column; }
    .step-item { display: flex; align-items: center; gap: 8px; padding: 5px 10px;
                 cursor: pointer; user-select: none; }
    .step-item:hover  { background: var(--vscode-list-hoverBackground, rgba(255,255,255,0.05)); }
    .step-item.active { background: var(--vscode-list-activeSelectionBackground, rgba(255,255,255,0.1)); }
    .step-icon { width: 20px; height: 20px; border-radius: 50%; flex-shrink: 0;
                 display: flex; align-items: center; justify-content: center;
                 font-size: 10px; font-weight: bold; }
    .step-icon.idle    { background: var(--c-idle);    color: var(--bg); }
    .step-icon.running { background: var(--c-running); color: #000;
                         animation: blink 1s ease-in-out infinite; }
    .step-icon.success { background: var(--c-success); color: #000; }
    .step-icon.failure { background: var(--c-failure); color: #fff; }
    .step-icon.skipped { background: var(--c-idle);    color: var(--bg); opacity: 0.5; }
    @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
    .step-label { font-size: 12px; overflow: hidden; text-overflow: ellipsis;
                  white-space: nowrap; flex: 1; }
    .step-exit  { font-size: 10px; color: var(--c-idle); flex-shrink: 0; }
    .connector  { width: 2px; height: 10px; background: var(--border);
                  margin: 0 0 0 19px; flex-shrink: 0; }
    .branch-label { font-size: 10px; color: var(--c-idle); padding: 0 10px 2px 38px; }

    /* ── Terminal area ───────────────────────────────────────────────────── */
    #terminal-area { flex: 1; display: flex; flex-direction: column; overflow: hidden;
                     min-width: 0; }
    #tab-bar { display: flex; border-bottom: 1px solid var(--border);
               background: var(--sidebar-bg); overflow-x: auto; flex-shrink: 0;
               scrollbar-width: none; }
    #tab-bar::-webkit-scrollbar { display: none; }
    .tab { padding: 5px 14px; font-size: 12px; cursor: pointer; white-space: nowrap;
           border-right: 1px solid var(--border);
           color: var(--vscode-tab-inactiveForeground, #888);
           background: var(--vscode-tab-inactiveBackground, transparent);
           display: flex; align-items: center; gap: 5px; }
    .tab.active { color: var(--vscode-tab-activeForeground, var(--fg));
                  background: var(--vscode-tab-activeBackground, var(--bg));
                  border-bottom: 2px solid var(--vscode-focusBorder, #007acc); }
    .tab-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    #terminal-output { flex: 1; overflow-y: auto; background: var(--term-bg);
                       color: var(--term-fg);
                       font-family: var(--vscode-editor-font-family, "Cascadia Code", "Consolas", monospace);
                       font-size: 12px; padding: 8px 10px;
                       white-space: pre-wrap; word-break: break-all; line-height: 1.5; }
    #terminal-output::-webkit-scrollbar { width: 6px; }
    #terminal-output::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    .term-empty { color: var(--c-idle); font-style: italic; }
    /* ANSI colours */
    .ag { color: var(--vscode-terminal-ansiGreen,          #4ec9b0); }
    .ar { color: var(--vscode-terminal-ansiRed,            #f44747); }
    .ay { color: var(--vscode-terminal-ansiYellow,         #cca700); }
    .ab { color: var(--vscode-terminal-ansiBlue,           #569cd6); }
    .ac { color: var(--vscode-terminal-ansiCyan,           #9cdcfe); }
    .am { color: var(--vscode-terminal-ansiMagenta,        #c586c0); }
    .aB { font-weight: bold; }
    .aD { opacity: 0.6; }
    .aU { text-decoration: underline; }
  </style>
</head>
<body>
  <div id="header">
    <span id="pipeline-name">No pipeline loaded</span>
    <button id="btn-load"  title="Load pipeline.json">📂 Load</button>
    <button id="btn-run"   title="Run pipeline" disabled>▶ Run</button>
    <button id="btn-stop"  title="Stop" disabled>■ Stop</button>
    <button id="btn-reset" title="Reset all steps to idle" disabled>↺ Reset</button>
    <span id="run-status"></span>
  </div>
  <div id="main">
    <div id="sidebar"></div>
    <div id="terminal-area">
      <div id="tab-bar"></div>
      <div id="terminal-output"><span class="term-empty">Load a pipeline.json to begin.</span></div>
    </div>
  </div>

  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    let pipeline  = null;
    let states    = {};
    let outputs   = {};   // stepId → raw string
    let exitCodes = {};   // stepId → number | null
    let activeTab = null;

    // ── ANSI → HTML ──────────────────────────────────────────────────────────
    const ANSI_CLASSES = {
      '1':'aB','2':'aD','4':'aU',
      '31':'ar','32':'ag','33':'ay','34':'ab','35':'am','36':'ac',
      '91':'ar','92':'ag','93':'ay','94':'ab','95':'am','96':'ac',
    };
    function ansiToHtml(raw) {
      // Escape HTML entities first
      let s = raw.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      // Replace ANSI SGR sequences with spans
      let openSpans = 0;
      s = s.replace(/\\x1b\\[([0-9;]*)m/g, (_, codes) => {
        if (!codes || codes === '0') {
          let close = openSpans > 0 ? '</span>'.repeat(openSpans) : '';
          openSpans = 0;
          return close;
        }
        const cls = codes.split(';').map(c => ANSI_CLASSES[c]).filter(Boolean).join(' ');
        if (!cls) return '';
        openSpans++;
        return \`<span class="\${cls}">\`;
      });
      if (openSpans > 0) s += '</span>'.repeat(openSpans);
      return s;
    }

    // ── Step ordering ────────────────────────────────────────────────────────
    function getOrderedSteps() {
      if (!pipeline) return [];
      const order = [], visited = new Set();
      let cur = pipeline.entry;
      while (cur && pipeline.steps[cur] && !visited.has(cur)) {
        order.push(cur); visited.add(cur);
        const s = pipeline.steps[cur];
        cur = s.onSuccess || s.onFailure || null;
      }
      for (const id of Object.keys(pipeline.steps)) {
        if (!visited.has(id)) order.push(id);
      }
      return order;
    }

    // ── Colour helper ────────────────────────────────────────────────────────
    const STATE_COLOR = {
      idle:'var(--c-idle)', running:'var(--c-running)',
      success:'var(--c-success)', failure:'var(--c-failure)', skipped:'var(--c-idle)'
    };
    const STATE_ICON  = { idle:'○', running:'●', success:'✓', failure:'✗', skipped:'—' };

    // ── Sidebar ──────────────────────────────────────────────────────────────
    function renderSidebar() {
      if (!pipeline) return;
      const el = document.getElementById('sidebar');
      const ids = getOrderedSteps();
      el.innerHTML = ids.map((id, i) => {
        const step  = pipeline.steps[id];
        const state = states[id] || 'idle';
        const icon  = STATE_ICON[state] || '○';
        const exit  = exitCodes[id] != null ? exitCodes[id] : '';
        const branchHint = (state === 'success' && step.onSuccess)
          ? \`<div class="branch-label">→ \${step.onSuccess}</div>\`
          : (state === 'failure' && step.onFailure)
          ? \`<div class="branch-label">→ \${step.onFailure}</div>\`
          : '';
        return \`<div class="step-row">
          \${i > 0 ? '<div class="connector"></div>' : ''}
          <div class="step-item\${activeTab === id ? ' active':''}" onclick="selectTab('\${id}')">
            <div class="step-icon \${state}">\${icon}</div>
            <span class="step-label" title="\${step.label||id}">\${step.label||id}</span>
            \${exit !== '' ? \`<span class="step-exit">\${exit === 0 ? '✓':'✗\${exit}'}</span>\` : ''}
          </div>
          \${branchHint}
        </div>\`;
      }).join('');
    }

    // ── Tab bar ──────────────────────────────────────────────────────────────
    function renderTabs() {
      if (!pipeline) return;
      const el  = document.getElementById('tab-bar');
      const ids = getOrderedSteps();
      el.innerHTML = ids.map(id => {
        const step  = pipeline.steps[id];
        const state = states[id] || 'idle';
        const color = STATE_COLOR[state];
        return \`<div class="tab\${activeTab === id ? ' active':''}" onclick="selectTab('\${id}')">
          <span class="tab-dot" style="background:\${color}"></span>\${step.label||id}
        </div>\`;
      }).join('');
    }

    // ── Terminal ─────────────────────────────────────────────────────────────
    function renderTerminal(id) {
      const el  = document.getElementById('terminal-output');
      const raw = outputs[id] || '';
      if (!raw) {
        el.innerHTML = \`<span class="term-empty">No output yet for "\${pipeline?.steps[id]?.label||id}".</span>\`;
      } else {
        el.innerHTML = ansiToHtml(raw);
      }
    }

    function selectTab(id) {
      activeTab = id;
      renderSidebar();
      renderTabs();
      renderTerminal(id);
      // Scroll to bottom
      const el = document.getElementById('terminal-output');
      el.scrollTop = el.scrollHeight;
    }

    // ── Controls state ───────────────────────────────────────────────────────
    function updateControls() {
      const anyRunning = Object.values(states).some(s => s === 'running');
      const loaded = !!pipeline;
      document.getElementById('btn-run').disabled   = !loaded || anyRunning;
      document.getElementById('btn-stop').disabled  = !anyRunning;
      document.getElementById('btn-reset').disabled = !loaded;
      const statusEl = document.getElementById('run-status');
      if (anyRunning) {
        const runningId = Object.entries(states).find(([,v]) => v === 'running')?.[0];
        statusEl.textContent = runningId ? \`Running: \${pipeline.steps[runningId]?.label||runningId}\` : 'Running…';
      } else if (loaded) {
        const done   = Object.values(states).filter(s => s === 'success').length;
        const failed = Object.values(states).filter(s => s === 'failure').length;
        const total  = Object.keys(states).length;
        statusEl.textContent = done + failed > 0
          ? \`\${done}/\${total} passed\${failed ? \`, \${failed} failed\` : ''}\`
          : '';
      } else {
        statusEl.textContent = '';
      }
    }

    // ── Message handler ──────────────────────────────────────────────────────
    window.addEventListener('message', e => {
      const msg = e.data;

      if (msg.type === 'pipeline-loaded') {
        pipeline = msg.pipeline;
        states = {}; outputs = {}; exitCodes = {};
        for (const id of Object.keys(pipeline.steps)) {
          states[id] = 'idle'; outputs[id] = ''; exitCodes[id] = null;
        }
        activeTab = pipeline.entry;
        document.getElementById('pipeline-name').textContent = pipeline.name || 'Pipeline';
        renderSidebar(); renderTabs(); renderTerminal(activeTab);
        updateControls();
      }

      else if (msg.type === 'fsm-state') {
        states = msg.states;
        renderSidebar(); renderTabs();
        updateControls();
        if (activeTab) renderTerminal(activeTab);
      }

      else if (msg.type === 'step-start') {
        outputs[msg.stepId]   = outputs[msg.stepId] || '';
        exitCodes[msg.stepId] = null;
        // Auto-switch to the running step
        selectTab(msg.stepId);
      }

      else if (msg.type === 'step-output') {
        outputs[msg.stepId] = (outputs[msg.stepId] || '') + msg.text;
        if (activeTab === msg.stepId) {
          const el = document.getElementById('terminal-output');
          const atBottom = el.scrollHeight - el.clientHeight <= el.scrollTop + 40;
          renderTerminal(msg.stepId);
          if (atBottom) el.scrollTop = el.scrollHeight;
        }
      }

      else if (msg.type === 'step-done') {
        exitCodes[msg.stepId] = msg.exitCode;
        if (activeTab === msg.stepId) renderSidebar();
      }
    });

    // ── Button handlers ──────────────────────────────────────────────────────
    document.getElementById('btn-run').addEventListener('click',
      () => vscode.postMessage({ type: 'run' }));
    document.getElementById('btn-stop').addEventListener('click',
      () => vscode.postMessage({ type: 'stop' }));
    document.getElementById('btn-reset').addEventListener('click',
      () => { outputs = {}; exitCodes = {}; vscode.postMessage({ type: 'reset' }); });
    document.getElementById('btn-load').addEventListener('click',
      () => vscode.postMessage({ type: 'load-file' }));
  </script>
</body>
</html>`;
}

async function openPipelinePanel(context) {
  // Reuse existing panel if open
  if (pipelinePanel) {
    pipelinePanel.reveal(vscode.ViewColumn.One);
    return;
  }

  pipelinePanel = vscode.window.createWebviewPanel(
    "skillPipeline",
    "Skill Pipeline",
    vscode.ViewColumn.One,
    { enableScripts: true, retainContextWhenHidden: true },
  );

  const nonce = getNonce();
  pipelinePanel.webview.html = getPipelineWebviewHtml(nonce);

  // Post a message from the FSM to the webview
  const post = msg => pipelinePanel?.webview.postMessage(msg);

  // Load pipeline.json from workspace if present
  async function tryLoadPipeline(uri) {
    try {
      const raw  = fs.readFileSync(uri.fsPath, "utf8");
      const data = JSON.parse(raw);
      if (!data.entry || !data.steps) throw new Error("Missing 'entry' or 'steps' in pipeline.json");
      pipelineFsm = new PipelineFsm(data, post);
      post({ type: "pipeline-loaded", pipeline: data });
      pipelinePanel.title = `Pipeline: ${data.name || path.basename(uri.fsPath)}`;
    } catch (err) {
      vscode.window.showErrorMessage(`Failed to load pipeline: ${err.message}`);
    }
  }

  // Try workspace root pipeline.json automatically
  const wsFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  const defaultPath = wsFolder ? path.join(wsFolder, "pipeline.json") : null;
  if (defaultPath && fs.existsSync(defaultPath)) {
    await tryLoadPipeline(vscode.Uri.file(defaultPath));
  }

  // Handle messages from the webview
  pipelinePanel.webview.onDidReceiveMessage(async msg => {
    if (msg.type === "run") {
      if (pipelineFsm) pipelineFsm.run().catch(err =>
        vscode.window.showErrorMessage(`Pipeline error: ${err.message}`));
    } else if (msg.type === "stop") {
      if (pipelineFsm) pipelineFsm.stop();
    } else if (msg.type === "reset") {
      if (pipelineFsm) pipelineFsm.reset();
    } else if (msg.type === "load-file") {
      const uris = await vscode.window.showOpenDialog({
        canSelectMany: false,
        filters: { "Pipeline JSON": ["json"] },
        title: "Select pipeline.json",
      });
      if (uris?.[0]) await tryLoadPipeline(uris[0]);
    }
  }, undefined, context.subscriptions);

  pipelinePanel.onDidDispose(() => {
    if (pipelineFsm) pipelineFsm.stop();
    pipelinePanel = null;
    pipelineFsm   = null;
  }, null, context.subscriptions);
}

// ---------------------------------------------------------------------------
// Extension entry points
// ---------------------------------------------------------------------------

function activate(context) {
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBarItem.command = "skillBridge.showStatus";
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  startServer(context);

  context.subscriptions.push(
    vscode.commands.registerCommand("skillBridge.showStatus", async () => {
      const port   = vscode.workspace.getConfiguration("skillBridge").get("port", 7777);
      const models = await vscode.lm.selectChatModels({ vendor: "copilot" }).catch(() => []);
      const info   = [
        `Port:   ${port}`,
        `Mode:   ${server?.listening ? "running" : "stopped"}`,
        `Models: ${models.length > 0 ? models.map(m => m.id || m.name).join(", ") : "none detected"}`,
      ].join("\n");
      vscode.window.showInformationMessage(`Skill Bridge Status\n${info}`, { modal: true });
    }),

    vscode.commands.registerCommand("skillBridge.restart", () => {
      startServer(null);
      const port = vscode.workspace.getConfiguration("skillBridge").get("port", 7777);
      vscode.window.showInformationMessage(`Skill Bridge restarted on port ${port}.`);
    }),

    vscode.commands.registerCommand("skillBridge.openPipeline", () => {
      openPipelinePanel(context);
    }),

    vscode.workspace.onDidChangeConfiguration(e => {
      if (e.affectsConfiguration("skillBridge.port")) {
        startServer(null);
      }
    }),
  );
}

function deactivate() {
  if (pipelineFsm)   { pipelineFsm.stop(); pipelineFsm = null; }
  if (pipelinePanel) { pipelinePanel.dispose(); pipelinePanel = null; }
  if (server)        { server.close(); server = null; }
}

module.exports = { activate, deactivate };
