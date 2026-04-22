# skill-bridge — VS Code Copilot LM Bridge

A lightweight VS Code extension with three jobs:

1. **HTTP bridge** — exposes a loopback API so `skill.py` (and any local tool) can call GitHub Copilot's LM API with no API keys, no MCP servers, no external network traffic beyond what VS Code already handles
2. **MCP server** — serves a JSON-RPC 2.0 `/mcp` endpoint so any MCP-compatible client (Claude Desktop, Cursor, custom scripts) can reach Copilot on the same loopback
3. **Pipeline runner** — a VS Code webview panel that runs `skill.py` commands as a state machine, piping output between steps and streaming live terminal output

```
skill.py ──POST /chat──► skill-bridge extension ──vscode.lm.sendRequest()──► Copilot
MCP client ──POST /mcp──►        (port 7777)
Pipeline UI ──child_process──► skill.py subcommands
```

All traffic stays on `127.0.0.1`. No credentials leave VS Code.

---

## Installation

```bash
# From the skill-creator directory:
python vscode-bridge/install.py
```

Copies `vscode-bridge/` into `~/.vscode/extensions/skill-bridge-0.1.0/`.  
**Reload VS Code** (`Cmd/Ctrl+Shift+P` → *Reload Window*) for the extension to activate.

Verify it's running — look for `$(broadcast) Skill Bridge :7777` in the status bar, then:

```bash
python skill.py bridge-status
```

---

## Usage

### skill.py CLI

```bash
# Eval a skill's trigger description against a query set
python skill.py eval  --skill-path path/to/skill --eval-set evals/set.json

# Run the full optimisation loop (eval → improve → repeat)
python skill.py loop  --skill-path path/to/skill --eval-set evals/set.json --model gpt-4o

# With pseudo-RAG: inject relevant chunks from local docs into every prompt
python skill.py loop  --skill-path path/to/skill --eval-set evals/set.json \
                      --context-dir path/to/docs

# Use a non-default port
python skill.py --bridge-port 7778 eval --skill-path path/to/skill --eval-set evals/set.json
```

### Pipeline runner (VS Code UI)

Open the Command Palette (`Cmd/Ctrl+Shift+P`) → **Skill Bridge: Open Pipeline Runner**.

The panel auto-loads `pipeline.json` from the workspace root if present. Use `pipeline.example.json` as a starting point — copy it to `pipeline.json` and customise:

```jsonc
{
  "name": "Skill Eval Loop",
  "entry": "validate",
  "steps": {
    "validate": {
      "label": "Validate Skill",
      "command": "python",
      "args": ["skill.py", "validate", "."],
      "workingDir": "${workspaceFolder}",
      "onSuccess": "eval",
      "onFailure": null,
      "captureOutput": false
    },
    "eval": {
      "label": "Run Eval",
      "command": "python",
      "args": ["skill.py", "eval", "--skill-path", ".", "--eval-set", "evals/set.json"],
      "onSuccess": "loop",
      "onFailure": "loop",
      "captureOutput": true        // stdout becomes $PREV_OUTPUT in the next step
    }
    // ...
  }
}
```

**Step schema**

| Field | Type | Required | Description |
|---|---|---|---|
| `label` | string | no | Display name in the UI (falls back to step ID) |
| `command` | string | yes | Executable (e.g. `python`, `node`, `bash`) |
| `args` | string[] | no | CLI arguments |
| `workingDir` | string | no | Working directory; `${workspaceFolder}` expands to the VS Code workspace root |
| `onSuccess` | string\|null | no | Step ID to run on exit code 0 |
| `onFailure` | string\|null | no | Step ID to run on non-zero exit |
| `captureOutput` | bool | no | If true, stdout is stored and injected as `$PREV_OUTPUT` into the next step's environment |

### MCP client

Point any HTTP-transport MCP client at the bridge:

```jsonc
// .vscode/mcp.json  (or equivalent client config)
{
  "servers": {
    "copilot-bridge": {
      "type": "http",
      "url": "http://127.0.0.1:7777/mcp"
    }
  }
}
```

---

## HTTP API

All endpoints bind to `127.0.0.1` only. Non-loopback connections receive `403 Forbidden`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/status` | Health check — returns port and available model IDs |
| `POST` | `/chat` | Send a message thread, receive a completion |
| `POST` | `/list-models` | List available Copilot models |
| `POST` | `/mcp` | JSON-RPC 2.0 MCP endpoint (tools + resources) |

### POST /chat

```json
{
  "messages": [{"role": "user", "content": "Hello!"}],
  "model_family": "gpt-4o",
  "system": "Optional system prompt"
}
```

```json
{"response": "Hi there! How can I help?", "model": "copilot/gpt-4o"}
```

### POST /mcp — MCP tools

| Tool | Description |
|---|---|
| `copilot_chat` | Send a message thread; returns the completion text |
| `copilot_list_models` | List available Copilot models |

### GET /mcp — MCP resources

| URI | Description |
|---|---|
| `copilot://models` | JSON array of available model objects |

---

## VS Code settings

| Setting | Default | Description |
|---|---|---|
| `skillBridge.port` | `7777` | HTTP port (change if 7777 is taken) |
| `skillBridge.modelFamily` | `gpt-4o` | Default Copilot model family |
| `skillBridge.maxTokens` | `4096` | Max tokens per request |

---

## Security

- The HTTP server binds **only to `127.0.0.1`** and rejects non-loopback connections at the socket level
- Copilot authentication is handled entirely by VS Code — no tokens or credentials are ever handled by this extension
- The pipeline runner spawns child processes with `stdio: ['ignore', 'pipe', 'pipe']` — no TTY, no shell injection (args are passed as an array, never interpolated into a shell command string)

---

## Troubleshooting

**"No Copilot language model available"** — GitHub Copilot must be installed, signed in, and the subscription active. Run *GitHub Copilot: Sign In* from the Command Palette.

**"Connection refused on port 7777"** — The extension hasn't activated yet. Check the status bar for `$(broadcast) Skill Bridge :7777`. If it shows `$(circle-slash)`, run *Skill Bridge: Restart Server* from the Command Palette.

**Port conflict** — Change `skillBridge.port` in VS Code settings and pass `--bridge-port <n>` to `skill.py`.

**Pipeline step hangs** — Add a timeout wrapper to the step's `command`/`args` (e.g. `["timeout", "60", "python", "skill.py", "eval", ...]` on Linux/macOS). A per-step timeout field is on the roadmap.

**Pipeline panel blank after reload** — The webview is recreated on reveal; click *Load* and re-select your `pipeline.json`, or place it at the workspace root for auto-load.

