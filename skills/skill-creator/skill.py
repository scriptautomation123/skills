#!/usr/bin/env python3
"""skill.py — unified CLI for the skill-creator toolchain.

Commands
--------
  validate   Validate a skill directory
  package    Package a skill into a distributable .skill file
  eval       Run trigger-detection evaluation for a skill description
  improve    Improve a skill description using Claude (one iteration)
  loop       Run the full eval+improve optimization loop
  report     Generate an HTML report from loop output
  benchmark  Aggregate grading.json files into benchmark statistics
  review     Serve an interactive eval-output review viewer in the browser

Examples
--------
  python skill.py validate  path/to/my-skill
  python skill.py package   path/to/my-skill
  python skill.py eval      --skill-path path/to/my-skill --eval-set evals/set.json
  python skill.py loop      --skill-path path/to/my-skill --eval-set evals/set.json --model claude-opus-4-5
  python skill.py report    results/loop_output.json -o results/report.html
  python skill.py benchmark benchmarks/2026-01-15T10-30-00/
  python skill.py review    workspaces/my-workspace/

No dependencies beyond the Python standard library are required (except pyyaml
for the validate/package commands, which is a lightweight install).
"""

# =============================================================================
# Standard-library imports shared by all commands
# =============================================================================

import argparse
import base64
import fnmatch
import html as html_lib
import json
import math
import mimetypes
import os
import random
import re
import select
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import webbrowser
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# =============================================================================
# ── Shared utility ────────────────────────────────────────────────────────────
# =============================================================================

def parse_skill_md(skill_path: Path) -> tuple[str, str, str]:
    """Parse a SKILL.md file and return (name, description, full_content)."""
    content = (skill_path / "SKILL.md").read_text()
    lines = content.split("\n")

    if lines[0].strip() != "---":
        raise ValueError("SKILL.md missing frontmatter (no opening ---)")

    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        raise ValueError("SKILL.md missing frontmatter (no closing ---)")

    name = ""
    description = ""
    frontmatter_lines = lines[1:end_idx]
    i = 0
    while i < len(frontmatter_lines):
        line = frontmatter_lines[i]
        if line.startswith("name:"):
            name = line[len("name:"):].strip().strip('"').strip("'")
        elif line.startswith("description:"):
            value = line[len("description:"):].strip()
            if value in (">", "|", ">-", "|-"):
                continuation: list[str] = []
                i += 1
                while i < len(frontmatter_lines) and (
                    frontmatter_lines[i].startswith("  ")
                    or frontmatter_lines[i].startswith("\t")
                ):
                    continuation.append(frontmatter_lines[i].strip())
                    i += 1
                description = " ".join(continuation)
                continue
            else:
                description = value.strip('"').strip("'")
        i += 1

    return name, description, content


# =============================================================================
# ── validate ──────────────────────────────────────────────────────────────────
# =============================================================================

def _validate_skill(skill_path: Path) -> tuple[bool, str]:
    """Return (is_valid, message) for the skill at skill_path."""
    try:
        import yaml
    except ImportError:
        return False, "pyyaml is not installed — run: pip install pyyaml"

    skill_path = Path(skill_path)
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        return False, "SKILL.md not found"

    content = skill_md.read_text()
    if not content.startswith("---"):
        return False, "No YAML frontmatter found"

    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return False, "Invalid frontmatter format"

    try:
        frontmatter = yaml.safe_load(match.group(1))
        if not isinstance(frontmatter, dict):
            return False, "Frontmatter must be a YAML dictionary"
    except yaml.YAMLError as e:
        return False, f"Invalid YAML in frontmatter: {e}"

    ALLOWED = {"name", "description", "license", "allowed-tools", "metadata", "compatibility"}
    unexpected = set(frontmatter.keys()) - ALLOWED
    if unexpected:
        return False, (
            f"Unexpected key(s) in frontmatter: {', '.join(sorted(unexpected))}. "
            f"Allowed: {', '.join(sorted(ALLOWED))}"
        )

    if "name" not in frontmatter:
        return False, "Missing 'name' in frontmatter"
    if "description" not in frontmatter:
        return False, "Missing 'description' in frontmatter"

    name = frontmatter.get("name", "")
    if not isinstance(name, str):
        return False, f"Name must be a string, got {type(name).__name__}"
    name = name.strip()
    if name:
        if not re.match(r"^[a-z0-9-]+$", name):
            return False, f"Name '{name}' should be kebab-case (lowercase letters, digits, hyphens)"
        if name.startswith("-") or name.endswith("-") or "--" in name:
            return False, f"Name '{name}' cannot start/end with hyphen or contain consecutive hyphens"
        if len(name) > 64:
            return False, f"Name is too long ({len(name)} chars). Max 64."

    description = frontmatter.get("description", "")
    if not isinstance(description, str):
        return False, f"Description must be a string, got {type(description).__name__}"
    description = description.strip()
    if description:
        if "<" in description or ">" in description:
            return False, "Description cannot contain angle brackets (< or >)"
        if len(description) > 1024:
            return False, f"Description is too long ({len(description)} chars). Max 1024."

    compatibility = frontmatter.get("compatibility", "")
    if compatibility:
        if not isinstance(compatibility, str):
            return False, f"Compatibility must be a string, got {type(compatibility).__name__}"
        if len(compatibility) > 500:
            return False, f"Compatibility is too long ({len(compatibility)} chars). Max 500."

    return True, "Skill is valid!"


def cmd_validate(args: argparse.Namespace) -> int:
    valid, message = _validate_skill(Path(args.skill_path))
    print(message)
    return 0 if valid else 1


# =============================================================================
# ── package ───────────────────────────────────────────────────────────────────
# =============================================================================

_EXCLUDE_DIRS  = {"__pycache__", "node_modules"}
_EXCLUDE_GLOBS = {"*.pyc"}
_EXCLUDE_FILES = {".DS_Store"}
_ROOT_EXCLUDE_DIRS = {"evals"}


def _should_exclude(rel_path: Path) -> bool:
    parts = rel_path.parts
    if any(part in _EXCLUDE_DIRS for part in parts):
        return True
    if len(parts) > 1 and parts[1] in _ROOT_EXCLUDE_DIRS:
        return True
    name = rel_path.name
    if name in _EXCLUDE_FILES:
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in _EXCLUDE_GLOBS)


def cmd_package(args: argparse.Namespace) -> int:
    skill_path = Path(args.skill_path).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    if not skill_path.exists():
        print(f"❌ Skill folder not found: {skill_path}")
        return 1
    if not skill_path.is_dir():
        print(f"❌ Not a directory: {skill_path}")
        return 1
    if not (skill_path / "SKILL.md").exists():
        print(f"❌ SKILL.md not found in {skill_path}")
        return 1

    print("🔍 Validating skill...")
    valid, message = _validate_skill(skill_path)
    if not valid:
        print(f"❌ Validation failed: {message}")
        print("   Please fix the validation errors before packaging.")
        return 1
    print(f"✅ {message}\n")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path.cwd()

    skill_filename = output_dir / f"{skill_path.name}.skill"

    try:
        with zipfile.ZipFile(skill_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in skill_path.rglob("*"):
                if not file_path.is_file():
                    continue
                arcname = file_path.relative_to(skill_path.parent)
                if _should_exclude(arcname):
                    print(f"  Skipped: {arcname}")
                    continue
                zipf.write(file_path, arcname)
                print(f"  Added:   {arcname}")
        print(f"\n✅ Packaged to: {skill_filename}")
        return 0
    except Exception as e:
        print(f"❌ Error creating .skill file: {e}")
        return 1


# =============================================================================
# ── eval ──────────────────────────────────────────────────────────────────────
# =============================================================================

def _find_project_root() -> Path:
    """Walk up from cwd looking for .claude/ (mirrors Claude Code discovery)."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".claude").is_dir():
            return parent
    return current


def _run_single_query(
    query: str,
    skill_name: str,
    skill_description: str,
    timeout: int,
    project_root: str,
    model: str | None = None,
) -> bool:
    """Run one query and return True if the skill was triggered."""
    unique_id = uuid.uuid4().hex[:8]
    clean_name = f"{skill_name}-skill-{unique_id}"
    commands_dir = Path(project_root) / ".claude" / "commands"
    command_file = commands_dir / f"{clean_name}.md"

    try:
        commands_dir.mkdir(parents=True, exist_ok=True)
        indented_desc = "\n  ".join(skill_description.split("\n"))
        command_file.write_text(
            f"---\ndescription: |\n  {indented_desc}\n---\n\n"
            f"# {skill_name}\n\nThis skill handles: {skill_description}\n"
        )

        cmd = [
            "claude", "-p", query,
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
        ]
        if model:
            cmd.extend(["--model", model])

        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=project_root, env=env,
        )

        triggered = False
        start_time = time.time()
        buffer = ""
        pending_tool_name = None
        accumulated_json = ""

        try:
            while time.time() - start_time < timeout:
                if process.poll() is not None:
                    remaining = process.stdout.read()
                    if remaining:
                        buffer += remaining.decode("utf-8", errors="replace")
                    break

                ready, _, _ = select.select([process.stdout], [], [], 1.0)
                if not ready:
                    continue

                chunk = os.read(process.stdout.fileno(), 8192)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if event.get("type") == "stream_event":
                        se = event.get("event", {})
                        se_type = se.get("type", "")

                        if se_type == "content_block_start":
                            cb = se.get("content_block", {})
                            if cb.get("type") == "tool_use":
                                tool_name = cb.get("name", "")
                                if tool_name in ("Skill", "Read"):
                                    pending_tool_name = tool_name
                                    accumulated_json = ""
                                else:
                                    return False

                        elif se_type == "content_block_delta" and pending_tool_name:
                            delta = se.get("delta", {})
                            if delta.get("type") == "input_json_delta":
                                accumulated_json += delta.get("partial_json", "")
                                if clean_name in accumulated_json:
                                    return True

                        elif se_type in ("content_block_stop", "message_stop"):
                            if pending_tool_name:
                                return clean_name in accumulated_json
                            if se_type == "message_stop":
                                return False

                    elif event.get("type") == "assistant":
                        for item in event.get("message", {}).get("content", []):
                            if item.get("type") != "tool_use":
                                continue
                            tool_name = item.get("name", "")
                            tool_input = item.get("input", {})
                            if tool_name == "Skill" and clean_name in tool_input.get("skill", ""):
                                triggered = True
                            elif tool_name == "Read" and clean_name in tool_input.get("file_path", ""):
                                triggered = True
                            return triggered

                    elif event.get("type") == "result":
                        return triggered
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

        return triggered
    finally:
        if command_file.exists():
            command_file.unlink()


def _run_eval(
    eval_set: list[dict],
    skill_name: str,
    description: str,
    num_workers: int,
    timeout: int,
    project_root: Path,
    runs_per_query: int = 1,
    trigger_threshold: float = 0.5,
    model: str | None = None,
) -> dict:
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_info: dict = {}
        for item in eval_set:
            for run_idx in range(runs_per_query):
                future = executor.submit(
                    _run_single_query,
                    item["query"], skill_name, description,
                    timeout, str(project_root), model,
                )
                future_to_info[future] = (item, run_idx)

        query_triggers: dict[str, list[bool]] = {}
        query_items: dict[str, dict] = {}
        for future in as_completed(future_to_info):
            item, _ = future_to_info[future]
            query = item["query"]
            query_items[query] = item
            query_triggers.setdefault(query, [])
            try:
                query_triggers[query].append(future.result())
            except Exception as e:
                print(f"Warning: query failed: {e}", file=sys.stderr)
                query_triggers[query].append(False)

    for query, triggers in query_triggers.items():
        item = query_items[query]
        trigger_rate = sum(triggers) / len(triggers)
        should_trigger = item["should_trigger"]
        did_pass = (trigger_rate >= trigger_threshold) if should_trigger else (trigger_rate < trigger_threshold)
        results.append({
            "query": query,
            "should_trigger": should_trigger,
            "trigger_rate": trigger_rate,
            "triggers": sum(triggers),
            "runs": len(triggers),
            "pass": did_pass,
        })

    passed = sum(1 for r in results if r["pass"])
    total = len(results)

    return {
        "skill_name": skill_name,
        "description": description,
        "results": results,
        "summary": {"total": total, "passed": passed, "failed": total - passed},
    }


def cmd_eval(args: argparse.Namespace) -> int:
    eval_set = json.loads(Path(args.eval_set).read_text())
    skill_path = Path(args.skill_path)

    if not (skill_path / "SKILL.md").exists():
        print(f"Error: No SKILL.md found at {skill_path}", file=sys.stderr)
        return 1

    name, original_description, _ = parse_skill_md(skill_path)
    description = args.description or original_description
    project_root = _find_project_root()

    if args.verbose:
        print(f"Evaluating: {description}", file=sys.stderr)

    output = _run_eval(
        eval_set=eval_set,
        skill_name=name,
        description=description,
        num_workers=args.num_workers,
        timeout=args.timeout,
        project_root=project_root,
        runs_per_query=args.runs_per_query,
        trigger_threshold=args.trigger_threshold,
        model=args.model,
    )

    if args.verbose:
        s = output["summary"]
        print(f"Results: {s['passed']}/{s['total']} passed", file=sys.stderr)
        for r in output["results"]:
            status = "PASS" if r["pass"] else "FAIL"
            print(f"  [{status}] {r['triggers']}/{r['runs']}  {r['query'][:70]}", file=sys.stderr)

    print(json.dumps(output, indent=2))
    return 0


# =============================================================================
# ── improve ───────────────────────────────────────────────────────────────────
# =============================================================================

def _call_claude(prompt: str, model: str | None, timeout: int = 300) -> str:
    cmd = ["claude", "-p", "--output-format", "text"]
    if model:
        cmd.extend(["--model", model])
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    result = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, env=env, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude -p exited {result.returncode}\nstderr: {result.stderr}")
    return result.stdout


def _improve_description(
    skill_name: str,
    skill_content: str,
    current_description: str,
    eval_results: dict,
    history: list[dict],
    model: str,
    test_results: dict | None = None,
    log_dir: Path | None = None,
    iteration: int | None = None,
) -> str:
    failed_triggers = [r for r in eval_results["results"] if r["should_trigger"] and not r["pass"]]
    false_triggers  = [r for r in eval_results["results"] if not r["should_trigger"] and not r["pass"]]

    train_score = f"{eval_results['summary']['passed']}/{eval_results['summary']['total']}"
    if test_results:
        test_score = f"{test_results['summary']['passed']}/{test_results['summary']['total']}"
        scores_summary = f"Train: {train_score}, Test: {test_score}"
    else:
        scores_summary = f"Train: {train_score}"

    prompt = (
        f'You are optimizing a skill description for a Claude Code skill called "{skill_name}". '
        f"A \"skill\" is sort of like a prompt, but with progressive disclosure -- there's a title "
        f"and description that Claude sees when deciding whether to use the skill, and then if it "
        f"does use the skill, it reads the .md file which has lots more details and potentially "
        f"links to other resources in the skill folder like helper files and scripts and additional "
        f"documentation or examples.\n\n"
        f"The description appears in Claude's \"available_skills\" list. When a user sends a query, "
        f"Claude decides whether to invoke the skill based solely on the title and on this "
        f"description. Your goal is to write a description that triggers for relevant queries, and "
        f"doesn't trigger for irrelevant ones.\n\n"
        f"Here's the current description:\n<current_description>\n\"{current_description}\"\n</current_description>\n\n"
        f"Current scores ({scores_summary}):\n<scores_summary>\n"
    )

    if failed_triggers:
        prompt += "FAILED TO TRIGGER (should have triggered but didn't):\n"
        for r in failed_triggers:
            prompt += f'  - "{r["query"]}" (triggered {r["triggers"]}/{r["runs"]} times)\n'
        prompt += "\n"

    if false_triggers:
        prompt += "FALSE TRIGGERS (triggered but shouldn't have):\n"
        for r in false_triggers:
            prompt += f'  - "{r["query"]}" (triggered {r["triggers"]}/{r["runs"]} times)\n'
        prompt += "\n"

    if history:
        prompt += "PREVIOUS ATTEMPTS (do NOT repeat these — try something structurally different):\n\n"
        for h in history:
            train_s = f"{h.get('train_passed', h.get('passed', 0))}/{h.get('train_total', h.get('total', 0))}"
            test_s = (
                f"{h.get('test_passed', '?')}/{h.get('test_total', '?')}"
                if h.get("test_passed") is not None else None
            )
            score_str = f"train={train_s}" + (f", test={test_s}" if test_s else "")
            prompt += f'<attempt {score_str}>\nDescription: "{h["description"]}"\n'
            if "results" in h:
                prompt += "Train results:\n"
                for r in h["results"]:
                    status = "PASS" if r["pass"] else "FAIL"
                    prompt += f'  [{status}] "{r["query"][:80]}" (triggered {r["triggers"]}/{r["runs"]})\n'
            if h.get("note"):
                prompt += f'Note: {h["note"]}\n'
            prompt += "</attempt>\n\n"

    prompt += (
        f"</scores_summary>\n\n"
        f"Skill content (for context on what the skill does):\n<skill_content>\n{skill_content}\n</skill_content>\n\n"
        f"Based on the failures, write a new and improved description that is more likely to trigger "
        f"correctly. When I say \"based on the failures\", it's a bit of a tricky line to walk because "
        f"we don't want to overfit to the specific cases you're seeing. So what I DON'T want you to do "
        f"is produce an ever-expanding list of specific queries that this skill should or shouldn't "
        f"trigger for. Instead, try to generalize from the failures to broader categories of user intent "
        f"and situations where this skill would be useful or not useful. The reason for this is twofold:\n\n"
        f"1. Avoid overfitting\n"
        f"2. The list might get loooong and it's injected into ALL queries and there might be a lot of "
        f"skills, so we don't want to blow too much space on any given description.\n\n"
        f"Concretely, your description should not be more than about 100-200 words, even if that comes "
        f"at the cost of accuracy. There is a hard limit of 1024 characters — descriptions over that "
        f"will be truncated, so stay comfortably under it.\n\n"
        f"Here are some tips that we've found to work well in writing these descriptions:\n"
        f"- The skill should be phrased in the imperative -- \"Use this skill for\" rather than \"this skill does\"\n"
        f"- The skill description should focus on the user's intent, what they are trying to achieve, "
        f"vs. the implementation details of how the skill works.\n"
        f"- The description competes with other skills for Claude's attention — make it distinctive and "
        f"immediately recognizable.\n"
        f"- If you're getting lots of failures after repeated attempts, change things up. Try different "
        f"sentence structures or wordings.\n\n"
        f"I'd encourage you to be creative and mix up the style in different iterations since you'll "
        f"have multiple opportunities to try different approaches and we'll just grab the highest-scoring "
        f"one at the end.\n\n"
        f"Please respond with only the new description text in <new_description> tags, nothing else."
    )

    text = _call_claude(prompt, model)
    match = re.search(r"<new_description>(.*?)</new_description>", text, re.DOTALL)
    description = match.group(1).strip().strip('"') if match else text.strip().strip('"')

    transcript: dict = {
        "iteration": iteration,
        "prompt": prompt,
        "response": text,
        "parsed_description": description,
        "char_count": len(description),
        "over_limit": len(description) > 1024,
    }

    if len(description) > 1024:
        shorten_prompt = (
            f"{prompt}\n\n---\n\n"
            f"A previous attempt produced this description, which at {len(description)} characters "
            f"is over the 1024-character hard limit:\n\n\"{description}\"\n\n"
            f"Rewrite it to be under 1024 characters while keeping the most important trigger words "
            f"and intent coverage. Respond with only the new description in <new_description> tags."
        )
        shorten_text = _call_claude(shorten_prompt, model)
        match = re.search(r"<new_description>(.*?)</new_description>", shorten_text, re.DOTALL)
        shortened = match.group(1).strip().strip('"') if match else shorten_text.strip().strip('"')
        transcript["rewrite_description"] = shortened
        description = shortened

    transcript["final_description"] = description

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"improve_iter_{iteration or 'unknown'}.json").write_text(
            json.dumps(transcript, indent=2)
        )

    return description


def cmd_improve(args: argparse.Namespace) -> int:
    skill_path = Path(args.skill_path)
    if not (skill_path / "SKILL.md").exists():
        print(f"Error: No SKILL.md found at {skill_path}", file=sys.stderr)
        return 1

    eval_results = json.loads(Path(args.eval_results).read_text())
    history: list[dict] = []
    if args.history:
        history = json.loads(Path(args.history).read_text())

    name, _, content = parse_skill_md(skill_path)
    current_description = eval_results["description"]

    if args.verbose:
        print(f"Current: {current_description}", file=sys.stderr)
        print(f"Score: {eval_results['summary']['passed']}/{eval_results['summary']['total']}", file=sys.stderr)

    new_description = _improve_description(
        skill_name=name,
        skill_content=content,
        current_description=current_description,
        eval_results=eval_results,
        history=history,
        model=args.model,
    )

    if args.verbose:
        print(f"Improved: {new_description}", file=sys.stderr)

    output = {
        "description": new_description,
        "history": history + [{
            "description": current_description,
            "passed": eval_results["summary"]["passed"],
            "failed": eval_results["summary"]["failed"],
            "total": eval_results["summary"]["total"],
            "results": eval_results["results"],
        }],
    }
    print(json.dumps(output, indent=2))
    return 0


# =============================================================================
# ── report (generate HTML from loop output) ───────────────────────────────────
# =============================================================================

def _generate_loop_report_html(data: dict, auto_refresh: bool = False, skill_name: str = "") -> str:
    """Build an HTML optimisation-report from run_loop output data."""
    history = data.get("history", [])
    title_prefix = html_lib.escape(skill_name + " \u2014 ") if skill_name else ""

    train_queries: list[dict] = []
    test_queries: list[dict] = []
    if history:
        for r in history[0].get("train_results", history[0].get("results", [])):
            train_queries.append({"query": r["query"], "should_trigger": r.get("should_trigger", True)})
        if history[0].get("test_results"):
            for r in history[0].get("test_results", []):
                test_queries.append({"query": r["query"], "should_trigger": r.get("should_trigger", True)})

    refresh_tag = '    <meta http-equiv="refresh" content="5">\n' if auto_refresh else ""

    parts = [
        f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
{refresh_tag}    <title>{title_prefix}Skill Description Optimization</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600&family=Lora:wght@400;500&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Lora', Georgia, serif; max-width: 100%; margin: 0 auto; padding: 20px; background: #faf9f5; color: #141413; }}
        h1 {{ font-family: 'Poppins', sans-serif; color: #141413; }}
        .explainer {{ background: white; padding: 15px; border-radius: 6px; margin-bottom: 20px; border: 1px solid #e8e6dc; color: #b0aea5; font-size: 0.875rem; line-height: 1.6; }}
        .summary {{ background: white; padding: 15px; border-radius: 6px; margin-bottom: 20px; border: 1px solid #e8e6dc; }}
        .summary p {{ margin: 5px 0; }}
        .best {{ color: #788c5d; font-weight: bold; }}
        .table-container {{ overflow-x: auto; width: 100%; }}
        table {{ border-collapse: collapse; background: white; border: 1px solid #e8e6dc; border-radius: 6px; font-size: 12px; min-width: 100%; }}
        th, td {{ padding: 8px; text-align: left; border: 1px solid #e8e6dc; white-space: normal; word-wrap: break-word; }}
        th {{ font-family: 'Poppins', sans-serif; background: #141413; color: #faf9f5; font-weight: 500; }}
        th.test-col {{ background: #6a9bcc; }}
        th.query-col {{ min-width: 200px; }}
        td.description {{ font-family: monospace; font-size: 11px; word-wrap: break-word; max-width: 400px; }}
        td.result {{ text-align: center; font-size: 16px; min-width: 40px; }}
        td.test-result {{ background: #f0f6fc; }}
        .pass {{ color: #788c5d; }} .fail {{ color: #c44; }}
        .rate {{ font-size: 9px; color: #b0aea5; display: block; }}
        tr:hover {{ background: #faf9f5; }}
        .score {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 11px; }}
        .score-good {{ background: #eef2e8; color: #788c5d; }}
        .score-ok  {{ background: #fef3c7; color: #d97706; }}
        .score-bad {{ background: #fceaea; color: #c44; }}
        .best-row {{ background: #f5f8f2; }}
        th.positive-col {{ border-bottom: 3px solid #788c5d; }}
        th.negative-col {{ border-bottom: 3px solid #c44; }}
        th.test-col.positive-col {{ border-bottom: 3px solid #788c5d; }}
        th.test-col.negative-col {{ border-bottom: 3px solid #c44; }}
        .legend {{ font-family: 'Poppins', sans-serif; display: flex; gap: 20px; margin-bottom: 10px; font-size: 13px; align-items: center; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .legend-swatch {{ width: 16px; height: 16px; border-radius: 3px; display: inline-block; }}
        .swatch-positive {{ background: #141413; border-bottom: 3px solid #788c5d; }}
        .swatch-negative {{ background: #141413; border-bottom: 3px solid #c44; }}
        .swatch-test {{ background: #6a9bcc; }}
        .swatch-train {{ background: #141413; }}
    </style>
</head>
<body>
    <h1>{title_prefix}Skill Description Optimization</h1>
    <div class="explainer">
        <strong>Optimizing your skill's description.</strong> Each row is an iteration — a new
        description attempt. Green ✓ means the skill triggered correctly; red ✗ means it got it wrong.
        <em>Train</em> columns were used during optimization; <em>Test</em> columns (blue) are held-out.
    </div>
"""
    ]

    best_test_score = data.get("best_test_score")
    parts.append(f"""
    <div class="summary">
        <p><strong>Original:</strong> {html_lib.escape(data.get('original_description', 'N/A'))}</p>
        <p class="best"><strong>Best:</strong> {html_lib.escape(data.get('best_description', 'N/A'))}</p>
        <p><strong>Best Score:</strong> {data.get('best_score', 'N/A')} {'(test)' if best_test_score else '(train)'}</p>
        <p><strong>Iterations:</strong> {data.get('iterations_run', 0)} | <strong>Train:</strong> {data.get('train_size', '?')} | <strong>Test:</strong> {data.get('test_size', '?')}</p>
    </div>
""")

    parts.append("""
    <div class="legend">
        <span style="font-weight:600">Query columns:</span>
        <span class="legend-item"><span class="legend-swatch swatch-positive"></span> Should trigger</span>
        <span class="legend-item"><span class="legend-swatch swatch-negative"></span> Should NOT trigger</span>
        <span class="legend-item"><span class="legend-swatch swatch-train"></span> Train</span>
        <span class="legend-item"><span class="legend-swatch swatch-test"></span> Test (held-out)</span>
    </div>
    <div class="table-container">
    <table>
        <thead>
            <tr>
                <th>Iter</th>
                <th>Train</th>
                <th>Test</th>
                <th class="query-col">Description</th>
""")

    for qinfo in train_queries:
        polarity = "positive-col" if qinfo["should_trigger"] else "negative-col"
        parts.append(f'                <th class="{polarity}">{html_lib.escape(qinfo["query"])}</th>\n')
    for qinfo in test_queries:
        polarity = "positive-col" if qinfo["should_trigger"] else "negative-col"
        parts.append(f'                <th class="test-col {polarity}">{html_lib.escape(qinfo["query"])}</th>\n')
    parts.append("            </tr>\n        </thead>\n        <tbody>\n")

    if test_queries:
        best_iter = max(history, key=lambda h: h.get("test_passed") or 0).get("iteration")
    elif history:
        best_iter = max(history, key=lambda h: h.get("train_passed", h.get("passed", 0))).get("iteration")
    else:
        best_iter = None

    def _score_class(correct: int, total: int) -> str:
        if total > 0:
            ratio = correct / total
            if ratio >= 0.8:
                return "score-good"
            elif ratio >= 0.5:
                return "score-ok"
        return "score-bad"

    for h in history:
        iteration = h.get("iteration", "?")
        train_results = h.get("train_results", h.get("results", []))
        test_results  = h.get("test_results", []) or []
        description   = h.get("description", "")

        train_by_query = {r["query"]: r for r in train_results}
        test_by_query  = {r["query"]: r for r in test_results}

        def _agg(results: list[dict]) -> tuple[int, int]:
            correct = total = 0
            for r in results:
                runs = r.get("runs", 0)
                triggers = r.get("triggers", 0)
                total += runs
                if r.get("should_trigger", True):
                    correct += triggers
                else:
                    correct += runs - triggers
            return correct, total

        train_correct, train_runs = _agg(train_results)
        test_correct,  test_runs  = _agg(test_results)
        row_class = "best-row" if iteration == best_iter else ""

        parts.append(
            f'            <tr class="{row_class}">\n'
            f'                <td>{iteration}</td>\n'
            f'                <td><span class="score {_score_class(train_correct, train_runs)}">{train_correct}/{train_runs}</span></td>\n'
            f'                <td><span class="score {_score_class(test_correct,  test_runs )}">{test_correct}/{test_runs}</span></td>\n'
            f'                <td class="description">{html_lib.escape(description)}</td>\n'
        )
        for qinfo in train_queries:
            r = train_by_query.get(qinfo["query"], {})
            icon = "✓" if r.get("pass") else "✗"
            css  = "pass" if r.get("pass") else "fail"
            parts.append(f'                <td class="result {css}">{icon}<span class="rate">{r.get("triggers",0)}/{r.get("runs",0)}</span></td>\n')
        for qinfo in test_queries:
            r = test_by_query.get(qinfo["query"], {})
            icon = "✓" if r.get("pass") else "✗"
            css  = "pass" if r.get("pass") else "fail"
            parts.append(f'                <td class="result test-result {css}">{icon}<span class="rate">{r.get("triggers",0)}/{r.get("runs",0)}</span></td>\n')
        parts.append("            </tr>\n")

    parts.append("        </tbody>\n    </table>\n    </div>\n</body>\n</html>\n")
    return "".join(parts)


def cmd_report(args: argparse.Namespace) -> int:
    if args.input == "-":
        data = json.load(sys.stdin)
    else:
        data = json.loads(Path(args.input).read_text())

    html_output = _generate_loop_report_html(data, skill_name=args.skill_name)

    if args.output:
        Path(args.output).write_text(html_output)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(html_output)
    return 0


# =============================================================================
# ── loop (eval + improve) ─────────────────────────────────────────────────────
# =============================================================================

def _split_eval_set(eval_set: list[dict], holdout: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    random.seed(seed)
    trigger    = [e for e in eval_set if     e["should_trigger"]]
    no_trigger = [e for e in eval_set if not e["should_trigger"]]
    random.shuffle(trigger)
    random.shuffle(no_trigger)
    n_trigger_test    = max(1, int(len(trigger)    * holdout))
    n_no_trigger_test = max(1, int(len(no_trigger) * holdout))
    test_set  = trigger[:n_trigger_test]    + no_trigger[:n_no_trigger_test]
    train_set = trigger[n_trigger_test:]    + no_trigger[n_no_trigger_test:]
    return train_set, test_set


def _run_loop(
    eval_set: list[dict],
    skill_path: Path,
    description_override: str | None,
    num_workers: int,
    timeout: int,
    max_iterations: int,
    runs_per_query: int,
    trigger_threshold: float,
    holdout: float,
    model: str,
    verbose: bool,
    live_report_path: Path | None = None,
    log_dir: Path | None = None,
) -> dict:
    project_root = _find_project_root()
    name, original_description, content = parse_skill_md(skill_path)
    current_description = description_override or original_description

    if holdout > 0:
        train_set, test_set = _split_eval_set(eval_set, holdout)
        if verbose:
            print(f"Split: {len(train_set)} train, {len(test_set)} test (holdout={holdout})", file=sys.stderr)
    else:
        train_set = eval_set
        test_set  = []

    history: list[dict] = []
    exit_reason = "unknown"

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n{'='*60}\nIteration {iteration}/{max_iterations}\nDescription: {current_description}\n{'='*60}", file=sys.stderr)

        all_queries = train_set + test_set
        t0 = time.time()
        all_results = _run_eval(
            eval_set=all_queries,
            skill_name=name,
            description=current_description,
            num_workers=num_workers,
            timeout=timeout,
            project_root=project_root,
            runs_per_query=runs_per_query,
            trigger_threshold=trigger_threshold,
            model=model,
        )
        eval_elapsed = time.time() - t0

        train_qs = {q["query"] for q in train_set}
        train_result_list = [r for r in all_results["results"] if     r["query"] in train_qs]
        test_result_list  = [r for r in all_results["results"] if not r["query"] in train_qs]

        train_passed = sum(1 for r in train_result_list if r["pass"])
        train_total  = len(train_result_list)
        train_summary = {"passed": train_passed, "failed": train_total - train_passed, "total": train_total}
        train_results = {"results": train_result_list, "summary": train_summary}

        if test_set:
            test_passed = sum(1 for r in test_result_list if r["pass"])
            test_total  = len(test_result_list)
            test_summary = {"passed": test_passed, "failed": test_total - test_passed, "total": test_total}
            test_results = {"results": test_result_list, "summary": test_summary}
        else:
            test_results = test_summary = None

        history.append({
            "iteration":   iteration,
            "description": current_description,
            "train_passed": train_summary["passed"],
            "train_failed": train_summary["failed"],
            "train_total":  train_summary["total"],
            "train_results": train_results["results"],
            "test_passed": test_summary["passed"]  if test_summary else None,
            "test_failed": test_summary["failed"]  if test_summary else None,
            "test_total":  test_summary["total"]   if test_summary else None,
            "test_results": test_results["results"] if test_results else None,
            # backward-compat keys for report generator
            "passed": train_summary["passed"],
            "failed": train_summary["failed"],
            "total":  train_summary["total"],
            "results": train_results["results"],
        })

        if live_report_path:
            partial_output = {
                "original_description": original_description,
                "best_description": current_description,
                "best_score": "in progress",
                "iterations_run": len(history),
                "holdout": holdout,
                "train_size": len(train_set),
                "test_size": len(test_set),
                "history": history,
            }
            live_report_path.write_text(
                _generate_loop_report_html(partial_output, auto_refresh=True, skill_name=name)
            )

        if verbose:
            def _print_stats(label: str, results: list[dict], elapsed: float) -> None:
                pos = [r for r in results if r["should_trigger"]]
                neg = [r for r in results if not r["should_trigger"]]
                tp = sum(r["triggers"] for r in pos)
                pos_runs = sum(r["runs"] for r in pos)
                fp = sum(r["triggers"] for r in neg)
                neg_runs = sum(r["runs"] for r in neg)
                tn = neg_runs - fp
                fn = pos_runs - tp
                total = tp + tn + fp + fn
                precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
                accuracy  = (tp + tn) / total if total > 0 else 0.0
                print(f"{label}: {tp+tn}/{total} correct  precision={precision:.0%} recall={recall:.0%} accuracy={accuracy:.0%} ({elapsed:.1f}s)", file=sys.stderr)
                for r in results:
                    status = "PASS" if r["pass"] else "FAIL"
                    print(f"  [{status}] {r['triggers']}/{r['runs']}  {r['query'][:60]}", file=sys.stderr)

            _print_stats("Train", train_results["results"], eval_elapsed)
            if test_summary:
                _print_stats("Test ", test_results["results"], 0)

        if train_summary["failed"] == 0:
            exit_reason = f"all_passed (iteration {iteration})"
            if verbose:
                print(f"\nAll train queries passed on iteration {iteration}!", file=sys.stderr)
            break

        if iteration == max_iterations:
            exit_reason = f"max_iterations ({max_iterations})"
            if verbose:
                print(f"\nMax iterations reached ({max_iterations}).", file=sys.stderr)
            break

        if verbose:
            print("\nImproving description...", file=sys.stderr)

        blinded_history = [
            {k: v for k, v in h.items() if not k.startswith("test_")}
            for h in history
        ]
        current_description = _improve_description(
            skill_name=name,
            skill_content=content,
            current_description=current_description,
            eval_results=train_results,
            history=blinded_history,
            model=model,
            log_dir=log_dir,
            iteration=iteration,
        )

        if verbose:
            print(f"Proposed: {current_description}", file=sys.stderr)

    if test_set:
        best = max(history, key=lambda h: h["test_passed"] or 0)
        best_score = f"{best['test_passed']}/{best['test_total']}"
    else:
        best = max(history, key=lambda h: h["train_passed"])
        best_score = f"{best['train_passed']}/{best['train_total']}"

    if verbose:
        print(f"\nExit: {exit_reason}", file=sys.stderr)
        print(f"Best: {best_score} (iteration {best['iteration']})", file=sys.stderr)

    return {
        "exit_reason":        exit_reason,
        "original_description": original_description,
        "best_description":   best["description"],
        "best_score":         best_score,
        "best_train_score":   f"{best['train_passed']}/{best['train_total']}",
        "best_test_score":    f"{best['test_passed']}/{best['test_total']}" if test_set else None,
        "final_description":  current_description,
        "iterations_run":     len(history),
        "holdout":            holdout,
        "train_size":         len(train_set),
        "test_size":          len(test_set),
        "history":            history,
    }


def cmd_loop(args: argparse.Namespace) -> int:
    eval_set  = json.loads(Path(args.eval_set).read_text())
    skill_path = Path(args.skill_path)

    if not (skill_path / "SKILL.md").exists():
        print(f"Error: No SKILL.md found at {skill_path}", file=sys.stderr)
        return 1

    name, _, _ = parse_skill_md(skill_path)

    # Set up live report
    live_report_path: Path | None = None
    if args.report != "none":
        if args.report == "auto":
            tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, prefix="skill_loop_")
            live_report_path = Path(tmp.name)
            tmp.close()
        else:
            live_report_path = Path(args.report)
        webbrowser.open(live_report_path.as_uri())

    # Output directory
    results_dir: Path | None = None
    if args.results_dir:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        results_dir = Path(args.results_dir) / ts
        results_dir.mkdir(parents=True, exist_ok=True)

    log_dir = results_dir / "logs" if results_dir else None

    output = _run_loop(
        eval_set=eval_set,
        skill_path=skill_path,
        description_override=args.description,
        num_workers=args.num_workers,
        timeout=args.timeout,
        max_iterations=args.max_iterations,
        runs_per_query=args.runs_per_query,
        trigger_threshold=args.trigger_threshold,
        holdout=args.holdout,
        model=args.model,
        verbose=args.verbose,
        live_report_path=live_report_path,
        log_dir=log_dir,
    )

    json_output = json.dumps(output, indent=2)
    print(json_output)

    if results_dir:
        (results_dir / "results.json").write_text(json_output)

    if live_report_path:
        live_report_path.write_text(
            _generate_loop_report_html(output, auto_refresh=False, skill_name=name)
        )
        if results_dir:
            (results_dir / "report.html").write_text(live_report_path.read_text())

    if results_dir:
        print(f"\nOutputs saved to: {results_dir}", file=sys.stderr)

    return 0


# =============================================================================
# ── benchmark ─────────────────────────────────────────────────────────────────
# =============================================================================

def _calculate_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0}
    n = len(values)
    mean = sum(values) / n
    stddev = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1)) if n > 1 else 0.0
    return {
        "mean":   round(mean,   4),
        "stddev": round(stddev, 4),
        "min":    round(min(values), 4),
        "max":    round(max(values), 4),
    }


def _load_run_results(benchmark_dir: Path) -> dict:
    runs_dir = benchmark_dir / "runs"
    if runs_dir.exists():
        search_dir = runs_dir
    elif list(benchmark_dir.glob("eval-*")):
        search_dir = benchmark_dir
    else:
        print(f"No eval directories found in {benchmark_dir}", file=sys.stderr)
        return {}

    results: dict[str, list] = {}

    for eval_idx, eval_dir in enumerate(sorted(search_dir.glob("eval-*"))):
        metadata_path = eval_dir / "eval_metadata.json"
        if metadata_path.exists():
            try:
                eval_id = json.loads(metadata_path.read_text()).get("eval_id", eval_idx)
            except (json.JSONDecodeError, OSError):
                eval_id = eval_idx
        else:
            try:
                eval_id = int(eval_dir.name.split("-")[1])
            except ValueError:
                eval_id = eval_idx

        for config_dir in sorted(eval_dir.iterdir()):
            if not config_dir.is_dir() or not list(config_dir.glob("run-*")):
                continue
            config = config_dir.name
            results.setdefault(config, [])

            for run_dir in sorted(config_dir.glob("run-*")):
                run_number = int(run_dir.name.split("-")[1])
                grading_file = run_dir / "grading.json"
                if not grading_file.exists():
                    print(f"Warning: grading.json not found in {run_dir}", file=sys.stderr)
                    continue
                try:
                    grading = json.loads(grading_file.read_text())
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in {grading_file}: {e}", file=sys.stderr)
                    continue

                result: dict = {
                    "eval_id":    eval_id,
                    "run_number": run_number,
                    "pass_rate":  grading.get("summary", {}).get("pass_rate", 0.0),
                    "passed":     grading.get("summary", {}).get("passed", 0),
                    "failed":     grading.get("summary", {}).get("failed", 0),
                    "total":      grading.get("summary", {}).get("total", 0),
                }
                timing = grading.get("timing", {})
                result["time_seconds"] = timing.get("total_duration_seconds", 0.0)
                timing_file = run_dir / "timing.json"
                if result["time_seconds"] == 0.0 and timing_file.exists():
                    try:
                        td = json.loads(timing_file.read_text())
                        result["time_seconds"] = td.get("total_duration_seconds", 0.0)
                        result["tokens"]       = td.get("total_tokens", 0)
                    except json.JSONDecodeError:
                        pass

                metrics = grading.get("execution_metrics", {})
                result["tool_calls"] = metrics.get("total_tool_calls", 0)
                result.setdefault("tokens", metrics.get("output_chars", 0))
                result["errors"]     = metrics.get("errors_encountered", 0)

                raw_expectations = grading.get("expectations", [])
                for exp in raw_expectations:
                    if "text" not in exp or "passed" not in exp:
                        print(f"Warning: expectation in {grading_file} missing 'text' or 'passed'", file=sys.stderr)
                result["expectations"] = raw_expectations

                notes_summary = grading.get("user_notes_summary", {})
                result["notes"] = (
                    notes_summary.get("uncertainties", [])
                    + notes_summary.get("needs_review", [])
                    + notes_summary.get("workarounds", [])
                )
                results[config].append(result)

    return results


def _aggregate_results(results: dict) -> dict:
    run_summary: dict = {}
    configs = list(results.keys())

    for config in configs:
        runs = results.get(config, [])
        if not runs:
            run_summary[config] = {
                "pass_rate":    {"mean": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0},
                "time_seconds": {"mean": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0},
                "tokens":       {"mean": 0,   "stddev": 0,   "min": 0,   "max": 0},
            }
            continue
        run_summary[config] = {
            "pass_rate":    _calculate_stats([r["pass_rate"]      for r in runs]),
            "time_seconds": _calculate_stats([r["time_seconds"]   for r in runs]),
            "tokens":       _calculate_stats([r.get("tokens", 0)  for r in runs]),
        }

    primary  = run_summary.get(configs[0], {}) if configs else {}
    baseline = run_summary.get(configs[1], {}) if len(configs) >= 2 else {}

    run_summary["delta"] = {
        "pass_rate":    f"{primary.get('pass_rate',{}).get('mean',0) - baseline.get('pass_rate',{}).get('mean',0):+.2f}",
        "time_seconds": f"{primary.get('time_seconds',{}).get('mean',0) - baseline.get('time_seconds',{}).get('mean',0):+.1f}",
        "tokens":       f"{primary.get('tokens',{}).get('mean',0) - baseline.get('tokens',{}).get('mean',0):+.0f}",
    }
    return run_summary


def _generate_benchmark(benchmark_dir: Path, skill_name: str = "", skill_path_str: str = "") -> dict:
    results     = _load_run_results(benchmark_dir)
    run_summary = _aggregate_results(results)

    runs = []
    for config, config_runs in results.items():
        for result in config_runs:
            runs.append({
                "eval_id":       result["eval_id"],
                "configuration": config,
                "run_number":    result["run_number"],
                "result": {
                    "pass_rate":    result["pass_rate"],
                    "passed":       result["passed"],
                    "failed":       result["failed"],
                    "total":        result["total"],
                    "time_seconds": result["time_seconds"],
                    "tokens":       result.get("tokens", 0),
                    "tool_calls":   result.get("tool_calls", 0),
                    "errors":       result.get("errors", 0),
                },
                "expectations": result["expectations"],
                "notes":        result["notes"],
            })

    eval_ids = sorted({r["eval_id"] for cfg_runs in results.values() for r in cfg_runs})

    return {
        "metadata": {
            "skill_name":             skill_name or "<skill-name>",
            "skill_path":             skill_path_str or "<path/to/skill>",
            "executor_model":         "<model-name>",
            "analyzer_model":         "<model-name>",
            "timestamp":              datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "evals_run":              eval_ids,
            "runs_per_configuration": 3,
        },
        "runs":        runs,
        "run_summary": run_summary,
        "notes":       [],
    }


def _generate_benchmark_markdown(benchmark: dict) -> str:
    metadata    = benchmark["metadata"]
    run_summary = benchmark["run_summary"]
    configs     = [k for k in run_summary if k != "delta"]
    config_a    = configs[0] if len(configs) >= 1 else "config_a"
    config_b    = configs[1] if len(configs) >= 2 else "config_b"
    label_a     = config_a.replace("_", " ").title()
    label_b     = config_b.replace("_", " ").title()

    a = run_summary.get(config_a, {})
    b = run_summary.get(config_b, {})
    d = run_summary.get("delta", {})

    lines = [
        f"# Skill Benchmark: {metadata['skill_name']}",
        "",
        f"**Model**: {metadata['executor_model']}",
        f"**Date**: {metadata['timestamp']}",
        f"**Evals**: {', '.join(map(str, metadata['evals_run']))} ({metadata['runs_per_configuration']} runs each per configuration)",
        "",
        "## Summary",
        "",
        f"| Metric | {label_a} | {label_b} | Delta |",
        "|--------|------------|---------------|-------|",
        f"| Pass Rate | {a.get('pass_rate',{}).get('mean',0)*100:.0f}% ± {a.get('pass_rate',{}).get('stddev',0)*100:.0f}% | {b.get('pass_rate',{}).get('mean',0)*100:.0f}% ± {b.get('pass_rate',{}).get('stddev',0)*100:.0f}% | {d.get('pass_rate','—')} |",
        f"| Time | {a.get('time_seconds',{}).get('mean',0):.1f}s ± {a.get('time_seconds',{}).get('stddev',0):.1f}s | {b.get('time_seconds',{}).get('mean',0):.1f}s ± {b.get('time_seconds',{}).get('stddev',0):.1f}s | {d.get('time_seconds','—')}s |",
        f"| Tokens | {a.get('tokens',{}).get('mean',0):.0f} ± {a.get('tokens',{}).get('stddev',0):.0f} | {b.get('tokens',{}).get('mean',0):.0f} ± {b.get('tokens',{}).get('stddev',0):.0f} | {d.get('tokens','—')} |",
    ]

    if benchmark.get("notes"):
        lines += ["", "## Notes", ""]
        for note in benchmark["notes"]:
            lines.append(f"- {note}")

    return "\n".join(lines)


def cmd_benchmark(args: argparse.Namespace) -> int:
    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.exists():
        print(f"Directory not found: {benchmark_dir}", file=sys.stderr)
        return 1

    benchmark   = _generate_benchmark(benchmark_dir, args.skill_name, args.skill_path)
    output_json = Path(args.output) if args.output else (benchmark_dir / "benchmark.json")
    output_md   = output_json.with_suffix(".md")

    output_json.write_text(json.dumps(benchmark, indent=2))
    print(f"Generated: {output_json}")
    output_md.write_text(_generate_benchmark_markdown(benchmark))
    print(f"Generated: {output_md}")

    run_summary = benchmark["run_summary"]
    configs     = [k for k in run_summary if k != "delta"]
    delta       = run_summary.get("delta", {})
    print("\nSummary:")
    for config in configs:
        pr    = run_summary[config]["pass_rate"]["mean"]
        label = config.replace("_", " ").title()
        print(f"  {label}: {pr*100:.1f}% pass rate")
    print(f"  Delta:  {delta.get('pass_rate', '—')}")
    return 0


# =============================================================================
# ── review (interactive eval-output viewer) ───────────────────────────────────
# =============================================================================

_REVIEW_METADATA_FILES = {"transcript.md", "user_notes.md", "metrics.json"}
_REVIEW_TEXT_EXTS = {
    ".txt", ".md", ".json", ".csv", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".yaml", ".yml", ".xml", ".html", ".css", ".sh", ".rb", ".go", ".rs",
    ".java", ".c", ".cpp", ".h", ".hpp", ".sql", ".r", ".toml",
}
_REVIEW_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
_REVIEW_MIME_OVERRIDES = {
    ".svg":  "image/svg+xml",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _review_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _REVIEW_MIME_OVERRIDES:
        return _REVIEW_MIME_OVERRIDES[ext]
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _review_find_runs(workspace: Path) -> list[dict]:
    runs: list[dict] = []
    _review_find_recursive(workspace, workspace, runs)
    runs.sort(key=lambda r: (r.get("eval_id", float("inf")), r["id"]))
    return runs


def _review_find_recursive(root: Path, current: Path, runs: list[dict]) -> None:
    if not current.is_dir():
        return
    outputs_dir = current / "outputs"
    if outputs_dir.is_dir():
        run = _review_build_run(root, current)
        if run:
            runs.append(run)
        return
    skip = {"node_modules", ".git", "__pycache__", "skill", "inputs"}
    for child in sorted(current.iterdir()):
        if child.is_dir() and child.name not in skip:
            _review_find_recursive(root, child, runs)


def _review_build_run(root: Path, run_dir: Path) -> dict | None:
    prompt  = ""
    eval_id = None

    for candidate in [run_dir / "eval_metadata.json", run_dir.parent / "eval_metadata.json"]:
        if candidate.exists():
            try:
                metadata = json.loads(candidate.read_text())
                prompt   = metadata.get("prompt", "")
                eval_id  = metadata.get("eval_id")
            except (json.JSONDecodeError, OSError):
                pass
            if prompt:
                break

    if not prompt:
        for candidate in [run_dir / "transcript.md", run_dir / "outputs" / "transcript.md"]:
            if candidate.exists():
                try:
                    text  = candidate.read_text()
                    match = re.search(r"## Eval Prompt\n\n([\s\S]*?)(?=\n##|$)", text)
                    if match:
                        prompt = match.group(1).strip()
                except OSError:
                    pass
                if prompt:
                    break

    if not prompt:
        prompt = "(No prompt found)"

    run_id      = str(run_dir.relative_to(root)).replace("/", "-").replace("\\", "-")
    outputs_dir = run_dir / "outputs"
    output_files: list[dict] = []
    if outputs_dir.is_dir():
        for f in sorted(outputs_dir.iterdir()):
            if f.is_file() and f.name not in _REVIEW_METADATA_FILES:
                output_files.append(_review_embed_file(f))

    grading = None
    for candidate in [run_dir / "grading.json", run_dir.parent / "grading.json"]:
        if candidate.exists():
            try:
                grading = json.loads(candidate.read_text())
            except (json.JSONDecodeError, OSError):
                pass
            if grading:
                break

    return {"id": run_id, "prompt": prompt, "eval_id": eval_id, "outputs": output_files, "grading": grading}


def _review_embed_file(path: Path) -> dict:
    ext  = path.suffix.lower()
    mime = _review_mime(path)

    if ext in _REVIEW_TEXT_EXTS:
        try:
            content = path.read_text(errors="replace")
        except OSError:
            content = "(Error reading file)"
        return {"name": path.name, "type": "text", "content": content}

    if ext in _REVIEW_IMAGE_EXTS:
        try:
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            return {"name": path.name, "type": "error", "content": "(Error reading file)"}
        return {"name": path.name, "type": "image", "mime": mime, "data_uri": f"data:{mime};base64,{b64}"}

    if ext == ".pdf":
        try:
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            return {"name": path.name, "type": "error", "content": "(Error reading file)"}
        return {"name": path.name, "type": "pdf", "data_uri": f"data:{mime};base64,{b64}"}

    if ext == ".xlsx":
        try:
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            return {"name": path.name, "type": "error", "content": "(Error reading file)"}
        return {"name": path.name, "type": "xlsx", "data_b64": b64}

    # binary / unknown
    try:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return {"name": path.name, "type": "error", "content": "(Error reading file)"}
    return {"name": path.name, "type": "binary", "mime": mime, "data_uri": f"data:{mime};base64,{b64}"}


def _review_load_previous(workspace: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    feedback_map: dict[str, str] = {}
    feedback_path = workspace / "feedback.json"
    if feedback_path.exists():
        try:
            data = json.loads(feedback_path.read_text())
            feedback_map = {
                r["run_id"]: r["feedback"]
                for r in data.get("reviews", [])
                if r.get("feedback", "").strip()
            }
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    for run in _review_find_runs(workspace):
        result[run["id"]] = {
            "feedback": feedback_map.get(run["id"], ""),
            "outputs":  run.get("outputs", []),
        }
    for run_id, fb in feedback_map.items():
        if run_id not in result:
            result[run_id] = {"feedback": fb, "outputs": []}
    return result


def _review_generate_html(
    runs: list[dict],
    skill_name: str,
    previous: dict[str, dict] | None = None,
    benchmark: dict | None = None,
) -> str:
    # The viewer shell lives next to the original generate_review.py
    viewer_candidates = [
        Path(__file__).parent / "eval-viewer" / "viewer.html",
        Path(__file__).parent / "skill-creator" / "eval-viewer" / "viewer.html",
    ]
    template = ""
    for candidate in viewer_candidates:
        if candidate.exists():
            template = candidate.read_text()
            break
    if not template:
        # Graceful fallback: plain JSON dump
        return f"<pre>{html_lib.escape(json.dumps({'runs': runs}, indent=2))}</pre>"

    previous_feedback: dict[str, str] = {}
    previous_outputs:  dict[str, list] = {}
    if previous:
        for run_id, data in previous.items():
            if data.get("feedback"):
                previous_feedback[run_id] = data["feedback"]
            if data.get("outputs"):
                previous_outputs[run_id]  = data["outputs"]

    embedded: dict = {
        "skill_name":         skill_name,
        "runs":               runs,
        "previous_feedback":  previous_feedback,
        "previous_outputs":   previous_outputs,
    }
    if benchmark:
        embedded["benchmark"] = benchmark

    return template.replace("/*__EMBEDDED_DATA__*/", f"const EMBEDDED_DATA = {json.dumps(embedded)};")


def _kill_port(port: int) -> None:
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=5,
        )
        for pid_str in result.stdout.strip().split("\n"):
            if pid_str.strip():
                try:
                    os.kill(int(pid_str.strip()), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass
        if result.stdout.strip():
            time.sleep(0.5)
    except subprocess.TimeoutExpired:
        pass
    except FileNotFoundError:
        print("Note: lsof not found, cannot check if port is in use", file=sys.stderr)


class _ReviewHandler(BaseHTTPRequestHandler):
    def __init__(
        self,
        workspace:      Path,
        skill_name:     str,
        feedback_path:  Path,
        previous:       dict,
        benchmark_path: Path | None,
        *args,
        **kwargs,
    ):
        self.workspace      = workspace
        self.skill_name     = skill_name
        self.feedback_path  = feedback_path
        self.previous       = previous
        self.benchmark_path = benchmark_path
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            runs = _review_find_runs(self.workspace)
            benchmark = None
            if self.benchmark_path and self.benchmark_path.exists():
                try:
                    benchmark = json.loads(self.benchmark_path.read_text())
                except (json.JSONDecodeError, OSError):
                    pass
            content = _review_generate_html(runs, self.skill_name, self.previous, benchmark).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/api/feedback":
            data = self.feedback_path.read_bytes() if self.feedback_path.exists() else b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/api/feedback":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data = json.loads(body)
                if not isinstance(data, dict) or "reviews" not in data:
                    raise ValueError("Expected JSON object with 'reviews' key")
                self.feedback_path.write_text(json.dumps(data, indent=2) + "\n")
                resp = b'{"ok":true}'
                self.send_response(200)
            except (json.JSONDecodeError, OSError, ValueError) as e:
                resp = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        pass  # suppress request logs


def cmd_review(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        print(f"Error: {workspace} is not a directory", file=sys.stderr)
        return 1

    runs = _review_find_runs(workspace)
    if not runs:
        print(f"No runs found in {workspace}", file=sys.stderr)
        return 1

    skill_name    = args.skill_name or workspace.name.replace("-workspace", "")
    feedback_path = workspace / "feedback.json"

    previous: dict = {}
    if args.previous_workspace:
        previous = _review_load_previous(Path(args.previous_workspace).resolve())

    benchmark_path = Path(args.benchmark).resolve() if args.benchmark else None
    benchmark = None
    if benchmark_path and benchmark_path.exists():
        try:
            benchmark = json.loads(benchmark_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    if args.static:
        out = Path(args.static)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_review_generate_html(runs, skill_name, previous, benchmark))
        print(f"\n  Static viewer written to: {out}\n")
        return 0

    port = args.port
    _kill_port(port)
    handler = partial(_ReviewHandler, workspace, skill_name, feedback_path, previous, benchmark_path)
    try:
        server = HTTPServer(("127.0.0.1", port), handler)
    except OSError:
        server = HTTPServer(("127.0.0.1", 0), handler)
        port   = server.server_address[1]

    url = f"http://localhost:{port}"
    print(f"\n  Eval Viewer")
    print(f"  ─────────────────────────────────")
    print(f"  URL:       {url}")
    print(f"  Workspace: {workspace}")
    print(f"  Feedback:  {feedback_path}")
    if previous:
        print(f"  Previous:  {args.previous_workspace} ({len(previous)} runs)")
    if benchmark_path:
        print(f"  Benchmark: {benchmark_path}")
    print(f"\n  Press Ctrl+C to stop.\n")

    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()

    return 0


# =============================================================================
# ── CLI wiring ────────────────────────────────────────────────────────────────
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skill",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ── validate ──────────────────────────────────────────────────────────────
    p_val = sub.add_parser("validate", help="Validate a skill directory")
    p_val.add_argument("skill_path", metavar="SKILL_DIR", help="Path to skill directory")

    # ── package ───────────────────────────────────────────────────────────────
    p_pkg = sub.add_parser("package", help="Package a skill into a .skill zip file")
    p_pkg.add_argument("skill_path",  metavar="SKILL_DIR",  help="Path to skill directory")
    p_pkg.add_argument("output_dir",  metavar="OUTPUT_DIR", nargs="?", default=None,
                       help="Output directory (default: current directory)")

    # ── eval ──────────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Run trigger-detection evaluation")
    p_eval.add_argument("--eval-set",         required=True,  help="Path to eval set JSON file")
    p_eval.add_argument("--skill-path",       required=True,  help="Path to skill directory")
    p_eval.add_argument("--description",      default=None,   help="Override description to test")
    p_eval.add_argument("--num-workers",      type=int, default=10)
    p_eval.add_argument("--timeout",          type=int, default=30, help="Timeout per query (seconds)")
    p_eval.add_argument("--runs-per-query",   type=int, default=3)
    p_eval.add_argument("--trigger-threshold",type=float, default=0.5)
    p_eval.add_argument("--model",            default=None, help="claude -p model override")
    p_eval.add_argument("--verbose", "-v",    action="store_true")

    # ── improve ───────────────────────────────────────────────────────────────
    p_imp = sub.add_parser("improve", help="Improve a skill description (one Claude call)")
    p_imp.add_argument("--eval-results", required=True, help="Path to eval results JSON (from eval command)")
    p_imp.add_argument("--skill-path",   required=True, help="Path to skill directory")
    p_imp.add_argument("--history",      default=None,  help="Path to history JSON (previous attempts)")
    p_imp.add_argument("--model",        required=True, help="Model for improvement")
    p_imp.add_argument("--verbose", "-v", action="store_true")

    # ── loop ──────────────────────────────────────────────────────────────────
    p_loop = sub.add_parser("loop", help="Run the full eval+improve optimization loop")
    p_loop.add_argument("--eval-set",          required=True,   help="Path to eval set JSON file")
    p_loop.add_argument("--skill-path",        required=True,   help="Path to skill directory")
    p_loop.add_argument("--model",             required=True,   help="Model for improvement")
    p_loop.add_argument("--description",       default=None,    help="Override starting description")
    p_loop.add_argument("--num-workers",       type=int, default=10)
    p_loop.add_argument("--timeout",           type=int, default=30)
    p_loop.add_argument("--max-iterations",    type=int, default=5)
    p_loop.add_argument("--runs-per-query",    type=int, default=3)
    p_loop.add_argument("--trigger-threshold", type=float, default=0.5)
    p_loop.add_argument("--holdout",           type=float, default=0.4,
                        help="Fraction of eval set to hold out for testing (0 to disable)")
    p_loop.add_argument("--verbose", "-v",     action="store_true")
    p_loop.add_argument("--report",            default="auto",
                        help="HTML report path ('auto' = temp file, 'none' = disable)")
    p_loop.add_argument("--results-dir",       default=None,
                        help="Save outputs to a timestamped subdirectory here")

    # ── report ────────────────────────────────────────────────────────────────
    p_rep = sub.add_parser("report", help="Generate HTML report from loop output JSON")
    p_rep.add_argument("input",        help="Path to loop output JSON (or '-' for stdin)")
    p_rep.add_argument("-o", "--output", default=None, help="Output HTML file (default: stdout)")
    p_rep.add_argument("--skill-name",  default="",   help="Skill name for the report title")

    # ── benchmark ─────────────────────────────────────────────────────────────
    p_bench = sub.add_parser("benchmark", help="Aggregate grading.json files into benchmark stats")
    p_bench.add_argument("benchmark_dir", metavar="BENCHMARK_DIR", type=str,
                         help="Path to benchmark directory")
    p_bench.add_argument("--skill-name", default="",  help="Skill name")
    p_bench.add_argument("--skill-path", default="",  help="Skill path (for metadata)")
    p_bench.add_argument("--output", "-o", default=None, help="Output path for benchmark.json")

    # ── review ────────────────────────────────────────────────────────────────
    p_rev = sub.add_parser("review", help="Serve an interactive eval-output review viewer")
    p_rev.add_argument("workspace",          metavar="WORKSPACE_DIR", help="Path to workspace directory")
    p_rev.add_argument("--port", "-p",       type=int, default=3117)
    p_rev.add_argument("--skill-name", "-n", default=None)
    p_rev.add_argument("--previous-workspace", default=None,
                       help="Previous iteration's workspace (shows old outputs/feedback)")
    p_rev.add_argument("--benchmark",        default=None, help="Path to benchmark.json")
    p_rev.add_argument("--static", "-s",     default=None,
                       help="Write standalone HTML here instead of starting server")

    return parser


_COMMANDS = {
    "validate":  cmd_validate,
    "package":   cmd_package,
    "eval":      cmd_eval,
    "improve":   cmd_improve,
    "loop":      cmd_loop,
    "report":    cmd_report,
    "benchmark": cmd_benchmark,
    "review":    cmd_review,
}


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    sys.exit(_COMMANDS[args.command](args))


if __name__ == "__main__":
    main()
