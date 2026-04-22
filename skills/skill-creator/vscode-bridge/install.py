#!/usr/bin/env python3
"""Install the VS Code skill-bridge extension into the user's VS Code extensions directory.

Usage:
    python install.py
"""
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    bridge_src = Path(__file__).parent

    # Resolve destination: ~/.vscode/extensions/ on all platforms
    if sys.platform == "win32":
        base = Path(os.environ.get("USERPROFILE", Path.home()))
    else:
        base = Path.home()
    ext_dir = base / ".vscode" / "extensions"
    dest = ext_dir / "skill-bridge-0.1.0"

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(bridge_src, dest, ignore=shutil.ignore_patterns("install.py"))

    print(f"Installed skill-bridge to: {dest}")
    print()
    print("Next steps:")
    print("  1. Reload VS Code (Cmd/Ctrl+Shift+P → 'Reload Window')")
    print("  2. Ensure GitHub Copilot is signed in")
    print("  3. Look for '$(broadcast) Skill Bridge :7777' in the status bar")
    print("  4. Run:  python skill.py bridge-status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
