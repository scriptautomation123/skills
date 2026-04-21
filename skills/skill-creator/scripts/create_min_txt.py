#!/usr/bin/env python3
"""Create the smallest possible .txt file (empty file)."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create an empty .txt file")
    parser.add_argument(
        "--output",
        default="skills/skill-creator/assets/min.txt",
        help="Path to the txt file to create (default: skills/skill-creator/assets/min.txt)",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("")
    print(f"Created: {output.resolve()}")


if __name__ == "__main__":
    main()
