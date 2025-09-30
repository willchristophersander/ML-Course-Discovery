#!/usr/bin/env python3
"""Backwards compatible wrapper around the consolidated CollegeTransfer client."""

import asyncio
import sys
from pathlib import Path

# Ensure the src package is importable when the script is launched from repo root
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.college_transfer_search import search_university_courses


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python simplecollegetransfercoursesearch.py 'University Name'")
        print("Example: python simplecollegetransfercoursesearch.py 'University of Vermont'")
        sys.exit(1)

    university_name = sys.argv[1]
    asyncio.run(search_university_courses(university_name))


if __name__ == "__main__":
    main()
