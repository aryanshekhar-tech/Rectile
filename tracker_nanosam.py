#!/usr/bin/env python3
"""Same public entry point; only the dashboard presentation has been replaced."""
from pathlib import Path
import sys
from uav_gui.cli import parse_args
from uav_gui.bootstrap import activate_local_dependencies


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parent
    activate_local_dependencies(root)
    try:
        from uav_gui.app import run
        return run(args, root)
    except ImportError as exc:
        if (exc.name or '').split('.')[0] in ('PySide6', 'shiboken6'):
            print('The new GUI dependency is not installed for this Python interpreter.', file=sys.stderr)
            print(f'One-time setup: {sys.executable} "{root / "setup_gui.py"}"', file=sys.stderr)
            print('Then use your original video or camera command unchanged.', file=sys.stderr)
            return 2
        raise


if __name__ == '__main__':
    raise SystemExit(main())
