#!/usr/bin/env python3
"""Read-only ABI probe for a DDT ``libtita_robot.so``.

The D1 vendor library is a C++ ABI and cannot safely be guessed from Python.
This probe intentionally uses only ``file``/``nm`` output: it never loads the
shared object, constructs ``CanfdApi`` or opens a CAN interface.  It is useful
on the Jetson before injecting a binding into ``D1Backend``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


# ``nm -C`` prints the constructor's third argument as either ``std::string``
# or its fully expanded ``std::__cxx11::basic_string`` spelling.
REQUIRED_CONSTRUCTORS = (
    "can_device::CanfdApi::CanfdApi(unsigned long, unsigned long,",
    "can_device::CanfdApi::CanfdApi(unsigned long, unsigned long, std::",
)
SEND_SYMBOLS = {
    "send_all": "can_device::CanfdApi::send_motors_can(",
    "send_leg": "can_device::CanfdApi::send_leg_motors_can(",
}


def _command(executable: str, *args: str) -> str:
    try:
        result = subprocess.run(
            [executable, *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"required executable not found: {executable}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{executable} failed: {exc.stdout.strip()}") from exc
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument(
        "--expected-arch",
        choices=("aarch64", "x86-64"),
        default=None,
        help="optionally require the Jetson (aarch64) or workstation (x86-64) build",
    )
    args = parser.parse_args()
    if not args.library.is_file():
        raise SystemExit(f"library not found: {args.library}")
    if shutil.which("file") is None or shutil.which("nm") is None:
        raise SystemExit("both 'file' and 'nm' are required")

    file_output = _command("file", str(args.library)).strip()
    nm_output = _command("nm", "-D", "-C", str(args.library))
    symbols = [line.strip() for line in nm_output.splitlines() if line.strip()]
    constructor = any(
        any(required in symbol for required in REQUIRED_CONSTRUCTORS)
        for symbol in symbols
    )
    send = {name: any(pattern in symbol for symbol in symbols) for name, pattern in SEND_SYMBOLS.items()}
    if not constructor:
        raise SystemExit("ABI mismatch: CanfdApi constructor symbol was not found")
    if not any(send.values()):
        raise SystemExit("ABI mismatch: neither send_motors_can nor send_leg_motors_can was found")
    if args.expected_arch == "aarch64" and "ARM aarch64" not in file_output:
        raise SystemExit(f"architecture mismatch: expected aarch64, got: {file_output}")
    if args.expected_arch == "x86-64" and "x86-64" not in file_output:
        raise SystemExit(f"architecture mismatch: expected x86-64, got: {file_output}")

    report = {
        "library": str(args.library),
        "file": file_output,
        "canfd_constructor": constructor,
        "send_motors_can": send["send_all"],
        "send_leg_motors_can": send["send_leg"],
        "note": "C++ struct layout and std::vector ABI still require a compiled adapter test",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
