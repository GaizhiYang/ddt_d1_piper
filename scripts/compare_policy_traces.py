#!/usr/bin/env python3
"""Compare two policy-rate sim2sim traces column by column.

The MuJoCo controller writes a CSV with ``frame_*`` (82 values),
``history_*`` (246 values), ``action_*``, ``tau_*``, ``q_*`` and ``dq_*``
columns when ``--policy-trace-path`` is supplied.  IsaacLab recordings can be
converted to the same column names and compared without depending on either
simulator's Python API.

The default comparison is by sample index.  Use ``--align-time`` when the two
files have different timestamps but represent the same policy-rate samples;
the candidate is linearly interpolated at the reference timestamps.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PREFIXES = ("frame_", "history_", "action_", "tau_", "q_", "dq_")


def _read_trace(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        names = list(reader.fieldnames)
        required = [name for name in ("time",) if name not in names]
        if required:
            raise ValueError(f"{path} is missing required columns: {required}")
        selected = [name for name in names if name.startswith(PREFIXES)]
        if not selected:
            raise ValueError(f"{path} has no frame/history/action trace columns")
        rows: list[list[float]] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                rows.append([float(row[name]) for name in ("time", *selected)])
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"non-numeric or incomplete row at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"{path} is empty")
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path} contains NaN or infinity")
    if np.any(np.diff(matrix[:, 0]) < 0):
        raise ValueError(f"{path} time column is not non-decreasing")
    return matrix[:, 0], {name: matrix[:, i + 1] for i, name in enumerate(selected)}


def _ordered_common(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> list[str]:
    common = [name for name in reference if name in candidate]
    if not common:
        raise ValueError("the two traces have no columns in common")
    missing = [name for name in reference if name not in candidate]
    if missing:
        raise ValueError(f"candidate trace is missing {len(missing)} reference columns; first={missing[:5]}")
    return common


def _compare(reference_time: np.ndarray, reference: dict[str, np.ndarray],
             candidate_time: np.ndarray, candidate: dict[str, np.ndarray],
             align_time: bool) -> dict[str, object]:
    names = _ordered_common(reference, candidate)
    if align_time:
        if candidate_time[0] > reference_time[0] or candidate_time[-1] < reference_time[-1]:
            raise ValueError("candidate time range does not cover reference time range")
        candidate_values = {
            name: np.interp(reference_time, candidate_time, candidate[name])
            for name in names
        }
        reference_values = {name: reference[name] for name in names}
        compared_samples = int(reference_time.size)
    else:
        if reference_time.size != candidate_time.size:
            raise ValueError(
                f"trace lengths differ ({reference_time.size} vs {candidate_time.size}); use --align-time or trim them"
            )
        reference_values = {name: reference[name] for name in names}
        candidate_values = {name: candidate[name] for name in names}
        compared_samples = int(reference_time.size)

    errors: dict[str, dict[str, float]] = {}
    all_error = []
    for name in names:
        delta = candidate_values[name] - reference_values[name]
        abs_delta = np.abs(delta)
        all_error.append(abs_delta)
        errors[name] = {
            "max_abs": float(np.max(abs_delta)),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "mean_abs": float(np.mean(abs_delta)),
        }
    stacked = np.concatenate(all_error)
    return {
        "reference_samples": int(reference_time.size),
        "candidate_samples": int(candidate_time.size),
        "compared_samples": compared_samples,
        "align_time": bool(align_time),
        "columns": len(names),
        "overall_max_abs": float(np.max(stacked)),
        "overall_rmse": float(np.sqrt(np.mean(stacked * stacked))),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="IsaacLab/reference trace CSV")
    parser.add_argument("candidate", type=Path, help="MuJoCo/candidate trace CSV")
    parser.add_argument("--align-time", action="store_true",
                        help="linearly interpolate candidate values at reference timestamps")
    parser.add_argument("--max-abs", type=float, default=None,
                        help="optional pass/fail threshold for every compared scalar")
    parser.add_argument("--report", type=Path, default=None,
                        help="optional JSON report path")
    args = parser.parse_args()
    if args.max_abs is not None and (not np.isfinite(args.max_abs) or args.max_abs < 0):
        raise SystemExit("--max-abs must be a finite non-negative number")
    try:
        reference_time, reference = _read_trace(args.reference)
        candidate_time, candidate = _read_trace(args.candidate)
        report = _compare(reference_time, reference, candidate_time, candidate, args.align_time)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.max_abs is not None:
        report["max_abs_threshold"] = float(args.max_abs)
        report["pass"] = bool(report["overall_max_abs"] <= args.max_abs)
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report.get("pass", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
