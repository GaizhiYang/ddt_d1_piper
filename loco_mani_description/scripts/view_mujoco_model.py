#!/usr/bin/env python3
"""Open a MuJoCo D1 + Piper-L model without loading a controller or policy.

This is intentionally a model/URDF inspection tool.  It only parses the MJCF,
creates ``MjData`` and opens the standard passive viewer; no ONNX Runtime,
ROS controller, CAN interface or actuator command is created.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco


def main() -> None:
    here = Path(__file__).resolve()
    default_scene = here.parents[1] / "mujoco" / "scene.xml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-path", default=str(default_scene),
                        help="MJCF/scene XML to inspect")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="auto-close after N wall-clock seconds; 0 keeps the viewer open")
    args = parser.parse_args()
    if args.duration < 0:
        raise ValueError("duration must be non-negative")

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(
        f"[loco_mani] model-only viewer: {args.xml_path}\n"
        f"[loco_mani] nq={model.nq} nv={model.nv} nu={model.nu} "
        f"nbody={model.nbody} ngeom={model.ngeom} njnt={model.njnt}\n"
        "[loco_mani] no policy/controller/CAN is loaded"
    )

    from mujoco import viewer
    handle = viewer.launch_passive(model, data)
    started = time.monotonic()
    try:
        while handle.is_running():
            # No dynamics stepping is performed: this command is for static
            # URDF/MJCF inspection.  The viewer still refreshes the scene.
            handle.sync()
            if args.duration > 0 and time.monotonic() - started >= args.duration:
                break
            time.sleep(0.02)
    finally:
        handle.close()
        # Give the daemon UI thread a moment to release its GL context before
        # Python exits, avoiding a GLFW cleanup race on some X11 drivers.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                if not handle.is_running():
                    break
            except Exception:
                break
            time.sleep(0.01)


if __name__ == "__main__":
    main()
