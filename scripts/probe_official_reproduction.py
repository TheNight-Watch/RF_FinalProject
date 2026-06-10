#!/usr/bin/env python3
"""Record official Diffusion Policy reproduction availability.

The probe is deliberately light: it checks repository metadata and the official
Push-T checkpoint HTTP headers without downloading the 995 MiB checkpoint.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "official_reproduction_status.json"
CKPT = "https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt"
REPO = "https://github.com/real-stanford/diffusion_policy.git"


def run(cmd: list[str], timeout: int) -> dict[str, object]:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "elapsed_sec": round(time.time() - t0, 3),
            "output": p.stdout[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": "timeout",
            "elapsed_sec": round(time.time() - t0, 3),
            "output": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
        }


def main() -> None:
    status = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_repo": REPO,
        "official_checkpoint": CKPT,
        "repo_head_probe": run(["git", "-c", "http.proxy=", "-c", "https.proxy=", "ls-remote", REPO, "HEAD"], 45),
        "checkpoint_header_probe": run(["curl", "-L", "--noproxy", "*", "-I", "--max-time", "45", CKPT], 50),
        "official_eval_completed": False,
        "reason": "Official checkpoint evaluation requires cloning the upstream repository and downloading a 995 MiB checkpoint; clone/download were unreliable in this interactive session.",
        "local_surrogate_results": str(ROOT / "results" / "summary_results.csv"),
        "official_eval_script": str(ROOT / "scripts" / "run_official_diffusion_policy_eval.sh"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(status, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
