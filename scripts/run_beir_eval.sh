#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_beir_eval.sh [config_yaml] [checkpoint_path] [dataset_name ...]
# - config_yaml: YAML with a top-level `datasets` list (defaults to configs/eval_mistral_beir.yaml)
# - checkpoint_path: model checkpoint to load (defaults to the cached BlockRank checkpoint)
# - dataset_name: optional whitespace-separated list to limit which datasets to run (or comment them out in YAML)
#
# Multi-GPU:
# - Use CUDA_VISIBLE_DEVICES to choose GPUs (e.g., CUDA_VISIBLE_DEVICES=4,5,6,7 ...)
# - Optional override: EVAL_NPROC_PER_NODE=4
# - Optional port: EVAL_MASTER_PORT=29547

CONFIG=${1:-configs/eval_mistral_beir.yaml}
if [ $# -gt 0 ]; then shift; fi
CKPT=${1:-/data/mengrui/test_lx/OLM2Vec/BlockRank/outputs/blockrank-10p-msmarco-mistral-7b-only-copynet-segment-20-end}
if [ $# -gt 0 ]; then shift; fi

# Disable W&B network/login; override by exporting WANDB_DISABLED=false or WANDB_MODE=online if you really want logging.
export WANDB_DISABLED=${WANDB_DISABLED:-true}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_API_KEY=${WANDB_API_KEY:-"offline"}

python - <<'PY' "$CONFIG" "$CKPT" "$@"
import copy
import os
import subprocess
import sys
import tempfile

import yaml

cfg_path = sys.argv[1]
ckpt_path = sys.argv[2]
requested = set(sys.argv[3:])

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f) or {}

datasets = cfg.get("datasets", [])
if not datasets:
    raise SystemExit("No datasets found under the `datasets` key in the config.")

base_cfg = copy.deepcopy(cfg)
base_cfg.pop("datasets", None)

for ds in datasets:
    name = ds["name"]
    if requested and name not in requested:
        continue

    run_cfg = copy.deepcopy(base_cfg)
    run_cfg["data"] = {
        **copy.deepcopy(run_cfg.get("data", {})),
        "data_path": ds["data_path"],
        "qrels_path": ds.get("qrels_path"),
    }
    run_cfg["eval"] = {
        **copy.deepcopy(run_cfg.get("eval", {})),
        "output_dir": ds["output_dir"],
    }

    fd, tmp_path = tempfile.mkstemp(prefix=f"eval_{name}_", suffix=".yaml")
    with os.fdopen(fd, "w") as tmp:
        yaml.safe_dump(run_cfg, tmp, sort_keys=False)

    print(f"\n=== Running {name} ===")
    env = os.environ.copy()
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("WANDB_API_KEY", "offline")

    # Multi-GPU support:
    # - If EVAL_NPROC_PER_NODE is set, use it.
    # - Else infer process count from CUDA_VISIBLE_DEVICES.
    nproc_env = env.get("EVAL_NPROC_PER_NODE", "").strip()
    if nproc_env:
        nproc = int(nproc_env)
    else:
        cuda_visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cuda_visible and cuda_visible not in {"-1", "none", "None"}:
            nproc = len([x for x in cuda_visible.split(",") if x.strip()])
        else:
            nproc = 1
    nproc = max(1, nproc)

    if nproc > 1:
        master_port = env.get("EVAL_MASTER_PORT", env.get("MASTER_PORT", "29547"))
        cmd = [
            "torchrun",
            "--nproc_per_node", str(nproc),
            "--master_port", str(master_port),
            "scripts/eval_attn.py",
            "--config", tmp_path,
            "--checkpoint", ckpt_path,
        ]
    else:
        cmd = ["python", "scripts/eval_attn.py", "--config", tmp_path, "--checkpoint", ckpt_path]

    print("Launcher:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    os.remove(tmp_path)
PY
