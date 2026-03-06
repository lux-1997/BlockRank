#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_beir_eval.sh [config_yaml] [checkpoint_path] [dataset_name ...]
# - config_yaml: YAML with a top-level `datasets` list (defaults to configs/eval_mistral_beir.yaml)
# - checkpoint_path: model checkpoint to load (defaults to the cached BlockRank checkpoint)
# - dataset_name: optional whitespace-separated list to limit which datasets to run (or comment them out in YAML)

CONFIG=${1:-configs/eval_mistral_beir.yaml}
if [ $# -gt 0 ]; then shift; fi
CKPT=${1:-/code/in_context_retrieval/BlockRank/outputs/blockrank-10p-msmarco-mistral-7b-only-copynet}
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
    # Use attention-based evaluation (BlockRank attention scores).
    cmd = ["python", "scripts/eval_attn.py", "--config", tmp_path, "--checkpoint", ckpt_path]
    env = os.environ.copy()
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("WANDB_API_KEY", "offline")
    subprocess.run(cmd, check=True, env=env)

    os.remove(tmp_path)
PY
